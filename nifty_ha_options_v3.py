#!/usr/bin/env python3
"""
Nifty 50 Options Trader — Heikin-Ashi Strategy  v4
====================================================
Single-strategy script. No multi-strategy bloat.

STRATEGY RECAP:
  1. No trades before 10:15 AM
  2. Wait until 10:30 AM (so the first 15-min bar of the second hour is closed)
  3. Determine INITIAL bias at 10:30 by comparing two 15-minute HA candles:
       • Last  15-min bar of Hour 1  →  10:00–10:15  (H1_last)
       • First 15-min bar of Hour 2  →  10:15–10:30  (H2_first)
       If H2_first HA-close > H1_last HA-close  →  BULLISH  (look for CE only)
       If H2_first HA-close < H1_last HA-close  →  BEARISH  (look for PE only)
       If equal                                  →  no bias  (skip, retry next bar)
  4. After 10:30 the bias is refreshed every hour from the closing 1H HA candle
     (same behaviour as before — this only changes the very first bias decision)
  5. On the 15-minute chart, scan for an EARLY REVERSAL pattern:
       Bullish (CE):  RED RED → First GREEN HA + close above prev HA high + small lower shadow
       Bearish (PE):  GREEN GREEN → First RED HA + close below prev HA low + small upper shadow
  6. Exit at SL / target / time (15:15) or end of day

  ** v2/v3 IMPROVEMENTS (based on 29 May 2026 analysis) **

  1. Multi-Timeframe Confirmation (1H trend filter):
     Before allowing any 15m entry, verify the 1H HA trend is aligned with
     the current bias. Filters early fakeouts (e.g. weak signals pre-12:30).

  2. Adaptive Structure Filter:
     bearish_structure_ok / bullish_structure_ok now accept a bias_strength
     parameter. After 12:00 PM, during a strong directional trend, the filter
     is relaxed to allow flat-low (or flat-high) structure, capturing strong
     continuation moves that the strict filter would miss.

  3. Dynamic SL Order Modification:
     When the HA trail ratchets to a better level, the old SL order on the
     exchange is cancelled and a new SL-LIMIT order is placed at the updated
     level. Previous behaviour was memory-only.

  4. Volatility / Range Filter:
     Entry is blocked when the current 15m candle's range is below
     MIN_RANGE_RATIO × average 15m range. Prevents entries on low-volatility
     fake reversals.

  5. Preferred Entry Time Window:
     Before 11:00 AM or after 14:00 PM, entries require a stricter
     structure/confirmation check (BIAS_STRENGTH = "strict").
     Between 11:00–14:00 the normal or relaxed rules apply.

  6. Daily P&L Circuit-Breaker:
     If DAILY_PNL <= -MAX_DAILY_LOSS_ABS, no new entries are allowed for
     the rest of the day.

  7. Max Trades Per Day:
     MAX_TRADES_PER_DAY limits total entries to avoid overtrading.

  8. Bias Strength Scoring:
     Initial bias is graded STRONG / NORMAL / WEAK based on the HA-close
     delta. WEAK biases are skipped entirely.

  9. Doji Filter (carried from v1):
     A Heikin-Ashi candle with body < MIN_BODY_RATIO of total range is
     treated as a doji and ignored for reversal signals.

DATA:
  Upstox does NOT natively serve 15min or 1H candles via its v2 API.
  We build them ourselves:
    • 1-min candles  → fetch from /historical-candle/intraday/NSE_INDEX|Nifty 50/1minute
    • Resample       → pandas resample to 15T and 1h

USAGE:
  python nifty_ha_options_v2.py

REQUIREMENTS:
  pip install requests pandas scipy
"""

import os
import sys
import time
import csv
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (edit these)
# ─────────────────────────────────────────────────────────────────────────────

ACCESS_TOKEN = os.environ.get("UPSTOX_TOKEN")
if not ACCESS_TOKEN:
    raise ValueError(
        "\n❌  UPSTOX_TOKEN environment variable is not set.\n"
        "    Export it before running:\n"
        "      export UPSTOX_TOKEN='your_token_here'\n"
        "    or pass it inline:\n"
        "      UPSTOX_TOKEN='your_token_here' python nifty_ha_options_v3.py"
    )

# Upstox instrument keys
NIFTY_INDEX_KEY  = "NSE_INDEX|Nifty 50"   # used to fetch 1-min candles
NIFTY_OPTION_KEY = "NSE_INDEX|Nifty 50"   # used for option chain lookup

BASE_URL = "https://api.upstox.com/v2"

# Trading parameters
ORDER_QUANTITY     = 1          # lots
ORDER_PRODUCT      = "I"        # Intraday (was "D" – fixed for options)
STOPLOSS_PCT       = 15.0       # SL as % of option premium
TARGET_MULTIPLIER  = 2.0        # target = risk × multiplier
NO_NEW_ENTRY_AFTER = "15:15"    # no fresh entries after this time
MARKET_CLOSE_TIME  = "15:30"
SCAN_INTERVAL_SECS = 30         # how often the main loop wakes up

# HA reversal detection parameters
# Lower-shadow filter for bullish reversal:  shadow_pct = lower_shadow / candle_range
SMALL_LOWER_SHADOW_MAX = 0.35   # lower shadow ≤ 35% of range  (bullish reversal)
SMALL_UPPER_SHADOW_MAX = 0.35   # upper shadow ≤ 35% of range  (bearish reversal)

# HA trailing-stop buffer
# CE: trail = prev_ha_low  + TRAIL_BUFFER_PCT × candle_range   (stop sits above raw HA low)
# PE: trail = prev_ha_high − TRAIL_BUFFER_PCT × candle_range   (stop sits below raw HA high)
# Set to 0.0 to trail exactly at the HA boundary (original behaviour).
# 0.10 = 10% of the last HA candle's range added as a cushion.
TRAIL_BUFFER_PCT = 0.10

# ** NEW: Doji filter **
# Only consider a HA candle "meaningful" if its body is at least this fraction
# of the candle's full range (high - low). Candles below this threshold are
# treated as doji and will break any reversal pattern sequence.
MIN_BODY_RATIO = 0.20   # e.g. 20%  (0 = disable filter)

# Enable automated order placement (set False for signal-only mode)
ENABLE_AUTO_TRADING = True

# ── v2: Multi-Timeframe Confirmation ─────────────────────────────────────────
# Require the last 1H HA candle to be trending in the same direction as BIAS
# before allowing a 15m entry signal.  Reduces fakeout entries on weak setups.
ENABLE_1H_TREND_FILTER = True

# ── v2: Volatility / Range Filter ────────────────────────────────────────────
# Skip entry if the last closed 15m candle range < MIN_RANGE_RATIO × average
# 15m range over the lookback window.  Set to 0.0 to disable.
MIN_RANGE_RATIO       = 0.40   # 40% of average range
RANGE_LOOKBACK_BARS   = 10     # how many bars to compute average over

# ── v2: Daily circuit-breaker ────────────────────────────────────────────────
# Expressed as a positive number (absolute rupees lost).
# v3: Renamed to MAX_DAILY_LOSS_ABS so it is unambiguously positive.
# The check in scan_15m_for_entry is: if DAILY_PNL <= -MAX_DAILY_LOSS_ABS
MAX_DAILY_LOSS_ABS = 5000.0    # stop new entries if DAILY_PNL drops below -5000

# ── v2: Max trades per day ────────────────────────────────────────────────────
MAX_TRADES_PER_DAY = 3

# ── v2: Bias strength thresholds ─────────────────────────────────────────────
# Initial bias is graded on HA-close delta (H2_first − H1_last in points).
# Trades are skipped on WEAK bias entirely.
BIAS_STRONG_MIN_DELTA = 5.0    # delta >= this → STRONG bias
BIAS_WEAK_MAX_DELTA   = 1.5    # delta <  this → WEAK  bias (skip trading)

# ── v4: 15m intra-hour bias flip ──────────────────────────────────────────────
# When the market makes a sustained move AGAINST the current bias, flip it.
# Flip requires ALL of:
#   1. BIAS_FLIP_CONSEC_CANDLES consecutive opposite-colour 15m HA bars
#   2. Latest HA close moved >= BIAS_FLIP_MIN_MOVE pts against the bias
#   3. >= BIAS_FLIP_COOLDOWN_MINS since the last flip (prevents thrashing)
# Set ENABLE_BIAS_FLIP = False to restore original behaviour.
ENABLE_BIAS_FLIP         = True
BIAS_FLIP_CONSEC_CANDLES = 3      # 3 consecutive opposite HA bars
BIAS_FLIP_MIN_MOVE       = 20.0   # HA close must move >= 20 pts against bias
BIAS_FLIP_COOLDOWN_MINS  = 30     # minimum minutes between flips

# ── v2: Preferred entry window ────────────────────────────────────────────────
# Outside [ENTRY_WINDOW_START, ENTRY_WINDOW_END], the structure filter is
# stricter (requires both lower-low AND lower-low-2 for bearish, etc.).
ENTRY_WINDOW_START = "11:00"
ENTRY_WINDOW_END   = "14:00"

# ── v3: Option chain cache ───────────────────────────────────────────────────
# The option chain endpoint is the heaviest API call. Cache it for this many
# seconds between entry attempts to reduce latency and rate-limit risk.
OPTION_CHAIN_CACHE_TTL = 300   # 5 minutes

# Logging
LOG_FILE = "nifty_ha_strategy.txt"
CSV_FILE = "nifty_ha_trades.csv"

# Test / debug
DEBUG_MODE = True

# ─────────────────────────────────────────────────────────────────────────────
# DATA AVAILABILITY LAG (fix for 10:15 issue)
# Upstox intraday 1‑minute candles become available about 60‑120 seconds
# after the minute ends. We wait an extra 90 seconds before considering a
# 1‑hour candle "closed".
# ─────────────────────────────────────────────────────────────────────────────
DATA_LAG_SECONDS = 90

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────────────

BIAS            = None   # "BULLISH" | "BEARISH" | None
BIAS_SET_AT     = None   # datetime when bias was last set/updated
BIAS_STRENGTH   = "NORMAL"  # "STRONG" | "NORMAL" | "WEAK"
ACTIVE_POSITION = {}     # tracks the open option position
DAILY_PNL       = 0.0
TRADES_TODAY    = 0      # count of completed trades this session

BIAS_OVERRIDE_ACTIVE   = False  # track if bias has been overridden for the current hour
PROCESSED_BIAS_CANDLES = set()  # 1H candle start times already used for bias
BIAS_OVERRIDE_DONE_FOR = set()  # candle_close_time values for which override already ran
_INITIAL_BIAS_SET      = False  # True once the 15-min comparison has fired for the day
_LAST_BIAS_FLIP_TIME   = None   # datetime of last intra-hour bias flip (cooldown guard)
_BIAS_SET_HA_CLOSE     = None   # HA close at the moment bias was last set (for move check)

# ── v3: Option chain cache ────────────────────────────────────────────────────
OPTION_CHAIN_CACHE: dict = {}       # {option_type: [contracts]}
LAST_CHAIN_FETCH:   datetime | None = None

# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _headers():
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}",
    }

def _order_headers():
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}",
    }

# ─────────────────────────────────────────────────────────────────────────────
# CANDLE FETCHING  (1-minute intraday → resample)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_1min_intraday() -> Optional[pd.DataFrame]:
    """
    Fetch today's 1-minute intraday candles for Nifty 50 index.
    Upstox /historical-candle/intraday/{key}/1minute returns today's bars.
    """
    url = f"{BASE_URL}/historical-candle/intraday/{NIFTY_INDEX_KEY}/1minute"
    try:
        resp = requests.get(url, headers=_headers(), timeout=20)
        if resp.status_code == 200:
            candles = resp.json().get("data", {}).get("candles", [])
            if not candles:
                return None
            df = pd.DataFrame(candles,
                              columns=["timestamp", "open", "high", "low",
                                       "close", "volume", "oi"])
            df["datetime"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("datetime").reset_index(drop=True)
            return df
        else:
            if DEBUG_MODE:
                print(f"⚠️  1min fetch HTTP {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚠️  1min fetch error: {e}")
        return None


def resample_to(df: pd.DataFrame, rule: str, offset: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Resample 1-minute DataFrame to a higher timeframe.

    Normalizes all timestamps to tz-naive minute-floored values so that
    exact pd.Timestamp equality lookups work reliably (no sub-second or
    timezone offset surprises from the Upstox API response).
    """
    if df is None or df.empty:
        return None
    src = df.copy()
    # Strip tz and floor to minute before resampling
    src["datetime"] = pd.to_datetime(src["datetime"]).dt.tz_localize(None).dt.floor("min")
    df_idx = src.set_index("datetime")[["open", "high", "low", "close", "volume"]]
    kwargs = {"offset": offset} if offset else {}
    resampled = df_idx.resample(rule, **kwargs).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    resampled = resampled.reset_index()
    # Normalize output: tz-naive, floored to minute, no sub-second noise
    resampled["datetime"] = pd.to_datetime(resampled["datetime"]).dt.tz_localize(None).dt.floor("min")
    return resampled


def get_candles():
    """
    Returns (df_15m, df_1h) — DataFrames of 15-minute and 1-hour candles
    built by resampling today's 1-minute intraday data.
    """
    df1 = fetch_1min_intraday()
    if df1 is None:
        return None, None
    df_15m = resample_to(df1, "15min")
    df_1h  = resample_to(df1, "1h", offset="15min")  # align to 09:15 NSE open
    return df_15m, df_1h

# ─────────────────────────────────────────────────────────────────────────────
# HEIKIN-ASHI CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_ha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ha_open, ha_high, ha_low, ha_close, ha_color columns to a copy of df.
    Requires columns: open, high, low, close.

    ha_open is a recursive average: ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    This is an IIR filter with coefficient 0.5, computed in O(n) via scipy.signal.lfilter
    instead of a slow Python loop.
    """
    from scipy.signal import lfilter

    df = df.copy().reset_index(drop=True)

    # ── ha_close: vectorised, no loop needed ─────────────────────────────────
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

    # ── ha_open: IIR recurrence  ha_open[i] = 0.5*ha_open[i-1] + 0.5*ha_close[i-1]
    # Rewrite as:  ha_open[i] = 0.5*(ha_open[i-1] + ha_close[i-1])
    # In lfilter terms (Direct Form II):
    #   b = [0, 0.5],  a = [1, -0.5]
    # Initial condition set so ha_open[0] = (open[0] + close[0]) / 2
    seed = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    # zi represents the filter memory; zi[0] seeds the output to `seed`
    zi = np.array([seed])
    ha_open_arr, _ = lfilter([0, 0.5], [1, -0.5], ha_close.to_numpy(), zi=zi)
    ha_open = pd.Series(ha_open_arr, dtype=float)

    # ── ha_high / ha_low: vectorised max/min ─────────────────────────────────
    ha_high = np.maximum(df["high"].to_numpy(),
               np.maximum(ha_open.to_numpy(), ha_close.to_numpy()))
    ha_low  = np.minimum(df["low"].to_numpy(),
               np.minimum(ha_open.to_numpy(), ha_close.to_numpy()))

    df["ha_open"]  = ha_open.values
    df["ha_high"]  = ha_high
    df["ha_low"]   = ha_low
    df["ha_close"] = ha_close.values
    df["ha_color"] = np.where(df["ha_close"] >= df["ha_open"], "green", "red")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: INITIAL BIAS AT 10:30  (15-min comparison method)
# ─────────────────────────────────────────────────────────────────────────────

# Key timestamps (NSE market, all times are today's date)
_T_H1_LAST_START  = "10:00"   # last  15-min bar of Hour-1 opens  at 10:00
_T_H2_FIRST_START = "10:15"   # first 15-min bar of Hour-2 opens  at 10:15
_T_BIAS_READY     = "10:30"   # H2-first bar closes at 10:30; bias can be set

# Sentinel so we run the 15-min comparison exactly once per trading day
_INITIAL_BIAS_SET = False


def determine_initial_bias_15m(df_15m: pd.DataFrame) -> tuple:
    """
    INITIAL bias determination at 10:30 using 15-minute HA candles.

    Compares two specific bars:
      H1_last  — the 10:00–10:15 bar (last bar of the first trading hour)
      H2_first — the 10:15–10:30 bar (first bar of the second trading hour)

    Rules:
      H2_first HA-close > H1_last HA-close  →  BULLISH  (buyers accelerating)
      H2_first HA-close < H1_last HA-close  →  BEARISH  (sellers taking over)
      Equal                                 →  None     (no conviction; retry)

    The function also requires that H2_first is 'data-lag confirmed', i.e.
    current time >= 10:30 + DATA_LAG_SECONDS, before it considers the bar final.

    Called ONLY once (guarded by _INITIAL_BIAS_SET flag in the main loop).
    Returns (bias_string, strength_string) or (None, None).
    """
    global _INITIAL_BIAS_SET

    if df_15m is None or df_15m.empty:
        return None, None

    now = datetime.now()
    today = now.date()

    # ── Timestamps for the two reference bars ─────────────────────────────────
    h1_last_ts  = pd.Timestamp(f"{today} {_T_H1_LAST_START}")   # 10:00
    h2_first_ts = pd.Timestamp(f"{today} {_T_H2_FIRST_START}")  # 10:15
    h2_close_dt = datetime(today.year, today.month, today.day, 10, 30)

    # Data-availability guard: H2_first bar must be fully closed + lag
    if now < h2_close_dt + timedelta(seconds=DATA_LAG_SECONDS):
        if DEBUG_MODE:
            eta = (h2_close_dt + timedelta(seconds=DATA_LAG_SECONDS) - now).seconds
            print(f"   15m bias: H2_first bar not yet confirmed — {eta}s remaining")
        return None, None

    # ── Locate the two bars in df_15m ─────────────────────────────────────────
    # Floor both sides to minute to guard against any residual sub-second noise
    df_15m = df_15m.copy()
    df_15m["datetime"] = pd.to_datetime(df_15m["datetime"]).dt.floor("min")
    h1_last_ts  = h1_last_ts.floor("min")
    h2_first_ts = h2_first_ts.floor("min")

    row_h1 = df_15m[df_15m["datetime"] == h1_last_ts]
    row_h2 = df_15m[df_15m["datetime"] == h2_first_ts]

    if row_h1.empty or row_h2.empty:
        if DEBUG_MODE:
            available = df_15m["datetime"].dt.strftime("%H:%M").tolist()
            print(f"   15m bias: Could not find reference bars. Available: {available[-10:]}")
            print(f"   Looking for: {h1_last_ts.strftime('%H:%M')} and {h2_first_ts.strftime('%H:%M')}")
        return None, None

    # ── Compute HA on a 2-bar slice (inherits context from full df_15m) ───────
    # Use the full df up to and including H2_first so HA seed is stable.
    ha_full = compute_ha(df_15m[df_15m["datetime"] <= h2_first_ts].copy())

    if len(ha_full) < 2:
        return None, None

    # Extract the two reference rows from the full HA series
    ha_h1 = ha_full[ha_full["datetime"] == h1_last_ts]
    ha_h2 = ha_full[ha_full["datetime"] == h2_first_ts]

    if ha_h1.empty or ha_h2.empty:
        if DEBUG_MODE:
            print("   15m bias: HA rows for reference bars not found after compute_ha")
        return None, None

    h1_ha_close = ha_h1.iloc[0]["ha_close"]
    h2_ha_close = ha_h2.iloc[0]["ha_close"]
    h1_ha_open  = ha_h1.iloc[0]["ha_open"]
    h2_ha_open  = ha_h2.iloc[0]["ha_open"]

    print(f"\n   📊 15-min Bias Comparison (initial @ 10:30):")
    print(f"      H1_last  (10:00): HA open={h1_ha_open:.1f}  HA close={h1_ha_close:.1f}  "
          f"color={ha_h1.iloc[0]['ha_color'].upper()}")
    print(f"      H2_first (10:15): HA open={h2_ha_open:.1f}  HA close={h2_ha_close:.1f}  "
          f"color={ha_h2.iloc[0]['ha_color'].upper()}")

    if h2_ha_close > h1_ha_close:
        bias = "BULLISH"
        delta = h2_ha_close - h1_ha_close
        print(f"      → H2 close ABOVE H1 close by {delta:.1f} pts  →  BULLISH (CE)")
    elif h2_ha_close < h1_ha_close:
        bias = "BEARISH"
        delta = h1_ha_close - h2_ha_close
        print(f"      → H2 close BELOW H1 close by {delta:.1f} pts  →  BEARISH (PE)")
    else:
        print(f"      → H2 close == H1 close ({h2_ha_close:.1f})  →  No conviction; will retry")
        return None, None

    # ── v2: Grade bias strength ───────────────────────────────────────────────
    if delta >= BIAS_STRONG_MIN_DELTA:
        strength = "STRONG"
    elif delta < BIAS_WEAK_MAX_DELTA:
        strength = "WEAK"
        print(f"      ⚠️  Bias delta {delta:.1f} < {BIAS_WEAK_MAX_DELTA} threshold → WEAK bias → skipping trading today")
        return None, None   # treat WEAK same as no bias — don't trade
    else:
        strength = "NORMAL"

    print(f"      Bias strength: {strength} (delta {delta:.1f} pts)")
    return bias, strength


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2b: HOURLY BIAS REFRESH  (called from 10:30 onwards for subsequent hours)
# ─────────────────────────────────────────────────────────────────────────────

def determine_1h_bias(df_1h: pd.DataFrame) -> Optional[str]:
    """
    Refresh bias each hour from the latest fully-closed 1H HA candle.
    This is used ONLY after the initial bias has been set by
    determine_initial_bias_15m() — i.e. from 11:15 onwards (second 1H candle).

    "Fully closed" means candle_end + DATA_LAG_SECONDS <= now.

    Uses PROCESSED_BIAS_CANDLES to avoid re-evaluating the same candle twice.
    Returns the new bias string, the unchanged BIAS, or None (nothing to do).
    """
    global BIAS, PROCESSED_BIAS_CANDLES

    if df_1h is None or df_1h.empty:
        return None

    ha = compute_ha(df_1h)

    now = datetime.now()
    last_candle       = None
    candle_start_found = None
    candle_end_found   = None

    for _, row in ha.iloc[::-1].iterrows():
        candle_start = pd.Timestamp(row["datetime"])
        candle_end   = candle_start + pd.Timedelta(hours=1)
        if now >= candle_end + timedelta(seconds=DATA_LAG_SECONDS):
            last_candle        = row
            candle_start_found = candle_start
            candle_end_found   = candle_end
            break

    if last_candle is None:
        if DEBUG_MODE:
            print("   1H HA: No fully-closed (with data lag) 1H candle available yet")
        return None

    # Skip the very first 1H candle (9:15–10:15) — that one is now handled by
    # determine_initial_bias_15m() above.  We only update from the 2nd candle
    # (10:15–11:15) onwards.
    first_candle_start = pd.Timestamp(f"{now.date()} 09:15")
    if candle_start_found == first_candle_start:
        if DEBUG_MODE:
            print("   1H HA: Skipping first candle (bias already set via 15-min comparison)")
        return BIAS   # return current bias unchanged

    # Already processed this candle?
    if candle_start_found in PROCESSED_BIAS_CANDLES:
        return BIAS

    color = last_candle["ha_color"]
    bias  = "BULLISH" if color == "green" else "BEARISH"

    PROCESSED_BIAS_CANDLES.add(candle_start_found)

    print(f"   📊 1H HA candle {candle_start_found.strftime('%H:%M')}–{candle_end_found.strftime('%H:%M')} "
          f"-> {color.upper()} -> bias refreshed: {bias}")
    print(f"      HA O:{last_candle['ha_open']:.1f}  H:{last_candle['ha_high']:.1f}  "
          f"L:{last_candle['ha_low']:.1f}  C:{last_candle['ha_close']:.1f}")

    return bias


def check_bias_override(df_15m):
    """
    At 15 min after each 1H candle closes (10:30, 11:30, 12:30, ...), check
    whether the new hour opened with a gap-down below the previous hour's low
    and has already broken that level.  If so, override BULLISH -> BEARISH.

    Runs in the :30-:35 minute window of any hour (i.e. the first 15m bar of
    the new hour has fully printed).  Uses BIAS_OVERRIDE_DONE_FOR to fire at
    most once per hour boundary, regardless of how many loop iterations land
    in that window.
    """
    global BIAS, BIAS_OVERRIDE_ACTIVE, BIAS_OVERRIDE_DONE_FOR

    if BIAS != "BULLISH":
        return False

    now = datetime.now()
    # Only act in the :30-:35 minute window of any hour
    if now.minute < 30 or now.minute > 35:
        return False

    # The 1H candle that just closed started at :15 of the current hour
    candle_close_time = now.replace(minute=15, second=0, microsecond=0)

    if candle_close_time in BIAS_OVERRIDE_DONE_FOR:
        return False   # already handled this hour boundary

    # Mark this hour boundary as checked (whether or not override fires)
    BIAS_OVERRIDE_DONE_FOR.add(candle_close_time)

    if df_15m is None or len(df_15m) < 2:
        return False

    # Candles straddling the :15 boundary of this hour:
    #   last_before = last 15m bar of the previous hour  (ends at :15)
    #   first_after = first 15m bar of the new hour      (starts at :15)
    before = df_15m[df_15m["datetime"] < candle_close_time]
    after  = df_15m[df_15m["datetime"] >= candle_close_time]

    if len(before) == 0 or len(after) == 0:
        return False

    last_before = before.iloc[-1]
    first_after = after.iloc[0]

    prev_hour_low = last_before["low"]
    new_hour_open = first_after["open"]

    # Condition 1: gap-down - new hour opened below previous hour's low
    if new_hour_open >= prev_hour_low:
        return False

    # Condition 2: confirmation - price actually traded below that low
    current_low = after["low"].min()
    if current_low < prev_hour_low:
        gap_pct = (new_hour_open - prev_hour_low) / prev_hour_low * 100
        print(f"\n⚠️  BIAS OVERRIDE TRIGGERED at {now.strftime('%H:%M:%S')}!")
        print(f"   Previous hour low: {prev_hour_low:.1f}, new hour open: {new_hour_open:.1f} (gap {gap_pct:.2f}%)")
        print(f"   Price broke below previous hour low -> Switching bias to BEARISH")
        BIAS = "BEARISH"
        BIAS_OVERRIDE_ACTIVE = True
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# v2: MULTI-TIMEFRAME CONFIRMATION  (1H trend filter)
# ─────────────────────────────────────────────────────────────────────────────

# Tolerance for 1H trend alignment check (points).
# Nifty moves in increments of ~5–10 pts per 1H HA candle during consolidation.
# A 5-pt tolerance allows minor counter-wiggles without blocking a valid trade.
# Set to 0.0 to restore strict alignment (original v2 behaviour).
TREND_ALIGN_TOLERANCE = 5.0


def is_1h_trend_aligned(df_1h: pd.DataFrame, bias: str) -> bool:
    """
    v3: Confirm the 1H Heikin-Ashi trend is aligned with the current bias
    before allowing a 15m entry.

    Uses a small tolerance (TREND_ALIGN_TOLERANCE points) so that minor
    counter-wiggles on the 1H HA candle do not block a valid trade.

    BEARISH → last ha_close <= prev ha_close + TREND_ALIGN_TOLERANCE
    BULLISH → last ha_close >= prev ha_close - TREND_ALIGN_TOLERANCE

    Rationale: in a strong intraday trend the 1H HA often forms a brief
    consolidation candle before the next leg. A strict (zero-tolerance)
    check flags this as "not bearish" and blocks the 15m entry — which
    is exactly the miss that occurred on 29 May 2026 around 12:30.

    If df_1h is too short or data is unavailable, defaults to True
    (fail-open: don't block trades just because 1H data is thin).
    """
    if not ENABLE_1H_TREND_FILTER:
        return True
    if df_1h is None or len(df_1h) < 2:
        if DEBUG_MODE:
            print("   1H filter: insufficient bars — allowing entry")
        return True
    ha = compute_ha(df_1h)
    if len(ha) < 2:
        return True
    last  = ha.iloc[-1]
    prev  = ha.iloc[-2]
    delta = last["ha_close"] - prev["ha_close"]
    if bias == "BEARISH":
        aligned = delta <= TREND_ALIGN_TOLERANCE    # allow up to +5 pt counter-move
    else:
        aligned = delta >= -TREND_ALIGN_TOLERANCE   # allow up to -5 pt counter-move
    if not aligned and DEBUG_MODE:
        print(f"   ❌ 1H trend NOT aligned with {bias}: "
              f"ha_close {prev['ha_close']:.1f} → {last['ha_close']:.1f} "
              f"(delta {delta:+.1f}, tolerance ±{TREND_ALIGN_TOLERANCE})")
    elif DEBUG_MODE and abs(delta) <= TREND_ALIGN_TOLERANCE and delta != 0:
        print(f"   ⚠️  1H tolerance used: {bias} entry allowed despite "
              f"delta {delta:+.1f} (within ±{TREND_ALIGN_TOLERANCE})")
    return aligned


# ─────────────────────────────────────────────────────────────────────────────
# v2: VOLATILITY / RANGE FILTER
# ─────────────────────────────────────────────────────────────────────────────

def has_sufficient_range(df_15m: pd.DataFrame) -> bool:
    """
    v2: Return True if the last closed 15m candle has a range (high - low)
    that is at least MIN_RANGE_RATIO × the average range over the last
    RANGE_LOOKBACK_BARS candles.

    Skips low-volatility fake reversals that tend to whipsaw.
    Returns True (pass) when there is insufficient history to judge.
    """
    if MIN_RANGE_RATIO <= 0:
        return True
    if df_15m is None or len(df_15m) < max(RANGE_LOOKBACK_BARS + 1, 2):
        return True   # not enough history — fail-open
    ranges = (df_15m["high"] - df_15m["low"])
    avg_range = ranges.iloc[-(RANGE_LOOKBACK_BARS + 1):-1].mean()
    last_range = ranges.iloc[-2]   # last *closed* candle
    if avg_range < 0.001:
        return True
    ratio = last_range / avg_range
    if ratio < MIN_RANGE_RATIO and DEBUG_MODE:
        print(f"   ❌ Range filter: last range {last_range:.1f} = "
              f"{ratio:.0%} of avg {avg_range:.1f} — below {MIN_RANGE_RATIO:.0%} threshold")
    return ratio >= MIN_RANGE_RATIO




def has_meaningful_body(ha_row: pd.Series, min_body_ratio: float = MIN_BODY_RATIO) -> bool:
    """
    Return True if the Heikin-Ashi candle has a body that is at least
    min_body_ratio of the total candle range (ha_high - ha_low).
    Used to filter out doji candles that create false reversal signals.
    """
    ha_range = ha_row["ha_high"] - ha_row["ha_low"]
    if ha_range < 0.001:
        return False
    body = abs(ha_row["ha_close"] - ha_row["ha_open"])
    return (body / ha_range) >= min_body_ratio


def detect_early_reversal_ce(ha: pd.DataFrame) -> Optional[dict]:
    """
    Bullish reversal → CE entry.

    Pattern:
      • Both the previous and current candles must have a meaningful body
        (avoid doji indecision).
      • Previous candle:  RED (ha_color == 'red')
      • Current candle:   GREEN (first green after reds)
      • Close above previous HA high  → momentum confirmation
      • Small lower shadow             → buyers stepped in cleanly
    """
    if len(ha) < 2:
        return None

    last = ha.iloc[-1]
    prev = ha.iloc[-2]

    # -- NEW: require meaningful bodies to filter doji -------------------------
    if not (has_meaningful_body(prev) and has_meaningful_body(last)):
        if DEBUG_MODE:
            print("   CE filter: doji candle detected (body too small) — no signal")
        return None

    if not (prev["ha_color"] == "red" and last["ha_color"] == "green"):
        return None

    # Close above previous HA high
    if last["ha_close"] <= prev["ha_high"]:
        if DEBUG_MODE:
            print(f"   CE filter: close {last['ha_close']:.1f} ≤ prev HA high {prev['ha_high']:.1f}")
        return None

    # Small lower shadow check
    candle_range = last["ha_high"] - last["ha_low"]
    if candle_range < 0.001:
        return None
    lower_shadow = last["ha_open"] - last["ha_low"]   # HA open > HA low for green candle
    lower_shadow_pct = lower_shadow / candle_range
    if lower_shadow_pct > SMALL_LOWER_SHADOW_MAX:
        if DEBUG_MODE:
            print(f"   CE filter: lower shadow {lower_shadow_pct:.2%} > max {SMALL_LOWER_SHADOW_MAX:.2%}")
        return None

    return {
        "signal":         "CE",
        "candle_time":    last["datetime"],
        "ha_close":       last["ha_close"],
        "prev_ha_high":   prev["ha_high"],
        "lower_shadow_pct": lower_shadow_pct,
    }


def detect_early_reversal_pe(ha: pd.DataFrame) -> Optional[dict]:
    """
    Bearish reversal → PE entry.

    Pattern:
      • Both the previous and current candles must have a meaningful body
        (avoid doji indecision).
      • Previous candle:  GREEN
      • Current candle:   RED (first red after greens)
      • Close below previous HA low   → momentum confirmation
      • Small upper shadow             → sellers stepped in cleanly
    """
    if len(ha) < 2:
        return None

    last = ha.iloc[-1]
    prev = ha.iloc[-2]

    # -- NEW: require meaningful bodies to filter doji -------------------------
    if not (has_meaningful_body(prev) and has_meaningful_body(last)):
        if DEBUG_MODE:
            print("   PE filter: doji candle detected (body too small) — no signal")
        return None

    if not (prev["ha_color"] == "green" and last["ha_color"] == "red"):
        return None

    # Close below previous HA low
    if last["ha_close"] >= prev["ha_low"]:
        if DEBUG_MODE:
            print(f"   PE filter: close {last['ha_close']:.1f} ≥ prev HA low {prev['ha_low']:.1f}")
        return None

    # Small upper shadow check
    candle_range = last["ha_high"] - last["ha_low"]
    if candle_range < 0.001:
        return None
    upper_shadow = last["ha_high"] - last["ha_open"]   # HA high > HA open for red candle
    upper_shadow_pct = upper_shadow / candle_range
    if upper_shadow_pct > SMALL_UPPER_SHADOW_MAX:
        if DEBUG_MODE:
            print(f"   PE filter: upper shadow {upper_shadow_pct:.2%} > max {SMALL_UPPER_SHADOW_MAX:.2%}")
        return None

    return {
        "signal":          "PE",
        "candle_time":     last["datetime"],
        "ha_close":        last["ha_close"],
        "prev_ha_low":     prev["ha_low"],
        "upper_shadow_pct": upper_shadow_pct,
    }


def bullish_structure_ok(df: pd.DataFrame, bias_strength: str = "normal") -> bool:
    """
    Confirm 15m price structure is making higher highs before a CE entry.

    Normal mode: requires last 3 swing highs to be strictly rising.
    Strong mode (bias_strength == 'strong'): allows one flat step — useful
    in powerful trends where the market consolidates briefly before continuing.

    Uses the last 4 raw (non-HA) highs.
    """
    if len(df) < 4:
        return False
    highs = df["high"].tail(4).values
    if bias_strength == "strong":
        ok = highs[-1] > highs[-2] and highs[-2] >= highs[-3]  # flat → rising allowed
    else:
        ok = highs[-1] > highs[-2] and highs[-2] > highs[-3]   # strict rising
    if not ok and DEBUG_MODE:
        print(f"   ⛔ Structure blocked (CE, {bias_strength}): "
              f"highs {highs[-3]:.1f} → {highs[-2]:.1f} → {highs[-1]:.1f} — need higher highs")
    return ok


def bearish_structure_ok(df: pd.DataFrame, bias_strength: str = "normal") -> bool:
    """
    Confirm 15m price structure is making lower lows before a PE entry.

    Normal mode: requires last 3 swing lows to be strictly falling.
    Strong mode (bias_strength == 'strong'): allows one flat step — useful
    in strong downtrends where lows stall briefly before the next leg down.

    Uses the last 4 raw (non-HA) lows.
    """
    if len(df) < 4:
        return False
    lows = df["low"].tail(4).values
    if bias_strength == "strong":
        ok = lows[-1] < lows[-2] and lows[-2] <= lows[-3]   # flat → falling allowed
    else:
        ok = lows[-1] < lows[-2] and lows[-2] < lows[-3]    # strict falling
    if not ok and DEBUG_MODE:
        print(f"   ⛔ Structure blocked (PE, {bias_strength}): "
              f"lows {lows[-3]:.1f} → {lows[-2]:.1f} → {lows[-1]:.1f} — need lower lows")
    return ok




# ─────────────────────────────────────────────────────────────────────────────
# v4: INTRA-HOUR BIAS FLIP  (15m HA reversal detector)
# ─────────────────────────────────────────────────────────────────────────────

def check_15m_bias_flip(df_15m: pd.DataFrame) -> bool:
    """
    v4: Flip BIAS when the market makes a sustained move against it.

    Conditions (all required):
      1. Cooldown: >= BIAS_FLIP_COOLDOWN_MINS since last flip
      2. Last BIAS_FLIP_CONSEC_CANDLES 15m HA bars all opposite colour
      3. HA close moved >= BIAS_FLIP_MIN_MOVE pts against current bias
         since bias was last set

    Returns True if flipped, False otherwise.
    """
    global BIAS, BIAS_STRENGTH, BIAS_SET_AT, _LAST_BIAS_FLIP_TIME, _BIAS_SET_HA_CLOSE

    if not ENABLE_BIAS_FLIP or BIAS is None:
        return False
    if df_15m is None or len(df_15m) < BIAS_FLIP_CONSEC_CANDLES + 1:
        return False

    # Cooldown
    if _LAST_BIAS_FLIP_TIME is not None:
        mins_since = (datetime.now() - _LAST_BIAS_FLIP_TIME).total_seconds() / 60
        if mins_since < BIAS_FLIP_COOLDOWN_MINS:
            return False

    # Slice from when bias was set
    if BIAS_SET_AT:
        bias_floor = pd.Timestamp(BIAS_SET_AT).floor("15min")
        df_check = df_15m[df_15m["datetime"] >= bias_floor]
    else:
        df_check = df_15m
    if len(df_check) < BIAS_FLIP_CONSEC_CANDLES + 1:
        return False

    ha = compute_ha(df_check)
    if len(ha) < BIAS_FLIP_CONSEC_CANDLES + 1:
        return False

    # Check consecutive opposite-colour bars
    opposite = "green" if BIAS == "BEARISH" else "red"
    last_n   = ha.iloc[-BIAS_FLIP_CONSEC_CANDLES:]
    all_opposite = all(row["ha_color"] == opposite for _, row in last_n.iterrows())
    if not all_opposite:
        return False

    # Check move magnitude
    current_close = float(ha.iloc[-1]["ha_close"])
    if _BIAS_SET_HA_CLOSE is not None:
        move = current_close - _BIAS_SET_HA_CLOSE
        move_ok = (move >= BIAS_FLIP_MIN_MOVE  if BIAS == "BEARISH"
                   else move <= -BIAS_FLIP_MIN_MOVE)
        if not move_ok:
            if DEBUG_MODE:
                print(f"   Flip candidate: {BIAS_FLIP_CONSEC_CANDLES}x {opposite} bars "
                      f"but move only {move:+.1f} pts (need {BIAS_FLIP_MIN_MOVE:.0f})")
            return False

    # Flip!
    old_bias = BIAS
    new_bias = "BULLISH" if BIAS == "BEARISH" else "BEARISH"
    print(f"\n{'='*70}")
    print(f"BIAS FLIP: {old_bias} -> {new_bias} at {datetime.now().strftime('%H:%M:%S')}")
    print(f"   Trigger: {BIAS_FLIP_CONSEC_CANDLES}x consecutive {opposite.upper()} HA bars")
    if _BIAS_SET_HA_CLOSE:
        print(f"   Move:    HA close {_BIAS_SET_HA_CLOSE:.1f} -> {current_close:.1f} "
              f"({current_close - _BIAS_SET_HA_CLOSE:+.1f} pts)")
    print(f"   Now scanning for {'CE' if new_bias == 'BULLISH' else 'PE'} entries")
    print(f"{'='*70}\n")

    BIAS               = new_bias
    BIAS_STRENGTH      = "NORMAL"
    BIAS_SET_AT        = datetime.now()
    _LAST_BIAS_FLIP_TIME = datetime.now()
    _BIAS_SET_HA_CLOSE = current_close
    return True

def scan_15m_for_entry(df_15m: pd.DataFrame, df_1h: Optional[pd.DataFrame] = None) -> Optional[dict]:
    """
    Run HA computation on 15-minute candles and check for the correct
    reversal pattern based on the current BIAS.

    v2/v3 additions:
      • 1H trend alignment check with tolerance (is_1h_trend_aligned)
      • Volatility / range filter (has_sufficient_range)
      • Adaptive structure filter (bias_strength-aware)
      • STRONG bias → relaxed structure regardless of time window
      • NORMAL bias → relaxed only inside 11:00–14:00 window, strict outside
      • Daily loss circuit-breaker (MAX_DAILY_LOSS_ABS) and max-trades guard

    Returns a signal dict or None.
    """
    global BIAS, BIAS_STRENGTH

    if df_15m is None or df_15m.empty or BIAS is None:
        return None

    # ── v2: Daily circuit-breaker ─────────────────────────────────────────────
    if DAILY_PNL <= -MAX_DAILY_LOSS_ABS:
        if DEBUG_MODE:
            print(f"   🛑 Daily loss limit hit (Rs {DAILY_PNL:+.0f} / limit -Rs {MAX_DAILY_LOSS_ABS:.0f}) — no new entries")
        return None

    # ── v2: Max trades guard ──────────────────────────────────────────────────
    if TRADES_TODAY >= MAX_TRADES_PER_DAY:
        if DEBUG_MODE:
            print(f"   🛑 Max trades/day reached ({TRADES_TODAY}/{MAX_TRADES_PER_DAY})")
        return None

    # ── v2: 1H trend alignment ────────────────────────────────────────────────
    if not is_1h_trend_aligned(df_1h, BIAS):
        if DEBUG_MODE:
            print("   ⛔ Entry blocked: 1H trend not aligned")
        return None

    # ── v2: Volatility / range filter ─────────────────────────────────────────
    if not has_sufficient_range(df_15m):
        if DEBUG_MODE:
            print("   ⛔ Entry blocked: candle range too small (low volatility)")
        return None

    # Only use candles after bias was set
    # Floor to 15-min boundary — BIAS_SET_AT is a wall-clock time (e.g. 10:38)
    # that falls between 15m bars; without flooring the filter would return 0 rows.
    bias_floor = pd.Timestamp(BIAS_SET_AT).floor("15min")
    df_15m = df_15m[df_15m["datetime"] >= bias_floor]
    if len(df_15m) < 2:
        return None

    ha = compute_ha(df_15m)

    # ── v3: Determine effective bias_strength for structure filter ─────────────
    # STRONG bias → always relaxed, regardless of time.
    #   Rationale: a strong initial bias means the 10:30 HA comparison showed
    #   a large delta. Blocking entries outside 11:00–14:00 on these days is
    #   overly conservative — the big move can start right after 10:30.
    #
    # NORMAL bias → relaxed only inside the preferred window.
    # Outside the window (early/late day) → strict structure required.
    now_t = current_hhmm()
    in_window = ENTRY_WINDOW_START <= now_t <= ENTRY_WINDOW_END
    if BIAS_STRENGTH == "STRONG":
        effective_strength = "strong"     # strong bias: relaxed always
    elif in_window:
        effective_strength = "strong"     # normal bias inside window: relaxed
    else:
        effective_strength = "normal"     # normal bias outside window: strict

    if BIAS == "BULLISH":
        if not bullish_structure_ok(df_15m, effective_strength):
            return None
        return detect_early_reversal_ce(ha)
    elif BIAS == "BEARISH":
        if not bearish_structure_ok(df_15m, effective_strength):
            return None
        return detect_early_reversal_pe(ha)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# OPTION CHAIN & ORDER PLACEMENT (FIXED: Upstox v2 correct fields)
# ─────────────────────────────────────────────────────────────────────────────

def get_nifty_spot() -> Optional[float]:
    """Fetch Nifty 50 LTP."""
    url = f"{BASE_URL}/market-quote/ltp"
    try:
        resp = requests.get(url, headers=_headers(),
                            params={"instrument_key": NIFTY_INDEX_KEY}, timeout=15)
        if resp.status_code == 200:
            inner = resp.json().get("data", {})
            # Upstox may return key as NSE_INDEX:Nifty 50
            for k, v in inner.items():
                ltp = v.get("last_price")
                if ltp:
                    return float(ltp)
        if DEBUG_MODE:
            print(f"⚠️  Spot fetch HTTP {resp.status_code}")
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚠️  Spot fetch error: {e}")
    return None


def get_nifty_option_chain(option_type: str) -> list:
    """
    Fetch Nifty option contracts (CE or PE) sorted by expiry.
    Returns list of contract dicts.
    """
    url = f"{BASE_URL}/option/contract"
    try:
        resp = requests.get(url, headers=_headers(),
                            params={"instrument_key": NIFTY_OPTION_KEY}, timeout=20)
        if resp.status_code != 200:
            if DEBUG_MODE:
                print(f"⚠️  Option chain HTTP {resp.status_code}: {resp.text[:200]}")
            return []
        contracts = resp.json().get("data", [])
        today = datetime.now().date()
        result = []
        for c in contracts:
            if c.get("instrument_type") != option_type:
                continue
            try:
                exp = datetime.strptime(c["expiry"], "%Y-%m-%d").date()
                if exp <= today:  # skip expired (including today for safety)
                    continue
                c["_expiry_date"] = exp
                result.append(c)
            except Exception:
                continue
        result.sort(key=lambda x: x["_expiry_date"])
        return result
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚠️  Option chain error: {e}")
        return []


def select_atm_contract(contracts: list, spot: float) -> Optional[dict]:
    """Select the nearest-expiry ATM contract."""
    if not contracts:
        return None
    # All contracts are already sorted by expiry; get nearest expiry
    nearest_expiry = contracts[0]["_expiry_date"]
    nearest = [c for c in contracts if c["_expiry_date"] == nearest_expiry]
    # Find ATM
    atm = min(nearest, key=lambda c: abs(c["strike_price"] - spot))
    return atm


def get_ltp(instrument_key: str) -> Optional[float]:
    """Fetch LTP for any instrument."""
    url = f"{BASE_URL}/market-quote/ltp"
    try:
        resp = requests.get(url, headers=_headers(),
                            params={"instrument_key": instrument_key}, timeout=15)
        if resp.status_code == 200:
            inner = resp.json().get("data", {})
            for k, v in inner.items():
                ltp = v.get("last_price")
                if ltp:
                    return float(ltp)
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚠️  LTP error for {instrument_key}: {e}")
    return None


def place_order(instrument_key: str, qty: int, txn_type: str,
                order_type: str, price: float = 0,
                trigger_price: float = 0) -> Optional[str]:
    """
    Place a buy/sell order. Returns order_id or None.
    - For entry: order_type="LIMIT", price = limit price
    - For stop-loss: order_type="SL", price = limit price (slightly below trigger), trigger_price = SL trigger
    - For exit: order_type="MARKET", price=0, trigger_price=0
    """
    url = f"{BASE_URL}/order/place"
    payload = {
        "quantity":           qty,
        "product":            ORDER_PRODUCT,      # "I" for intraday options
        "validity":           "DAY",
        "price":              price,
        "tag":                "HA_BOT",
        "instrument_key":     instrument_key,     # FIXED: was "instrument_token"
        "order_type":         order_type.upper(),
        "transaction_type":   txn_type.upper(),
        "disclosed_quantity": 0,
        "trigger_price":      trigger_price,
        "is_amo":             False,
    }
    try:
        resp = requests.post(url, headers=_order_headers(),
                             json=payload, timeout=15)
        if DEBUG_MODE:
            print(f"   📤 Order response ({resp.status_code}): {resp.text[:300]}")
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                return data.get("data", {}).get("order_id")
    except Exception as e:
        print(f"   ❌ Order error: {e}")
    return None


def _cancel_order(order_id: str) -> bool:
    """
    Cancel an existing order using Upstox v2 DELETE /order/cancel.
    order_id passed as query parameter, not JSON body.
    """
    url = f"{BASE_URL}/order/cancel"
    try:
        resp = requests.delete(url, headers=_order_headers(),
                               params={"order_id": order_id}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                if DEBUG_MODE:
                    print(f"   🗑️  Cancelled order {order_id}")
                return True
        if DEBUG_MODE:
            print(f"   ⚠️  Cancel failed HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        if DEBUG_MODE:
            print(f"   ⚠️  Cancel error: {e}")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def execute_entry(signal: dict):
    """
    Given a confirmed entry signal, select option contract,
    fetch premium, and place BUY + SL orders.
    """
    global ACTIVE_POSITION

    option_type = signal["signal"]  # "CE" or "PE"
    print(f"\n{'='*70}")
    print(f"🎯 ENTRY SIGNAL: {option_type} at {datetime.now().strftime('%H:%M:%S')}")
    print(f"   HA close: {signal['ha_close']:.1f} | Bias: {BIAS}")
    if option_type == "CE":
        print(f"   Prev HA high: {signal['prev_ha_high']:.1f} | "
              f"Lower shadow: {signal['lower_shadow_pct']:.1%}")
    else:
        print(f"   Prev HA low:  {signal['prev_ha_low']:.1f} | "
              f"Upper shadow: {signal['upper_shadow_pct']:.1%}")
    print(f"{'='*70}")

    spot = get_nifty_spot()
    if not spot:
        print("   ❌ Cannot fetch Nifty spot — aborting entry")
        return

    # ── v3: Cached option chain fetch ─────────────────────────────────────────
    global OPTION_CHAIN_CACHE, LAST_CHAIN_FETCH
    cache_stale = (
        LAST_CHAIN_FETCH is None
        or (datetime.now() - LAST_CHAIN_FETCH).total_seconds() >= OPTION_CHAIN_CACHE_TTL
        or option_type not in OPTION_CHAIN_CACHE
    )
    if cache_stale:
        contracts = get_nifty_option_chain(option_type)
        OPTION_CHAIN_CACHE[option_type] = contracts
        LAST_CHAIN_FETCH = datetime.now()
        if DEBUG_MODE:
            print(f"   🔄 Option chain refreshed ({len(contracts)} contracts cached)")
    else:
        contracts = OPTION_CHAIN_CACHE[option_type]
        age_s = int((datetime.now() - LAST_CHAIN_FETCH).total_seconds())
        if DEBUG_MODE:
            print(f"   ✅ Option chain from cache (age {age_s}s / TTL {OPTION_CHAIN_CACHE_TTL}s)")
    contract  = select_atm_contract(contracts, spot)
    if not contract:
        print(f"   ❌ No {option_type} contracts found — aborting entry")
        return

    premium = get_ltp(contract["instrument_key"])
    if not premium or premium <= 0:
        print(f"   ❌ Cannot fetch premium for {contract.get('trading_symbol')} — aborting")
        return

    lot_size  = contract.get("lot_size", 50)
    total_qty = lot_size * ORDER_QUANTITY
    limit_px  = round(premium * 1.02, 2)   # 2% slippage buffer for LIMIT order

    print(f"   Option:  {contract.get('trading_symbol')}")
    print(f"   Strike:  {contract['strike_price']} | Expiry: {contract['expiry']}")
    print(f"   Spot:    {spot:.1f} | Premium LTP: {premium:.2f}")
    print(f"   Lot:     {lot_size} × {ORDER_QUANTITY} = {total_qty} qty")
    print(f"   Limit:   ₹{limit_px:.2f}")

    if not ENABLE_AUTO_TRADING:
        print("   ℹ️  Auto-trading DISABLED — signal logged only")
        _log_signal(signal, contract, premium, spot, order_id="SIGNAL_ONLY")
        return

    # ---- BUY order (LIMIT) ----
    order_id = place_order(
        instrument_key=contract["instrument_key"],
        qty=total_qty,
        txn_type="BUY",
        order_type="LIMIT",
        price=limit_px,
    )
    if not order_id:
        print("   ❌ BUY order failed — aborting")
        return
    print(f"   ✅ BUY order placed: {order_id}")

    # ---- Stop-Loss order (SL-LIMIT) ----
    sl_trigger = round(premium * (1 - STOPLOSS_PCT / 100), 2)
    sl_limit   = round(sl_trigger * 0.995, 2)   # 0.5% inside trigger for fill safety
    sl_id = place_order(
        instrument_key=contract["instrument_key"],
        qty=total_qty,
        txn_type="SELL",
        order_type="SL",                         # SL-LIMIT order
        price=sl_limit,
        trigger_price=sl_trigger,
    )
    if sl_id:
        print(f"   🛡️  SL order placed: {sl_id} (trigger ₹{sl_trigger:.2f}, limit ₹{sl_limit:.2f})")
    else:
        print("   ⚠️  SL order failed — position unprotected")

    # ---- Target calculation ----
    target = round(premium * (1 + (STOPLOSS_PCT / 100) * TARGET_MULTIPLIER), 2)

    # ---- Track position ----
    ACTIVE_POSITION = {
        "order_id":       order_id,
        "sl_order_id":    sl_id,
        "instrument_key": contract["instrument_key"],
        "trading_symbol": contract.get("trading_symbol"),
        "option_type":    option_type,
        "entry_price":    premium,
        "quantity":       total_qty,
        "sl_trigger":     sl_trigger,
        "target":         target,
        "entry_time":     datetime.now(),
        "signal":         signal,
    }

    _log_signal(signal, contract, premium, spot, order_id=order_id)
    print(f"   Target: ₹{target:.2f} | SL: ₹{sl_trigger:.2f}")
    print(f"{'='*70}\n")


def modify_sl_order(new_trigger: float, new_limit: float) -> bool:
    """
    v2: Replace the exchange SL order with an updated one at a better level.

    Steps:
      1. Cancel the existing SL order (if any).
      2. Place a new SL-LIMIT order at new_trigger / new_limit.
      3. Update ACTIVE_POSITION['sl_order_id'] with the new order id.

    Returns True if the new SL order was placed successfully.
    """
    global ACTIVE_POSITION

    if not ENABLE_AUTO_TRADING:
        return False

    old_sl_id = ACTIVE_POSITION.get("sl_order_id")
    if old_sl_id:
        _cancel_order(old_sl_id)
        ACTIVE_POSITION["sl_order_id"] = None

    new_sl_id = place_order(
        instrument_key=ACTIVE_POSITION["instrument_key"],
        qty=ACTIVE_POSITION["quantity"],
        txn_type="SELL",
        order_type="SL",
        price=new_limit,
        trigger_price=new_trigger,
    )
    if new_sl_id:
        ACTIVE_POSITION["sl_order_id"] = new_sl_id
        if DEBUG_MODE:
            print(f"   🔄 SL order updated on exchange: {new_sl_id} "
                  f"(trigger ₹{new_trigger:.2f}, limit ₹{new_limit:.2f})")
        return True
    else:
        if DEBUG_MODE:
            print(f"   ⚠️  Failed to place updated SL order — position may be unprotected")
        return False




def _compute_ha_trail(df_15m: pd.DataFrame) -> Optional[float]:
    """
    Compute the current HA-based trailing stop for the active position.

    CE  →  trail = prev_ha_low  + TRAIL_BUFFER_PCT × ha_range
           (stop trails ABOVE the HA low; tighter than raw HA low)

    PE  →  trail = prev_ha_high − TRAIL_BUFFER_PCT × ha_range
           (stop trails BELOW the HA high; tighter than raw HA high)

    Uses the last completed 15m HA candle (iloc[-2]) so we never react to
    a still-forming bar.  Returns None if data is insufficient.
    """
    if df_15m is None or len(df_15m) < 2:
        return None

    ha = compute_ha(df_15m)
    if len(ha) < 2:
        return None

    prev = ha.iloc[-2]        # last *closed* candle
    ha_range = prev["ha_high"] - prev["ha_low"]
    buffer   = TRAIL_BUFFER_PCT * ha_range
    opt_type = ACTIVE_POSITION.get("option_type")

    if opt_type == "CE":
        trail = prev["ha_low"] + buffer
    elif opt_type == "PE":
        trail = prev["ha_high"] - buffer
    else:
        return None

    return round(trail, 2)


def check_exit():
    """
    Monitor ACTIVE_POSITION for SL / target / EOD exit.

    Trailing stop logic (HA-based, updates every scan):
      CE:  stop = prev_ha_low  + TRAIL_BUFFER_PCT × ha_range  (rises as trend climbs)
      PE:  stop = prev_ha_high − TRAIL_BUFFER_PCT × ha_range  (falls as trend drops)

    The trail only ever moves in the favourable direction (ratchet).
    v2: When the trail ratchets, the SL order on the exchange is also
    replaced via modify_sl_order() so the exchange is always in sync.
    """
    global ACTIVE_POSITION, DAILY_PNL, TRADES_TODAY

    if not ACTIVE_POSITION:
        return

    current_px = get_ltp(ACTIVE_POSITION["instrument_key"])
    if not current_px:
        return

    entry_px  = ACTIVE_POSITION["entry_price"]
    target    = ACTIVE_POSITION["target"]
    qty       = ACTIVE_POSITION["quantity"]
    pnl       = (current_px - entry_px) * qty
    pnl_pct   = (current_px - entry_px) / entry_px * 100
    now_str   = datetime.now().strftime("%H:%M")
    opt_type  = ACTIVE_POSITION["option_type"]
    exit_rsn  = None

    # ── HA trailing stop: ratchet + exchange SL sync (v2) ─────────────────────
    df_15m, _ = get_candles()
    ha_trail  = _compute_ha_trail(df_15m)

    if ha_trail is not None:
        prev_sl = ACTIVE_POSITION["sl_trigger"]
        sl_improved = False
        if opt_type == "CE" and ha_trail > prev_sl:
            ACTIVE_POSITION["sl_trigger"] = ha_trail
            sl_improved = True
            if DEBUG_MODE:
                print(f"   🔼 CE trail raised: ₹{prev_sl:.2f} → ₹{ha_trail:.2f} "
                      f"(buffer {TRAIL_BUFFER_PCT:.0%} of HA range)")
        elif opt_type == "PE" and ha_trail < prev_sl:
            ACTIVE_POSITION["sl_trigger"] = ha_trail
            sl_improved = True
            if DEBUG_MODE:
                print(f"   🔽 PE trail lowered: ₹{prev_sl:.2f} → ₹{ha_trail:.2f} "
                      f"(buffer {TRAIL_BUFFER_PCT:.0%} of HA range)")

        # v2: sync the exchange SL order whenever trail improves
        if sl_improved:
            new_trigger = ha_trail
            new_limit   = round(new_trigger * (0.995 if opt_type == "CE" else 1.005), 2)
            modify_sl_order(new_trigger, new_limit)

    sl = ACTIVE_POSITION["sl_trigger"]

    # ── Exit conditions ───────────────────────────────────────────────────────
    if opt_type == "CE" and current_px <= sl:
        exit_rsn = "TRAIL_SL_HIT"
    elif opt_type == "PE" and current_px >= sl:
        exit_rsn = "TRAIL_SL_HIT"
    elif current_px >= target and opt_type == "CE":
        exit_rsn = "TARGET_HIT"
    elif current_px <= target and opt_type == "PE":
        exit_rsn = "TARGET_HIT"
    elif now_str >= NO_NEW_ENTRY_AFTER:
        exit_rsn = "TIME_EXIT"

    if not exit_rsn:
        if DEBUG_MODE:
            print(f"   📊 {ACTIVE_POSITION['trading_symbol']} | "
                  f"LTP: ₹{current_px:.2f} | Trail SL: ₹{sl:.2f} | "
                  f"P&L: ₹{pnl:+.0f} ({pnl_pct:+.1f}%)", flush=True)
        return

    # Execute exit (MARKET order)
    print(f"\n{'='*70}")
    print(f"🔚 EXIT: {exit_rsn} | {ACTIVE_POSITION['trading_symbol']}")
    print(f"   Entry: ₹{entry_px:.2f} | Exit: ₹{current_px:.2f}")
    print(f"   P&L:   ₹{pnl:+.0f} ({pnl_pct:+.1f}%)")

    if ENABLE_AUTO_TRADING:
        # Cancel SL order first if it exists
        if ACTIVE_POSITION.get("sl_order_id"):
            _cancel_order(ACTIVE_POSITION["sl_order_id"])

        exit_id = place_order(
            instrument_key=ACTIVE_POSITION["instrument_key"],
            qty=ACTIVE_POSITION["quantity"],
            txn_type="SELL",
            order_type="MARKET",
            price=0,
        )
        if exit_id:
            print(f"   ✅ Exit order: {exit_id}")
        else:
            print("   ⚠️  Exit order failed — place manually!")

    DAILY_PNL  += pnl
    TRADES_TODAY += 1
    _log_exit(ACTIVE_POSITION, current_px, exit_rsn, pnl, pnl_pct)
    ACTIVE_POSITION = {}
    print(f"   Daily P&L so far: ₹{DAILY_PNL:+.0f} | Trades today: {TRADES_TODAY}")
    print(f"{'='*70}\n")


def force_market_exit():
    """
    Unconditional market-order exit used at EOD.
    Unlike check_exit(), this always sells regardless of price vs SL/target.
    """
    global ACTIVE_POSITION, DAILY_PNL

    if not ACTIVE_POSITION:
        return

    current_px = get_ltp(ACTIVE_POSITION["instrument_key"])
    entry_px   = ACTIVE_POSITION["entry_price"]
    qty        = ACTIVE_POSITION["quantity"]
    pnl        = ((current_px - entry_px) * qty) if current_px else 0.0
    pnl_pct    = ((current_px - entry_px) / entry_px * 100) if current_px else 0.0

    print(f"\n{'='*70}")
    print(f"⏰ EOD FORCE-EXIT | {ACTIVE_POSITION['trading_symbol']}")
    if current_px:
        print(f"   Entry: ₹{entry_px:.2f} | LTP: ₹{current_px:.2f}")
        print(f"   P&L:   ₹{pnl:+.0f} ({pnl_pct:+.1f}%)")
    else:
        print(f"   Entry: ₹{entry_px:.2f} | LTP: unavailable — selling at market")

    if ENABLE_AUTO_TRADING:
        # Cancel any pending SL order first
        if ACTIVE_POSITION.get("sl_order_id"):
            _cancel_order(ACTIVE_POSITION["sl_order_id"])

        exit_id = place_order(
            instrument_key=ACTIVE_POSITION["instrument_key"],
            qty=qty,
            txn_type="SELL",
            order_type="MARKET",
            price=0,
        )
        if exit_id:
            print(f"   ✅ Force-exit order: {exit_id}")
        else:
            print("   ⚠️  Force-exit order FAILED — close manually NOW!")
    else:
        print("   ℹ️  Auto-trading DISABLED — log only")

    DAILY_PNL += pnl
    _log_exit(ACTIVE_POSITION, current_px or entry_px, "EOD_FORCE_EXIT", pnl, pnl_pct)
    ACTIVE_POSITION = {}
    print(f"   Daily P&L: ₹{DAILY_PNL:+.0f}")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

# Single definition of CSV columns — used by both _log_signal and _csv_append.
# If you add a column, update here and nowhere else.
CSV_HEADERS = [
    "timestamp", "signal", "bias", "ha_close",
    "premium", "symbol", "strike", "expiry", "spot", "order_id",
]

def _log_signal(signal, contract, premium, spot, order_id):
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        signal["signal"],
        BIAS,
        signal["ha_close"],
        premium,
        contract.get("trading_symbol"),
        contract["strike_price"],
        contract["expiry"],
        spot,
        order_id,
    ]
    _csv_append(row)
    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"ENTRY: {datetime.now()} | {signal['signal']} | Bias: {BIAS}\n")
        f.write(f"  Option: {contract.get('trading_symbol')} | Premium: ₹{premium:.2f}\n")
        f.write(f"  Order ID: {order_id}\n")


def _log_exit(pos, exit_px, reason, pnl, pnl_pct):
    with open(LOG_FILE, "a") as f:
        f.write(f"EXIT: {datetime.now()} | {reason}\n")
        f.write(f"  Entry: ₹{pos['entry_price']:.2f} | Exit: ₹{exit_px:.2f}\n")
        f.write(f"  P&L:   ₹{pnl:+.0f} ({pnl_pct:+.1f}%)\n")
        f.write(f"{'='*70}\n")


def _csv_append(row):
    """Append one data row to the trade CSV. Writes header automatically on first call."""
    write_header = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(CSV_HEADERS)
        w.writerow(row)

# ─────────────────────────────────────────────────────────────────────────────
# MARKET HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.strftime("%H:%M")
    return "09:15" <= t < MARKET_CLOSE_TIME


def current_hhmm() -> str:
    return datetime.now().strftime("%H:%M")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def banner():
    print("\n" + "=" * 70)
    print("  NIFTY 50 OPTIONS TRADER -- HEIKIN-ASHI STRATEGY v4 (Fixed API)")
    print("=" * 70)
    print(f"  Mode:          {'LIVE TRADING' if ENABLE_AUTO_TRADING else 'SIGNAL ONLY'}")
    print(f"  Entry:         After 10:30 AM | No new entries after {NO_NEW_ENTRY_AFTER}")
    print(f"  Bias:          10:30 -- 15m comparison (H1 last vs H2 first bar HA-close)")
    print(f"  SL:            {STOPLOSS_PCT}% of premium | Target: {TARGET_MULTIPLIER}x risk")
    print(f"  Pattern:       RED RED -> GREEN HA (CE)  /  GREEN GREEN -> RED HA (PE)")
    print(f"  Data:          1-min intraday resampled to 15min + 1H")
    print(f"  Data lag:      {DATA_LAG_SECONDS}s (to ensure 1H candles are final)")
    print(f"  Doji filter:   MIN_BODY_RATIO = {MIN_BODY_RATIO} (0 = off)")
    print(f"  Order types:   LIMIT (entry), SL (stop-loss), MARKET (exit)")
    print(f"  Product:       {ORDER_PRODUCT} (I = Intraday)")
    print(f"  -- v2 additions --")
    print(f"  1H filter:     {'ON' if ENABLE_1H_TREND_FILTER else 'OFF'}")
    print(f"  Range filter:  {MIN_RANGE_RATIO:.0%} of {RANGE_LOOKBACK_BARS}-bar avg (0 = off)")
    print(f"  Daily loss cap:  Rs -{MAX_DAILY_LOSS_ABS:.0f}")
    print(f"  Max trades/day: {MAX_TRADES_PER_DAY}")
    print(f"  Bias thresholds: STRONG >= {BIAS_STRONG_MIN_DELTA}pts | WEAK < {BIAS_WEAK_MAX_DELTA}pts (skip)")
    print(f"  Entry window:  {ENTRY_WINDOW_START}-{ENTRY_WINDOW_END} (relaxed structure in window)")
    print(f"  Dynamic SL:    ON (exchange SL order replaced on each trail ratchet)")
    print(f"  1H tolerance:  ±{TREND_ALIGN_TOLERANCE} pts ({('strict (0=off)' if TREND_ALIGN_TOLERANCE == 0 else 'tolerant')})")
    print(f"  Chain cache:   {OPTION_CHAIN_CACHE_TTL}s TTL")
    print(f"  Type hints:    Optional[...] (Python 3.8+ compatible)")
    print(f"  -- v4 additions --")
    _fs = f"ON  ({BIAS_FLIP_CONSEC_CANDLES} bars + {BIAS_FLIP_MIN_MOVE:.0f}pt move, {BIAS_FLIP_COOLDOWN_MINS}m cooldown)" if ENABLE_BIAS_FLIP else "OFF"
    print(f"  Bias flip:     {_fs}")
    print("=" * 70 + "\n")

def main():
    global BIAS, BIAS_SET_AT, BIAS_STRENGTH, ACTIVE_POSITION, BIAS_OVERRIDE_ACTIVE, \
           PROCESSED_BIAS_CANDLES, BIAS_OVERRIDE_DONE_FOR, _INITIAL_BIAS_SET, \
           DAILY_PNL, TRADES_TODAY, _LAST_BIAS_FLIP_TIME, _BIAS_SET_HA_CLOSE, \
           OPTION_CHAIN_CACHE, LAST_CHAIN_FETCH

    banner()

    scan_count = 0

    while True:
        scan_count += 1
        now = datetime.now()
        t   = current_hhmm()

        # -- Market-closed guard (longer sleep, then restart loop) -------------
        if not is_market_open():
            print(f"💤 Market closed ({now.strftime('%H:%M:%S')}) — waiting...", flush=True)
            time.sleep(60)
            continue

        # -- Fetch candles (always, to keep data fresh) ------------------------
        df_15m, df_1h = get_candles()

        print(f"🔍 Scan #{scan_count} | {now.strftime('%H:%M:%S')} | "
              f"Bias: {BIAS or 'Not set'} ({BIAS_STRENGTH}) | "
              f"15m bars: {len(df_15m) if df_15m is not None else 0} | "
              f"1H bars: {len(df_1h) if df_1h is not None else 0} | "
              f"Trades: {TRADES_TODAY}/{MAX_TRADES_PER_DAY} | P&L: Rs {DAILY_PNL:+.0f}",
              flush=True)

        # -- STEP 1: No trades before 10:30 (need H2 first 15m bar to close) --
        if t < "10:15":
            print("   ⏳ Waiting for 10:15 AM...")
        elif t < "10:30":
            print("   ⏳ Waiting for 10:30 AM (H2 first 15-min bar)...")

        # -- STEP 3: EOD cleanup -----------------------------------------------
        elif t >= MARKET_CLOSE_TIME:
            if ACTIVE_POSITION:
                force_market_exit()
            print(f"\n📊 Day complete | Daily P&L: ₹{DAILY_PNL:+.0f}")
            print("   Restarting state for tomorrow...")
            BIAS, BIAS_SET_AT = None, None
            BIAS_STRENGTH = "NORMAL"
            _LAST_BIAS_FLIP_TIME = None
            _BIAS_SET_HA_CLOSE   = None
            PROCESSED_BIAS_CANDLES.clear()
            BIAS_OVERRIDE_DONE_FOR.clear()
            ACTIVE_POSITION = {}
            BIAS_OVERRIDE_ACTIVE = False
            _INITIAL_BIAS_SET = False   # reset so tomorrow's 15-min comparison runs fresh
            TRADES_TODAY = 0
            DAILY_PNL    = 0.0
            time.sleep(300)
            continue

        # -- STEP 2+: Active window (10:15 <= t < 15:30) ----------------------
        elif t >= "10:15":
            # ── STEP 2: Set / refresh bias ────────────────────────────────────
            # Phase A — INITIAL bias (once per day, at 10:30):
            #   Compare last 15-min bar of Hour-1 (10:00) vs first 15-min bar
            #   of Hour-2 (10:15).  Fire only after 10:30 + DATA_LAG_SECONDS.
            if not _INITIAL_BIAS_SET and t >= "10:30":
                init_bias, init_strength = determine_initial_bias_15m(df_15m)
                if init_bias is not None:
                    print(f"   ✅ Initial bias set: {init_bias} ({init_strength}) at {now.strftime('%H:%M:%S')} "
                          f"(15-min comparison)")
                    BIAS          = init_bias
                    BIAS_STRENGTH = init_strength
                    BIAS_SET_AT   = now
                    BIAS_OVERRIDE_ACTIVE = False
                    _INITIAL_BIAS_SET = True
                    if df_15m is not None and len(df_15m) >= 1:
                        _ha_snap = compute_ha(df_15m)
                        _BIAS_SET_HA_CLOSE = float(_ha_snap.iloc[-1]["ha_close"]) if len(_ha_snap) else None
                else:
                    print("   ⚠️  Initial 15-min bias not ready yet (or WEAK bias) — will retry")

            # Phase B — HOURLY refresh (from second 1H candle onwards, 11:15+):
            #   Uses the closing 1H HA candle colour (skips the first candle).
            elif _INITIAL_BIAS_SET and df_1h is not None and not df_1h.empty:
                new_bias = determine_1h_bias(df_1h)
                if new_bias is not None and new_bias != BIAS:
                    print(f"   📊 Bias updated: {BIAS or 'None'} -> {new_bias} at {now.strftime('%H:%M:%S')}")
                    BIAS = new_bias
                    BIAS_SET_AT = now
                    BIAS_OVERRIDE_ACTIVE = False

            # Waiting for 10:30
            elif not _INITIAL_BIAS_SET and t < "10:30":
                print("   ⏳ Waiting for 10:30 (H2 first 15-min bar to close)...")

            # 2b. Gap-down override: runs at :30-:35 of each hour after 10:30.
            #     check_bias_override() is self-throttled via BIAS_OVERRIDE_DONE_FOR.
            if current_hhmm() >= "10:30":
                check_bias_override(df_15m)

            # -- STEP 3b: Intra-hour bias flip check ─────────────────────────
            if _INITIAL_BIAS_SET and not ACTIVE_POSITION:
                if check_15m_bias_flip(df_15m):
                    OPTION_CHAIN_CACHE.clear()
                    LAST_CHAIN_FETCH = None

            # -- STEP 4: Monitor active position exits -------------------------
            if ACTIVE_POSITION:
                check_exit()

            # -- STEP 5: Scan for entry ----------------------------------------
            # Allow fresh entry whenever there is no open position and bias is set,
            # regardless of how many trades have already been taken today.
            if not ACTIVE_POSITION and t < NO_NEW_ENTRY_AFTER and BIAS is not None:
                signal = scan_15m_for_entry(df_15m, df_1h)
                if signal:
                    print(f"\n   🔔 Reversal pattern detected: {signal['signal']}")
                    execute_entry(signal)
                elif DEBUG_MODE:
                    # Show last 2 HA colours + fire the pattern detector to expose
                    # which sub-filter is blocking entry (body, close, shadow, structure).
                    if df_15m is not None and len(df_15m) >= 2:
                        if BIAS_SET_AT:
                            bias_floor = pd.Timestamp(BIAS_SET_AT).floor("15min")
                            ha_df = df_15m[df_15m["datetime"] >= bias_floor]
                            ha = compute_ha(ha_df) if len(ha_df) >= 2 else compute_ha(df_15m)
                        else:
                            ha = compute_ha(df_15m)
                        if len(ha) >= 2:
                            c1 = ha.iloc[-2]["ha_color"]
                            c2 = ha.iloc[-1]["ha_color"]
                            last_ha  = ha.iloc[-1]
                            prev_ha  = ha.iloc[-2]
                            # Show candle detail alongside colours
                            print(f"   15m HA: [{c1}] -> [{c2}] | "
                                  f"prev close={prev_ha['ha_close']:.1f} low={prev_ha['ha_low']:.1f} | "
                                  f"last close={last_ha['ha_close']:.1f} open={last_ha['ha_open']:.1f} | "
                                  f"No pattern yet")
                            # When colours look right but no signal fired, run detector
                            # verbosely so the exact blocking filter is printed
                            if BIAS == "BEARISH" and c1 == "green" and c2 == "red":
                                print("   🔎 PE pattern check (green→red seen, checking all filters):")
                                detect_early_reversal_pe(ha)
                            elif BIAS == "BULLISH" and c1 == "red" and c2 == "green":
                                print("   🔎 CE pattern check (red→green seen, checking all filters):")
                                detect_early_reversal_ce(ha)

        # -- Single sleep point for every normal path --------------------------
        time.sleep(SCAN_INTERVAL_SECS)

if __name__ == "__main__":
    main()