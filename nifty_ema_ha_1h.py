#!/usr/bin/env python3
"""
Nifty 50 Options Trader — 1H Heikin-Ashi + EMA Crossover Strategy
==================================================================
A faster, more responsive strategy designed for 1-hour timeframe
options trading on Nifty 50 via Upstox API v2.

SIGNAL ENGINE (replaces the 15m reversal engine from the HA v4 script):

  LONG (CE):
    1. A GREEN Heikin-Ashi candle closes on the 1H chart
       (small/no lower wick preferred — SMALL_LOWER_SHADOW_MAX filter)
    2. Fast EMA (9) crosses ABOVE Slow EMA (21) on the 1H chart
    3. 1H close is ABOVE the Trend EMA (50)
    4. RSI(14) is between RSI_MIN_LONG (52) and RSI_OVERBOUGHT (70)

  SHORT (PE):
    1. A RED Heikin-Ashi candle closes on the 1H chart
       (small/no upper wick preferred — SMALL_UPPER_SHADOW_MAX filter)
    2. Fast EMA (9) crosses BELOW Slow EMA (21) on the 1H chart
    3. 1H close is BELOW the Trend EMA (50)
    4. RSI(14) is between RSI_OVERSOLD (30) and RSI_MAX_SHORT (48)

FILTERS:
  • Min body ratio (doji filter): body < MIN_BODY_RATIO × range → skip
  • India VIX guard: entry blocked if VIX > VIX_MAX (configurable; optional)
  • Daily loss circuit-breaker: no entries if DAILY_PNL ≤ -MAX_DAILY_LOSS_ABS
  • Max trades per day: capped at MAX_TRADES_PER_DAY
  • Entry window: only between ENTRY_START and ENTRY_END

EXIT:
  • Hard SL: STOPLOSS_PCT % of premium (exchange SL-LIMIT order)
  • Target:   TARGET_MULTIPLIER × risk
  • Trailing: HA-based trail (previous 1H HA candle low/high ± buffer)
  • Time:     NO_NEW_ENTRY_AFTER; force-exit at MARKET_CLOSE_TIME
  • Signal:   opposite HA candle closes → exit immediately

STRIKE SELECTION:
  • Targets 0.45–0.55 delta (slightly OTM) via STRIKE_OFFSET_PCTS config
  • Falls back to ATM if no contract found in delta range

DATA:
  1-min intraday from Upstox → resampled to 1H (15-min offset for NSE open)

USAGE:
  export UPSTOX_TOKEN="your_token"
  python nifty_ema_ha_1h.py

REQUIREMENTS:
  pip install requests pandas numpy scipy
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
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ACCESS_TOKEN = os.environ.get("UPSTOX_TOKEN")
if not ACCESS_TOKEN:
    raise ValueError(
        "\n❌  UPSTOX_TOKEN environment variable is not set.\n"
        "    Export it before running:\n"
        "      export UPSTOX_TOKEN='your_token_here'\n"
    )

NIFTY_INDEX_KEY  = "NSE_INDEX|Nifty 50"
NIFTY_OPTION_KEY = "NSE_INDEX|Nifty 50"
BASE_URL         = "https://api.upstox.com/v2"

# ── Trading parameters ────────────────────────────────────────────────────────
ORDER_QUANTITY     = 1         # number of lots
ORDER_PRODUCT      = "I"       # I = Intraday
STOPLOSS_PCT       = 20.0      # % of premium (wider than 15m — 1H moves are bigger)
TARGET_MULTIPLIER  = 2.0       # target = risk × this
NO_NEW_ENTRY_AFTER = "14:30"   # 1H candle at 14:30 still has time to work
MARKET_CLOSE_TIME  = "15:25"   # force-exit everything
SCAN_INTERVAL_SECS = 60        # 1-min scan — no point scanning more often on 1H

# ── Heikin-Ashi candle filters ────────────────────────────────────────────────
SMALL_LOWER_SHADOW_MAX = 0.30  # for CE: lower wick / range must be ≤ this
SMALL_UPPER_SHADOW_MAX = 0.30  # for PE: upper wick / range must be ≤ this
MIN_BODY_RATIO         = 0.25  # skip doji candles (body < 25% of range)
TRAIL_BUFFER_PCT       = 0.10  # HA trail buffer (10% of 1H HA candle range)

# ── EMA settings ─────────────────────────────────────────────────────────────
EMA_FAST   = 9    # fast EMA period
EMA_SLOW   = 21   # slow EMA period
EMA_TREND  = 50   # trend filter EMA (price must be above/below this)

# ── RSI settings ─────────────────────────────────────────────────────────────
RSI_PERIOD      = 14
RSI_MIN_LONG    = 52    # RSI must be above this for CE entries
RSI_MAX_SHORT   = 48    # RSI must be below this for PE entries
RSI_OVERBOUGHT  = 70    # CE entry blocked if RSI above this
RSI_OVERSOLD    = 30    # PE entry blocked if RSI below this

# ── Strike selection ──────────────────────────────────────────────────────────
# Try to get slightly OTM (0.45–0.55 delta proxy via spot offset pcts)
# If not found, falls back to ATM. Adjust for your premium preference.
STRIKE_OFFSET_PCTS = [0.0, 0.5, -0.5, 1.0]  # % offset from spot to try

# ── India VIX guard (optional) ────────────────────────────────────────────────
# Set VIX_MAX = None to disable. Otherwise entries blocked when VIX > VIX_MAX.
VIX_MAX = 22.0

# ── Risk controls ─────────────────────────────────────────────────────────────
MAX_DAILY_LOSS_ABS  = 6000.0   # stop entries if DAILY_PNL drops below -6000
MAX_TRADES_PER_DAY  = 4        # 1H gives fewer bars — allow 4 trades

# ── Timing ───────────────────────────────────────────────────────────────────
# Only scan for entries between these times (1H bar closes at :15, :30 etc)
ENTRY_START = "10:30"
ENTRY_END   = "14:30"

# ── Option chain cache ────────────────────────────────────────────────────────
OPTION_CHAIN_CACHE_TTL = 600   # 10 minutes (less frequent than 15m strategy)

# ── Misc ──────────────────────────────────────────────────────────────────────
ENABLE_AUTO_TRADING = True
DEBUG_MODE          = True
DATA_LAG_SECONDS    = 90       # wait this long after 1H close before acting
LOG_FILE            = "nifty_ema_ha_1h_strategy.txt"
CSV_FILE            = "nifty_ema_ha_1h_trades.csv"

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────────────

ACTIVE_POSITION    = {}
DAILY_PNL          = 0.0
TRADES_TODAY       = 0
LAST_SIGNAL_CANDLE = None   # datetime of last candle that generated an entry (dedup)

OPTION_CHAIN_CACHE: dict       = {}
LAST_CHAIN_FETCH:   Optional[datetime] = None

CSV_HEADERS = [
    "timestamp", "signal", "ema_fast", "ema_slow",
    "rsi", "ha_close", "ha_color",
    "premium", "symbol", "strike", "expiry", "spot", "order_id",
]

# ─────────────────────────────────────────────────────────────────────────────
# HTTP HELPERS  (identical to v4 script)
# ─────────────────────────────────────────────────────────────────────────────

def _headers():
    return {"Accept": "application/json",
            "Authorization": f"Bearer {ACCESS_TOKEN}"}

def _order_headers():
    return {"Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ACCESS_TOKEN}"}

# ─────────────────────────────────────────────────────────────────────────────
# CANDLE DATA
# ─────────────────────────────────────────────────────────────────────────────

def fetch_1min_intraday() -> Optional[pd.DataFrame]:
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
            df["datetime"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.floor("min")
            return df.sort_values("datetime").reset_index(drop=True)
        if DEBUG_MODE:
            print(f"⚠️  1min fetch HTTP {resp.status_code}")
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚠️  1min fetch error: {e}")
    return None


def resample_to_1h(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Resample 1-min data to 1H bars aligned to the NSE session open (09:15).
    Normalises timestamps to tz-naive minute-floor.
    """
    if df is None or df.empty:
        return None
    src = df.copy()
    src["datetime"] = pd.to_datetime(src["datetime"]).dt.tz_localize(None).dt.floor("min")
    df_idx = src.set_index("datetime")[["open", "high", "low", "close", "volume"]]
    resampled = df_idx.resample("1h", offset="15min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna().reset_index()
    resampled["datetime"] = (pd.to_datetime(resampled["datetime"])
                             .dt.tz_localize(None).dt.floor("min"))
    return resampled


def get_1h_candles() -> Optional[pd.DataFrame]:
    df1 = fetch_1min_intraday()
    if df1 is None:
        return None
    return resample_to_1h(df1)

# ─────────────────────────────────────────────────────────────────────────────
# HEIKIN-ASHI
# ─────────────────────────────────────────────────────────────────────────────

def compute_ha(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Heikin-Ashi OHLC from standard OHLC."""
    from scipy.signal import lfilter
    df = df.copy().reset_index(drop=True)
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    seed = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    zi   = np.array([seed])
    ha_open_arr, _ = lfilter([0, 0.5], [1, -0.5], ha_close.to_numpy(), zi=zi)
    ha_open  = pd.Series(ha_open_arr, dtype=float)
    ha_high  = np.maximum.reduce([df["high"].values, ha_open.values, ha_close.values])
    ha_low   = np.minimum.reduce([df["low"].values, ha_open.values, ha_close.values])
    df["ha_open"]  = ha_open.values
    df["ha_high"]  = ha_high
    df["ha_low"]   = ha_low
    df["ha_close"] = ha_close.values
    df["ha_color"] = np.where(df["ha_close"] >= df["ha_open"], "green", "red")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs    = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add EMA fast/slow/trend and RSI to the 1H DataFrame.
    Uses HA close for EMAs (smoother trend signal) and raw close for RSI.
    """
    df = df.copy()
    ha = compute_ha(df)
    df["ha_open"]  = ha["ha_open"]
    df["ha_high"]  = ha["ha_high"]
    df["ha_low"]   = ha["ha_low"]
    df["ha_close"] = ha["ha_close"]
    df["ha_color"] = ha["ha_color"]

    # EMAs on HA close for cleaner crossover
    df["ema_fast"]  = ema(df["ha_close"], EMA_FAST)
    df["ema_slow"]  = ema(df["ha_close"], EMA_SLOW)
    df["ema_trend"] = ema(df["ha_close"], EMA_TREND)

    # RSI on raw close (standard)
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    return df


def has_ema_crossover(df: pd.DataFrame) -> str:
    """
    Detect EMA crossover on the last TWO closed candles.
    Returns 'bullish', 'bearish', or 'none'.
    A crossover requires:
      bullish: prev fast <= prev slow  AND  last fast > last slow
      bearish: prev fast >= prev slow  AND  last fast < last slow
    """
    if len(df) < 2:
        return "none"
    prev = df.iloc[-2]
    last = df.iloc[-1]
    if (prev["ema_fast"] <= prev["ema_slow"] and
            last["ema_fast"] > last["ema_slow"]):
        return "bullish"
    if (prev["ema_fast"] >= prev["ema_slow"] and
            last["ema_fast"] < last["ema_slow"]):
        return "bearish"
    return "none"

# ─────────────────────────────────────────────────────────────────────────────
# INDIA VIX
# ─────────────────────────────────────────────────────────────────────────────

def get_india_vix() -> Optional[float]:
    """Fetch India VIX LTP via Upstox market-quote."""
    if VIX_MAX is None:
        return None
    try:
        resp = requests.get(
            f"{BASE_URL}/market-quote/ltp",
            headers=_headers(),
            params={"instrument_key": "NSE_INDEX|India VIX"},
            timeout=10,
        )
        if resp.status_code == 200:
            inner = resp.json().get("data", {})
            for v in inner.values():
                ltp = v.get("last_price")
                if ltp:
                    return float(ltp)
    except Exception:
        pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def scan_1h_for_entry(df_1h: pd.DataFrame) -> Optional[dict]:
    """
    Core signal scanner for the 1H HA + EMA crossover strategy.

    Requires at least EMA_TREND + 5 bars of history for stable EMAs.
    Only acts on a fully-closed 1H candle (iloc[-2]; iloc[-1] is forming).

    Checklist (CE):
      ✓ Last CLOSED 1H HA candle is GREEN
      ✓ Small lower wick (≤ SMALL_LOWER_SHADOW_MAX)
      ✓ Meaningful body (≥ MIN_BODY_RATIO)
      ✓ Bullish EMA crossover on this candle (fast crossed above slow)
      ✓ HA close above EMA trend
      ✓ RSI between RSI_MIN_LONG and RSI_OVERBOUGHT

    Checklist (PE):
      ✓ Last CLOSED 1H HA candle is RED
      ✓ Small upper wick (≤ SMALL_UPPER_SHADOW_MAX)
      ✓ Meaningful body (≥ MIN_BODY_RATIO)
      ✓ Bearish EMA crossover on this candle
      ✓ HA close below EMA trend
      ✓ RSI between RSI_OVERSOLD and RSI_MAX_SHORT

    Returns a signal dict or None.
    """
    if df_1h is None or len(df_1h) < EMA_TREND + 5:
        if DEBUG_MODE:
            bars = len(df_1h) if df_1h is not None else 0
            print(f"   ⏳ Waiting for indicator warmup ({bars}/{EMA_TREND + 5} 1H bars)")
        return None

    # Daily guards
    if DAILY_PNL <= -MAX_DAILY_LOSS_ABS:
        if DEBUG_MODE:
            print(f"   🛑 Daily loss limit hit (Rs {DAILY_PNL:+.0f}) — no new entries")
        return None
    if TRADES_TODAY >= MAX_TRADES_PER_DAY:
        if DEBUG_MODE:
            print(f"   🛑 Max {MAX_TRADES_PER_DAY} trades/day reached")
        return None

    # Entry window
    t_now = datetime.now().strftime("%H:%M")
    if not (ENTRY_START <= t_now <= ENTRY_END):
        if DEBUG_MODE:
            print(f"   ⏰ Outside entry window ({ENTRY_START}–{ENTRY_END})")
        return None

    df = add_indicators(df_1h)

    # Use the last CLOSED candle (iloc[-2]); iloc[-1] is still forming
    c = df.iloc[-2]   # closed signal candle
    p = df.iloc[-3]   # candle before it (for crossover check)

    candle_time = c["datetime"]

    # Dedup — don't re-enter on the same closed candle
    if LAST_SIGNAL_CANDLE is not None and candle_time <= LAST_SIGNAL_CANDLE:
        return None

    # EMA crossover on the closed candle
    crossover = "none"
    if (p["ema_fast"] <= p["ema_slow"] and c["ema_fast"] > c["ema_slow"]):
        crossover = "bullish"
    elif (p["ema_fast"] >= p["ema_slow"] and c["ema_fast"] < c["ema_slow"]):
        crossover = "bearish"

    ha_range   = c["ha_high"] - c["ha_low"]
    body       = abs(c["ha_close"] - c["ha_open"])
    body_ratio = (body / ha_range) if ha_range > 0.001 else 0

    if DEBUG_MODE:
        print(f"   1H candle [{candle_time.strftime('%H:%M')}] "
              f"color={c['ha_color']} | "
              f"EMA9={c['ema_fast']:.1f} EMA21={c['ema_slow']:.1f} EMA50={c['ema_trend']:.1f} | "
              f"RSI={c['rsi']:.1f} | crossover={crossover} | body={body_ratio:.0%}")

    # ── CE signal ─────────────────────────────────────────────────────────────
    if crossover == "bullish" and c["ha_color"] == "green":
        # Body filter
        if body_ratio < MIN_BODY_RATIO:
            if DEBUG_MODE:
                print(f"   ⛔ CE blocked: doji (body {body_ratio:.0%} < {MIN_BODY_RATIO:.0%})")
            return None
        # Wick filter
        lower_shadow = (c["ha_open"] - c["ha_low"]) / ha_range
        if lower_shadow > SMALL_LOWER_SHADOW_MAX:
            if DEBUG_MODE:
                print(f"   ⛔ CE blocked: lower wick too long ({lower_shadow:.0%})")
            return None
        # Trend filter
        if c["ha_close"] <= c["ema_trend"]:
            if DEBUG_MODE:
                print(f"   ⛔ CE blocked: HA close {c['ha_close']:.1f} <= EMA50 {c['ema_trend']:.1f}")
            return None
        # RSI filter
        if not (RSI_MIN_LONG <= c["rsi"] <= RSI_OVERBOUGHT):
            if DEBUG_MODE:
                print(f"   ⛔ CE blocked: RSI {c['rsi']:.1f} not in [{RSI_MIN_LONG}, {RSI_OVERBOUGHT}]")
            return None

        print(f"   ✅ CE SIGNAL: green HA + bullish EMA cross | "
              f"EMA9={c['ema_fast']:.1f} > EMA21={c['ema_slow']:.1f} | "
              f"RSI={c['rsi']:.1f} | close={c['ha_close']:.1f}")
        return {
            "signal":          "CE",
            "candle_time":     candle_time,
            "ha_close":        c["ha_close"],
            "ha_color":        "green",
            "ema_fast":        c["ema_fast"],
            "ema_slow":        c["ema_slow"],
            "ema_trend":       c["ema_trend"],
            "rsi":             c["rsi"],
            "lower_shadow_pct": lower_shadow,
        }

    # ── PE signal ─────────────────────────────────────────────────────────────
    if crossover == "bearish" and c["ha_color"] == "red":
        # Body filter
        if body_ratio < MIN_BODY_RATIO:
            if DEBUG_MODE:
                print(f"   ⛔ PE blocked: doji (body {body_ratio:.0%} < {MIN_BODY_RATIO:.0%})")
            return None
        # Wick filter
        upper_shadow = (c["ha_high"] - c["ha_open"]) / ha_range
        if upper_shadow > SMALL_UPPER_SHADOW_MAX:
            if DEBUG_MODE:
                print(f"   ⛔ PE blocked: upper wick too long ({upper_shadow:.0%})")
            return None
        # Trend filter
        if c["ha_close"] >= c["ema_trend"]:
            if DEBUG_MODE:
                print(f"   ⛔ PE blocked: HA close {c['ha_close']:.1f} >= EMA50 {c['ema_trend']:.1f}")
            return None
        # RSI filter
        if not (RSI_OVERSOLD <= c["rsi"] <= RSI_MAX_SHORT):
            if DEBUG_MODE:
                print(f"   ⛔ PE blocked: RSI {c['rsi']:.1f} not in [{RSI_OVERSOLD}, {RSI_MAX_SHORT}]")
            return None

        print(f"   ✅ PE SIGNAL: red HA + bearish EMA cross | "
              f"EMA9={c['ema_fast']:.1f} < EMA21={c['ema_slow']:.1f} | "
              f"RSI={c['rsi']:.1f} | close={c['ha_close']:.1f}")
        return {
            "signal":          "PE",
            "candle_time":     candle_time,
            "ha_close":        c["ha_close"],
            "ha_color":        "red",
            "ema_fast":        c["ema_fast"],
            "ema_slow":        c["ema_slow"],
            "ema_trend":       c["ema_trend"],
            "rsi":             c["rsi"],
            "upper_shadow_pct": upper_shadow,
        }

    return None


def check_ha_reversal_exit(df_1h: pd.DataFrame) -> bool:
    """
    Exit trigger: opposite HA candle on the 1H chart.
    CE position  → exit if last closed candle is RED
    PE position  → exit if last closed candle is GREEN

    This is checked every scan alongside the SL/target checks.
    """
    if not ACTIVE_POSITION or df_1h is None or len(df_1h) < 2:
        return False
    df = compute_ha(df_1h)
    last_color = df.iloc[-2]["ha_color"]   # last closed candle
    opt_type   = ACTIVE_POSITION.get("option_type")
    if opt_type == "CE" and last_color == "red":
        print("   🔄 CE exit trigger: 1H HA turned RED")
        return True
    if opt_type == "PE" and last_color == "green":
        print("   🔄 PE exit trigger: 1H HA turned GREEN")
        return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
# MARKET DATA  (identical to v4 script)
# ─────────────────────────────────────────────────────────────────────────────

def get_nifty_spot() -> Optional[float]:
    url = f"{BASE_URL}/market-quote/ltp"
    try:
        resp = requests.get(url, headers=_headers(),
                            params={"instrument_key": NIFTY_INDEX_KEY}, timeout=15)
        if resp.status_code == 200:
            inner = resp.json().get("data", {})
            for v in inner.values():
                ltp = v.get("last_price")
                if ltp:
                    return float(ltp)
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚠️  Spot fetch error: {e}")
    return None


def get_nifty_option_chain(option_type: str) -> list:
    url = f"{BASE_URL}/option/contract"
    try:
        resp = requests.get(url, headers=_headers(),
                            params={"instrument_key": NIFTY_OPTION_KEY}, timeout=20)
        if resp.status_code != 200:
            return []
        contracts = resp.json().get("data", [])
        today = datetime.now().date()
        result = []
        for c in contracts:
            if c.get("instrument_type") != option_type:
                continue
            try:
                exp = datetime.strptime(c["expiry"], "%Y-%m-%d").date()
                if exp <= today:
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


def select_contract(contracts: list, spot: float, option_type: str) -> Optional[dict]:
    """
    Select nearest-expiry contract targeting 0.45–0.55 delta proxy.
    For CE: tries strikes slightly above spot (OTM).
    For PE: tries strikes slightly below spot (OTM).
    Falls back to ATM if no OTM found.
    """
    if not contracts:
        return None
    nearest_expiry = contracts[0]["_expiry_date"]
    nearest = [c for c in contracts if c["_expiry_date"] == nearest_expiry]

    # Round spot to nearest 50 (Nifty strike increment)
    atm_strike = round(spot / 50) * 50

    for pct in STRIKE_OFFSET_PCTS:
        offset = round((spot * pct / 100) / 50) * 50
        target_strike = atm_strike + offset
        for c in nearest:
            if abs(c["strike_price"] - target_strike) <= 25:
                return c

    # Fallback: pure ATM
    return min(nearest, key=lambda c: abs(c["strike_price"] - spot))


def get_ltp(instrument_key: str) -> Optional[float]:
    url = f"{BASE_URL}/market-quote/ltp"
    try:
        resp = requests.get(url, headers=_headers(),
                            params={"instrument_key": instrument_key}, timeout=15)
        if resp.status_code == 200:
            inner = resp.json().get("data", {})
            for v in inner.values():
                ltp = v.get("last_price")
                if ltp:
                    return float(ltp)
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚠️  LTP error: {e}")
    return None


def place_order(instrument_key: str, qty: int, txn_type: str,
                order_type: str, price: float = 0,
                trigger_price: float = 0) -> Optional[str]:
    url     = f"{BASE_URL}/order/place"
    payload = {
        "quantity":           qty,
        "product":            ORDER_PRODUCT,
        "validity":           "DAY",
        "price":              price,
        "tag":                "EMA_HA_BOT",
        "instrument_key":     instrument_key,
        "order_type":         order_type.upper(),
        "transaction_type":   txn_type.upper(),
        "disclosed_quantity": 0,
        "trigger_price":      trigger_price,
        "is_amo":             False,
    }
    try:
        resp = requests.post(url, headers=_order_headers(), json=payload, timeout=15)
        if DEBUG_MODE:
            print(f"   📤 Order ({resp.status_code}): {resp.text[:300]}")
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                return data.get("data", {}).get("order_id")
    except Exception as e:
        print(f"   ❌ Order error: {e}")
    return None


def _cancel_order(order_id: str) -> bool:
    url = f"{BASE_URL}/order/cancel"
    try:
        resp = requests.delete(url, headers=_order_headers(),
                               params={"order_id": order_id}, timeout=10)
        if resp.status_code == 200 and resp.json().get("status") == "success":
            if DEBUG_MODE:
                print(f"   🗑️  Cancelled order {order_id}")
            return True
    except Exception:
        pass
    return False

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def execute_entry(signal: dict):
    global ACTIVE_POSITION, LAST_SIGNAL_CANDLE, OPTION_CHAIN_CACHE, LAST_CHAIN_FETCH

    option_type = signal["signal"]
    print(f"\n{'='*70}")
    print(f"🎯 ENTRY: {option_type} | candle {signal['candle_time'].strftime('%H:%M')} "
          f"| {datetime.now().strftime('%H:%M:%S')}")
    print(f"   HA close={signal['ha_close']:.1f} | "
          f"EMA9={signal['ema_fast']:.1f} EMA21={signal['ema_slow']:.1f} | "
          f"RSI={signal['rsi']:.1f}")

    spot = get_nifty_spot()
    if not spot:
        print("   ❌ Cannot fetch spot — aborting")
        return

    # Cached option chain
    cache_stale = (
        LAST_CHAIN_FETCH is None
        or (datetime.now() - LAST_CHAIN_FETCH).total_seconds() >= OPTION_CHAIN_CACHE_TTL
        or option_type not in OPTION_CHAIN_CACHE
    )
    if cache_stale:
        contracts = get_nifty_option_chain(option_type)
        OPTION_CHAIN_CACHE[option_type] = contracts
        LAST_CHAIN_FETCH = datetime.now()
    else:
        contracts = OPTION_CHAIN_CACHE[option_type]

    contract = select_contract(contracts, spot, option_type)
    if not contract:
        print(f"   ❌ No {option_type} contracts — aborting")
        return

    premium = get_ltp(contract["instrument_key"])
    if not premium or premium <= 0:
        print(f"   ❌ Cannot fetch premium — aborting")
        return

    lot_size  = contract.get("lot_size", 50)
    total_qty = lot_size * ORDER_QUANTITY
    limit_px  = round(premium * 1.02, 2)
    sl_trigger = round(premium * (1 - STOPLOSS_PCT / 100), 2)
    sl_limit   = round(sl_trigger * 0.995, 2)
    target     = round(premium * (1 + (STOPLOSS_PCT / 100) * TARGET_MULTIPLIER), 2)

    print(f"   Option:  {contract.get('trading_symbol')}")
    print(f"   Strike:  {contract['strike_price']} | Expiry: {contract['expiry']}")
    print(f"   Spot:    {spot:.1f} | Premium: ₹{premium:.2f}")
    print(f"   Entry:   ₹{limit_px:.2f} | SL: ₹{sl_trigger:.2f} | Target: ₹{target:.2f}")
    print(f"   Risk:    ₹{(premium - sl_trigger) * total_qty:.0f} | "
          f"Reward: ₹{(target - premium) * total_qty:.0f}")

    if not ENABLE_AUTO_TRADING:
        print("   ℹ️  Signal-only mode")
        _log_signal(signal, contract, premium, spot, order_id="SIGNAL_ONLY")
        LAST_SIGNAL_CANDLE = signal["candle_time"]
        return

    order_id = place_order(contract["instrument_key"], total_qty,
                           "BUY", "LIMIT", limit_px)
    if not order_id:
        print("   ❌ BUY order failed — aborting")
        return
    print(f"   ✅ BUY: {order_id}")

    sl_id = place_order(contract["instrument_key"], total_qty,
                        "SELL", "SL", sl_limit, sl_trigger)
    if sl_id:
        print(f"   🛡️  SL: {sl_id}")
    else:
        print("   ⚠️  SL order failed — unprotected")

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
    LAST_SIGNAL_CANDLE = signal["candle_time"]
    _log_signal(signal, contract, premium, spot, order_id=order_id)
    print(f"{'='*70}\n")


def modify_sl_order(new_trigger: float, new_limit: float) -> bool:
    global ACTIVE_POSITION
    if not ENABLE_AUTO_TRADING:
        return False
    old_sl_id = ACTIVE_POSITION.get("sl_order_id")
    if old_sl_id:
        _cancel_order(old_sl_id)
        ACTIVE_POSITION["sl_order_id"] = None
    new_sl_id = place_order(
        ACTIVE_POSITION["instrument_key"], ACTIVE_POSITION["quantity"],
        "SELL", "SL", new_limit, new_trigger,
    )
    if new_sl_id:
        ACTIVE_POSITION["sl_order_id"] = new_sl_id
        if DEBUG_MODE:
            print(f"   🔄 SL updated: {new_sl_id} (₹{new_trigger:.2f})")
        return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
# EXIT MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ha_trail_1h(df_1h: pd.DataFrame) -> Optional[float]:
    """HA-based trailing stop using the last closed 1H HA candle."""
    if df_1h is None or len(df_1h) < 2:
        return None
    ha = compute_ha(df_1h)
    if len(ha) < 2:
        return None
    prev     = ha.iloc[-2]
    ha_range = prev["ha_high"] - prev["ha_low"]
    buffer   = TRAIL_BUFFER_PCT * ha_range
    opt_type = ACTIVE_POSITION.get("option_type")
    if opt_type == "CE":
        return round(prev["ha_low"] + buffer, 2)
    elif opt_type == "PE":
        return round(prev["ha_high"] - buffer, 2)
    return None


def check_exit(df_1h: Optional[pd.DataFrame] = None):
    global ACTIVE_POSITION, DAILY_PNL, TRADES_TODAY

    if not ACTIVE_POSITION:
        return

    current_px = get_ltp(ACTIVE_POSITION["instrument_key"])
    if not current_px:
        return

    entry_px = ACTIVE_POSITION["entry_price"]
    target   = ACTIVE_POSITION["target"]
    qty      = ACTIVE_POSITION["quantity"]
    pnl      = (current_px - entry_px) * qty
    pnl_pct  = (current_px - entry_px) / entry_px * 100
    now_str  = datetime.now().strftime("%H:%M")
    opt_type = ACTIVE_POSITION["option_type"]
    exit_rsn = None

    # ── HA-based trailing stop (1H) ───────────────────────────────────────────
    if df_1h is not None:
        ha_trail = _compute_ha_trail_1h(df_1h)
        if ha_trail is not None:
            prev_sl = ACTIVE_POSITION["sl_trigger"]
            sl_improved = False
            if opt_type == "CE" and ha_trail > prev_sl:
                ACTIVE_POSITION["sl_trigger"] = ha_trail
                sl_improved = True
                if DEBUG_MODE:
                    print(f"   🔼 Trail raised: ₹{prev_sl:.2f} → ₹{ha_trail:.2f}")
            elif opt_type == "PE" and ha_trail < prev_sl:
                ACTIVE_POSITION["sl_trigger"] = ha_trail
                sl_improved = True
                if DEBUG_MODE:
                    print(f"   🔽 Trail lowered: ₹{prev_sl:.2f} → ₹{ha_trail:.2f}")
            if sl_improved:
                new_limit = round(ha_trail * (0.995 if opt_type == "CE" else 1.005), 2)
                modify_sl_order(ha_trail, new_limit)

    sl = ACTIVE_POSITION["sl_trigger"]

    # ── HA reversal exit ──────────────────────────────────────────────────────
    if check_ha_reversal_exit(df_1h):
        exit_rsn = "HA_REVERSAL"
    elif opt_type == "CE" and current_px <= sl:
        exit_rsn = "TRAIL_SL_HIT"
    elif opt_type == "PE" and current_px >= sl:
        exit_rsn = "TRAIL_SL_HIT"
    elif opt_type == "CE" and current_px >= target:
        exit_rsn = "TARGET_HIT"
    elif opt_type == "PE" and current_px <= target:
        exit_rsn = "TARGET_HIT"
    elif now_str >= NO_NEW_ENTRY_AFTER and pnl > 0:
        exit_rsn = "TIME_EXIT_PROFIT"    # only time-exit if in profit
    elif now_str >= MARKET_CLOSE_TIME:
        exit_rsn = "FORCE_EOD"

    if not exit_rsn:
        if DEBUG_MODE:
            print(f"   📊 {ACTIVE_POSITION['trading_symbol']} | "
                  f"LTP: ₹{current_px:.2f} | SL: ₹{sl:.2f} | "
                  f"Target: ₹{target:.2f} | P&L: ₹{pnl:+.0f} ({pnl_pct:+.1f}%)")
        return

    print(f"\n{'='*70}")
    print(f"🔚 EXIT: {exit_rsn} | {ACTIVE_POSITION['trading_symbol']}")
    print(f"   Entry: ₹{entry_px:.2f} | Exit LTP: ₹{current_px:.2f}")
    print(f"   P&L:   ₹{pnl:+.0f} ({pnl_pct:+.1f}%)")

    if ENABLE_AUTO_TRADING:
        if ACTIVE_POSITION.get("sl_order_id"):
            _cancel_order(ACTIVE_POSITION["sl_order_id"])
        exit_id = place_order(
            ACTIVE_POSITION["instrument_key"], ACTIVE_POSITION["quantity"],
            "SELL", "MARKET", 0,
        )
        if exit_id:
            print(f"   ✅ Exit order: {exit_id}")
        else:
            print("   ⚠️  Exit order FAILED — close manually!")

    DAILY_PNL    += pnl
    TRADES_TODAY += 1
    _log_exit(ACTIVE_POSITION, current_px, exit_rsn, pnl, pnl_pct)
    ACTIVE_POSITION = {}
    print(f"   Daily P&L: ₹{DAILY_PNL:+.0f} | Trades: {TRADES_TODAY}")
    print(f"{'='*70}\n")

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _log_signal(signal, contract, premium, spot, order_id):
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        signal["signal"],
        round(signal.get("ema_fast", 0), 2),
        round(signal.get("ema_slow", 0), 2),
        round(signal.get("rsi", 0), 1),
        round(signal["ha_close"], 2),
        signal.get("ha_color", ""),
        premium,
        contract.get("trading_symbol"),
        contract["strike_price"],
        contract["expiry"],
        spot,
        order_id,
    ]
    write_header = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(CSV_HEADERS)
        w.writerow(row)
    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"ENTRY: {datetime.now()} | {signal['signal']}\n")
        f.write(f"  EMA9={signal.get('ema_fast', 0):.1f}  EMA21={signal.get('ema_slow', 0):.1f}  "
                f"RSI={signal.get('rsi', 0):.1f}\n")
        f.write(f"  Option: {contract.get('trading_symbol')} | Premium: ₹{premium:.2f}\n")
        f.write(f"  Order ID: {order_id}\n")


def _log_exit(pos, exit_px, reason, pnl, pnl_pct):
    with open(LOG_FILE, "a") as f:
        f.write(f"EXIT: {datetime.now()} | {reason}\n")
        f.write(f"  Entry: ₹{pos['entry_price']:.2f} | Exit: ₹{exit_px:.2f}\n")
        f.write(f"  P&L:   ₹{pnl:+.0f} ({pnl_pct:+.1f}%)\n")
        f.write(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.strftime("%H:%M")
    return "09:15" <= t < MARKET_CLOSE_TIME

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────

def banner():
    print("\n" + "=" * 70)
    print("  NIFTY 50 OPTIONS — 1H HEIKIN-ASHI + EMA CROSSOVER STRATEGY")
    print("=" * 70)
    print(f"  Mode:         {'LIVE TRADING' if ENABLE_AUTO_TRADING else 'SIGNAL ONLY'}")
    print(f"  Timeframe:    1H (resampled from 1-min intraday)")
    print(f"  EMA:          {EMA_FAST}/{EMA_SLOW} crossover | {EMA_TREND} trend filter")
    print(f"  RSI({RSI_PERIOD}):    CE >{RSI_MIN_LONG} & <{RSI_OVERBOUGHT} | PE <{RSI_MAX_SHORT} & >{RSI_OVERSOLD}")
    print(f"  HA filter:    body >{MIN_BODY_RATIO:.0%} | "
          f"lower wick <{SMALL_LOWER_SHADOW_MAX:.0%} (CE) | upper wick <{SMALL_UPPER_SHADOW_MAX:.0%} (PE)")
    print(f"  SL:           {STOPLOSS_PCT}% | Target: {TARGET_MULTIPLIER}x risk | Trail: {TRAIL_BUFFER_PCT:.0%} HA buffer")
    print(f"  Entry window: {ENTRY_START}–{ENTRY_END} | Force-exit: {MARKET_CLOSE_TIME}")
    print(f"  Max trades:   {MAX_TRADES_PER_DAY}/day | Loss cap: ₹{MAX_DAILY_LOSS_ABS:,.0f}")
    print(f"  VIX guard:    {'<' + str(VIX_MAX) if VIX_MAX else 'OFF'}")
    print(f"  Exit rules:   HA reversal | SL | Target | Time | Trail")
    print(f"  Chain cache:  {OPTION_CHAIN_CACHE_TTL}s TTL")
    print("=" * 70 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global ACTIVE_POSITION, DAILY_PNL, TRADES_TODAY, LAST_SIGNAL_CANDLE
    global OPTION_CHAIN_CACHE, LAST_CHAIN_FETCH

    banner()
    scan_count = 0

    while True:
        try:
            now = datetime.now()
            t   = now.strftime("%H:%M")

            if not is_market_open():
                if t >= MARKET_CLOSE_TIME:
                    print(f"\n📊 Session closed | Daily P&L: ₹{DAILY_PNL:+.0f} | "
                          f"Trades: {TRADES_TODAY}")
                    # EOD reset
                    ACTIVE_POSITION    = {}
                    DAILY_PNL          = 0.0
                    TRADES_TODAY       = 0
                    LAST_SIGNAL_CANDLE = None
                    OPTION_CHAIN_CACHE.clear()
                    LAST_CHAIN_FETCH   = None
                    time.sleep(300)
                    continue
                time.sleep(30)
                continue

            # Only act after enough bars for indicator warmup
            if t < ENTRY_START:
                print(f"⏳ {t} — waiting for entry window ({ENTRY_START})")
                time.sleep(30)
                continue

            scan_count += 1
            df_1h = get_1h_candles()
            n_bars = len(df_1h) if df_1h is not None else 0

            # ── VIX guard ─────────────────────────────────────────────────────
            vix_ok = True
            if VIX_MAX and not ACTIVE_POSITION:
                vix = get_india_vix()
                if vix and vix > VIX_MAX:
                    print(f"   ⚠️  VIX {vix:.1f} > {VIX_MAX} — skipping entries")
                    vix_ok = False

            print(f"🔍 Scan #{scan_count} | {now.strftime('%H:%M:%S')} | "
                  f"1H bars: {n_bars} | "
                  f"Trades: {TRADES_TODAY}/{MAX_TRADES_PER_DAY} | "
                  f"P&L: ₹{DAILY_PNL:+.0f} | "
                  f"Position: {'YES' if ACTIVE_POSITION else 'none'}",
                  flush=True)

            # ── Exit check ────────────────────────────────────────────────────
            if ACTIVE_POSITION:
                check_exit(df_1h)

            # ── Entry scan ────────────────────────────────────────────────────
            elif vix_ok and t < NO_NEW_ENTRY_AFTER:
                signal = scan_1h_for_entry(df_1h)
                if signal:
                    execute_entry(signal)

            # ── EOD force-exit ────────────────────────────────────────────────
            if ACTIVE_POSITION and t >= MARKET_CLOSE_TIME:
                print("⏰ EOD force-exit")
                check_exit(df_1h)

            time.sleep(SCAN_INTERVAL_SECS)

        except KeyboardInterrupt:
            print(f"\n⛔ Stopped by user | Daily P&L: ₹{DAILY_PNL:+.0f}")
            if ACTIVE_POSITION and ENABLE_AUTO_TRADING:
                print("   Closing open position...")
                check_exit(df_1h if 'df_1h' in dir() else None)
            sys.exit(0)
        except Exception as e:
            print(f"⚠️  Loop error: {e}")
            import traceback; traceback.print_exc()
            time.sleep(SCAN_INTERVAL_SECS)


if __name__ == "__main__":
    main()
