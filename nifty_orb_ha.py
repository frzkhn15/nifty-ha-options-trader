#!/usr/bin/env python3
"""
Nifty 50 ORB — Heikin-Ashi Filter  (Live Trading via Upstox API v2)
====================================================================

STRATEGY RULES:
  1. Opening Range  : First 1H HA candle (09:15–10:15) → High1, Low1
  2. Entry signals  : 15m standard candle close > High1 → CE
                      15m standard candle close < Low1  → PE
                      Signals only accepted 10:30–13:00
  3. HA Filter      : Entry allowed only if last CLOSED 1H HA candle
                      agrees with direction (green for CE, red for PE)
  4. Initial SL     : CE → Low1  |  PE → High1
  5. Trailing SL    : 15m swing low (for CE) / swing high (for PE)
                      using ta.pivotlow equivalent in pandas
  6. HA Comfort     : If live 1H HA flips against position, tighten SL
                      to previous 15m candle low/high (one-time per flip)
  7. One trade/day  : First valid signal only; no re-entries after exit
  8. EOD exit       : Force-close everything at 15:00

USAGE:
  export UPSTOX_TOKEN="your_token"
  python nifty_orb_ha.py
"""

import os, sys, time, csv
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional
from scipy.signal import lfilter

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ACCESS_TOKEN = os.environ.get("UPSTOX_TOKEN")
if not ACCESS_TOKEN:
    raise ValueError(
        "\n❌  Set UPSTOX_TOKEN before running:\n"
        "      export UPSTOX_TOKEN='your_token_here'\n"
    )

NIFTY_INDEX_KEY  = "NSE_INDEX|Nifty 50"
NIFTY_OPTION_KEY = "NSE_INDEX|Nifty 50"
BASE_URL         = "https://api.upstox.com/v2"

# Timing
RANGE_CAPTURE_TIME  = "10:15"    # capture ORB after this 15m bar closes
ENTRY_START         = "10:30"    # first valid entry
ENTRY_END           = "13:00"    # no new entries after this
EOD_EXIT_TIME       = "15:00"    # force-exit
SCAN_INTERVAL_SECS  = 30         # scan every 30s (need to catch 15m closes fast)

# Strategy
SWING_BARS          = 2          # pivot lookback on each side
TICK                = 0.05       # stop offset beyond swing point
STOPLOSS_PCT        = None       # None = use ORB levels; set a % as hard backstop
OPTION_LOT_SIZE     = 50         # Nifty lot size (override if different expiry)
ORDER_QUANTITY      = 1          # lots

# Options
ORDER_PRODUCT           = "I"
OPTION_CHAIN_CACHE_TTL  = 600

# Comfort tightening
ENABLE_COMFORT_TIGHTEN  = True

# Misc
ENABLE_AUTO_TRADING = True
DEBUG_MODE          = True
LOG_FILE            = "nifty_orb_ha.txt"
CSV_FILE            = "nifty_orb_ha_trades.csv"

CSV_HEADERS = ["date", "signal", "entry_px", "entry_time",
               "exit_px", "exit_time", "exit_reason", "pnl",
               "orb_high", "orb_low", "symbol", "order_id"]

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────────────

# Opening range
ORB_HIGH: Optional[float] = None
ORB_LOW:  Optional[float] = None
ORB_SET:  bool            = False

# Position
ACTIVE_POSITION: dict     = {}
TRADE_TAKEN:     bool     = False   # one trade per day

# Comfort tightening state
COMFORT_TIGHTENED: bool   = False
LAST_HA1H_COLOR:   str    = "none"  # "green" | "red" | "none"

# Option chain cache
OPTION_CHAIN_CACHE: dict          = {}
LAST_CHAIN_FETCH:   Optional[datetime] = None

# Dedup: last 15m candle bar-time that triggered a check (prevent re-fire)
LAST_CHECKED_BAR: Optional[datetime] = None

# ─────────────────────────────────────────────────────────────────────────────
# HTTP
# ─────────────────────────────────────────────────────────────────────────────

def _h():
    return {"Accept": "application/json",
            "Authorization": f"Bearer {ACCESS_TOKEN}"}

def _oh():
    return {"Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ACCESS_TOKEN}"}

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def fetch_1min() -> Optional[pd.DataFrame]:
    try:
        r = requests.get(
            f"{BASE_URL}/historical-candle/intraday/{NIFTY_INDEX_KEY}/1minute",
            headers=_h(), timeout=20)
        if r.status_code != 200:
            if DEBUG_MODE:
                print(f"⚠️  1min HTTP {r.status_code}")
            return None
        candles = r.json().get("data", {}).get("candles", [])
        if not candles:
            return None
        df = pd.DataFrame(candles,
                          columns=["ts", "open", "high", "low", "close", "vol", "oi"])
        df["datetime"] = (pd.to_datetime(df["ts"])
                          .dt.tz_localize(None).dt.floor("min"))
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚠️  fetch_1min: {e}")
        return None


def get_15m(df1: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Resample 1-min → standard 15-min candles (NSE-aligned at :15 offset)."""
    if df1 is None or df1.empty:
        return None
    idx = (df1.set_index("datetime")[["open", "high", "low", "close", "vol"]]
             .resample("15min", offset="15min")
             .agg({"open": "first", "high": "max", "low": "min",
                   "close": "last", "vol": "sum"})
             .dropna().reset_index())
    idx["datetime"] = idx["datetime"].dt.tz_localize(None).dt.floor("min")
    return idx


def get_1h(df1: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Resample 1-min → standard 1-hour candles."""
    if df1 is None or df1.empty:
        return None
    idx = (df1.set_index("datetime")[["open", "high", "low", "close", "vol"]]
             .resample("1h", offset="15min")
             .agg({"open": "first", "high": "max", "low": "min",
                   "close": "last", "vol": "sum"})
             .dropna().reset_index())
    idx["datetime"] = idx["datetime"].dt.tz_localize(None).dt.floor("min")
    return idx

# ─────────────────────────────────────────────────────────────────────────────
# HEIKIN-ASHI
# ─────────────────────────────────────────────────────────────────────────────

def compute_ha(df: pd.DataFrame) -> pd.DataFrame:
    """Standard Heikin-Ashi with recursive open via IIR filter."""
    df = df.copy().reset_index(drop=True)
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    seed = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    ha_open_arr, _ = lfilter([0, 0.5], [1, -0.5],
                             ha_close.to_numpy(), zi=np.array([seed]))
    ha_open = pd.Series(ha_open_arr, dtype=float)
    df["ha_open"]  = ha_open.values
    df["ha_high"]  = np.maximum.reduce([df["high"].values, ha_open.values, ha_close.values])
    df["ha_low"]   = np.minimum.reduce([df["low"].values,  ha_open.values, ha_close.values])
    df["ha_close"] = ha_close.values
    df["ha_color"] = np.where(df["ha_close"] >= df["ha_open"], "green", "red")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# SWING POINTS  (pandas equivalent of ta.pivotlow / ta.pivothigh)
# ─────────────────────────────────────────────────────────────────────────────

def last_swing_low(df_15m: pd.DataFrame, bars: int = SWING_BARS) -> Optional[float]:
    """
    Most recent confirmed swing low on the standard 15m chart.
    A swing low at index i: low[i] < low[i-1..i-bars] AND low[i] < low[i+1..i+bars]
    Needs at least `bars` candles after the pivot to confirm — so we look
    at iloc[-(bars+1)] as the candidate and check the candles after it.
    """
    if df_15m is None or len(df_15m) < 2 * bars + 1:
        return None
    lows = df_15m["low"].values
    for i in range(len(lows) - bars - 1, bars - 1, -1):
        candidate = lows[i]
        before = lows[max(0, i - bars): i]
        after  = lows[i + 1: i + bars + 1]
        if len(after) < bars:
            continue
        if all(candidate < b for b in before) and all(candidate < a for a in after):
            return float(candidate)
    return None


def last_swing_high(df_15m: pd.DataFrame, bars: int = SWING_BARS) -> Optional[float]:
    """Most recent confirmed swing high on the standard 15m chart."""
    if df_15m is None or len(df_15m) < 2 * bars + 1:
        return None
    highs = df_15m["high"].values
    for i in range(len(highs) - bars - 1, bars - 1, -1):
        candidate = highs[i]
        before = highs[max(0, i - bars): i]
        after  = highs[i + 1: i + bars + 1]
        if len(after) < bars:
            continue
        if all(candidate > b for b in before) and all(candidate > a for a in after):
            return float(candidate)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# OPENING RANGE
# ─────────────────────────────────────────────────────────────────────────────

def capture_orb(df_1h_ha: pd.DataFrame) -> bool:
    """
    Extract High1 / Low1 from the first 1H HA candle (09:15–10:15).
    Returns True if successfully set.
    """
    global ORB_HIGH, ORB_LOW, ORB_SET
    if ORB_SET:
        return True
    if df_1h_ha is None or df_1h_ha.empty:
        return False
    # The first 1H candle starts at 09:15
    first_bar = df_1h_ha[df_1h_ha["datetime"].dt.strftime("%H:%M") == "09:15"]
    if first_bar.empty:
        return False
    ha = compute_ha(first_bar)
    ORB_HIGH = float(ha.iloc[0]["ha_high"])
    ORB_LOW  = float(ha.iloc[0]["ha_low"])
    ORB_SET  = True
    print(f"   📐 ORB captured | High1={ORB_HIGH:.2f}  Low1={ORB_LOW:.2f}")
    return True

# ─────────────────────────────────────────────────────────────────────────────
# 1H HA FILTER  (closed candle colour)
# ─────────────────────────────────────────────────────────────────────────────

def get_last_closed_1h_ha_color(df_1h: pd.DataFrame) -> str:
    """
    Returns colour of the last CLOSED 1H HA candle: 'green', 'red', or 'none'.
    iloc[-1] is the currently forming bar; iloc[-2] is the last closed bar.
    """
    if df_1h is None or len(df_1h) < 2:
        return "none"
    ha = compute_ha(df_1h)
    return str(ha.iloc[-2]["ha_color"])


def get_live_1h_ha_color(df_1h: pd.DataFrame) -> str:
    """
    Returns colour of the CURRENTLY FORMING 1H HA candle (iloc[-1]).
    Used for comfort tightening check.
    """
    if df_1h is None or len(df_1h) < 1:
        return "none"
    ha = compute_ha(df_1h)
    return str(ha.iloc[-1]["ha_color"])

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY SCANNER
# ─────────────────────────────────────────────────────────────────────────────

def scan_entry(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> Optional[dict]:
    """
    Check the last CLOSED 15m standard candle for an ORB breakout.
    Returns signal dict or None.
    """
    global LAST_CHECKED_BAR

    if not ORB_SET or TRADE_TAKEN:
        return None
    if df_15m is None or len(df_15m) < 3:
        return None

    t = datetime.now().strftime("%H:%M")
    if not (ENTRY_START <= t <= ENTRY_END):
        return None

    # Last CLOSED 15m candle = iloc[-2] (iloc[-1] is forming)
    bar      = df_15m.iloc[-2]
    bar_time = bar["datetime"]

    # Dedup: don't re-check the same bar on every 30s scan tick
    if LAST_CHECKED_BAR is not None and bar_time <= LAST_CHECKED_BAR:
        return None

    LAST_CHECKED_BAR = bar_time
    ha_color = get_last_closed_1h_ha_color(df_1h)

    if DEBUG_MODE:
        print(f"   15m bar [{bar_time.strftime('%H:%M')}] "
              f"close={bar['close']:.2f} | "
              f"ORB H={ORB_HIGH:.2f} L={ORB_LOW:.2f} | "
              f"1H HA={ha_color}")

    # CE signal
    if bar["close"] > ORB_HIGH and ha_color == "green":
        print(f"\n   ✅ CE SIGNAL: 15m close {bar['close']:.2f} > ORB High {ORB_HIGH:.2f} | "
              f"1H HA green ✓")
        return {"signal": "CE", "bar_time": bar_time,
                "bar_close": bar["close"], "ha_color": ha_color}

    # PE signal
    if bar["close"] < ORB_LOW and ha_color == "red":
        print(f"\n   ✅ PE SIGNAL: 15m close {bar['close']:.2f} < ORB Low {ORB_LOW:.2f} | "
              f"1H HA red ✓")
        return {"signal": "PE", "bar_time": bar_time,
                "bar_close": bar["close"], "ha_color": ha_color}

    # Filter rejection messages
    if bar["close"] > ORB_HIGH and ha_color != "green":
        if DEBUG_MODE:
            print(f"   ⛔ CE blocked: 1H HA is {ha_color} (need green)")
    if bar["close"] < ORB_LOW and ha_color != "red":
        if DEBUG_MODE:
            print(f"   ⛔ PE blocked: 1H HA is {ha_color} (need red)")

    return None

# ─────────────────────────────────────────────────────────────────────────────
# TRAILING STOP UPDATE
# ─────────────────────────────────────────────────────────────────────────────

def update_trailing_stop(df_15m: pd.DataFrame):
    """
    Ratchet the trailing stop using 15m swing points.
    CE: stop moves UP to (swing_low - TICK).
    PE: stop moves DOWN to (swing_high + TICK).
    Stop only moves in the favourable direction.
    """
    if not ACTIVE_POSITION:
        return

    opt       = ACTIVE_POSITION["type"]
    cur_stop  = ACTIVE_POSITION["stop"]

    if opt == "CE":
        sl = last_swing_low(df_15m)
        if sl is not None:
            new_stop = round(sl - TICK, 2)
            if new_stop > cur_stop:
                ACTIVE_POSITION["stop"] = new_stop
                if DEBUG_MODE:
                    print(f"   🔼 Trail raised: ₹{cur_stop:.2f} → ₹{new_stop:.2f} "
                          f"(swing low {sl:.2f})")
    elif opt == "PE":
        sh = last_swing_high(df_15m)
        if sh is not None:
            new_stop = round(sh + TICK, 2)
            if new_stop < cur_stop:
                ACTIVE_POSITION["stop"] = new_stop
                if DEBUG_MODE:
                    print(f"   🔽 Trail lowered: ₹{cur_stop:.2f} → ₹{new_stop:.2f} "
                          f"(swing high {sh:.2f})")

# ─────────────────────────────────────────────────────────────────────────────
# HA COMFORT TIGHTENING
# ─────────────────────────────────────────────────────────────────────────────

def check_comfort_tighten(df_15m: pd.DataFrame, df_1h: pd.DataFrame):
    """
    If the live (unclosed) 1H HA candle flips against the position,
    tighten the stop to the previous 15m standard candle low/high.
    One-time per flip — resets if HA flips back then flips again.
    """
    global COMFORT_TIGHTENED, LAST_HA1H_COLOR

    if not ENABLE_COMFORT_TIGHTEN or not ACTIVE_POSITION or COMFORT_TIGHTENED:
        return
    if df_15m is None or len(df_15m) < 2:
        return

    live_color = get_live_1h_ha_color(df_1h)
    opt        = ACTIVE_POSITION["type"]
    cur_stop   = ACTIVE_POSITION["stop"]

    # Detect flip: colour must have CHANGED since last check
    if live_color == LAST_HA1H_COLOR:
        return

    LAST_HA1H_COLOR = live_color
    prev_bar = df_15m.iloc[-2]   # last closed 15m standard candle

    if opt == "CE" and live_color == "red":
        tight_stop = round(prev_bar["low"] - TICK, 2)
        if tight_stop > cur_stop:
            ACTIVE_POSITION["stop"] = tight_stop
            COMFORT_TIGHTENED       = True
            print(f"   🟡 Comfort tighten (CE): 1H HA turned RED → "
                  f"stop ₹{cur_stop:.2f} → ₹{tight_stop:.2f} "
                  f"(prev 15m low {prev_bar['low']:.2f})")

    elif opt == "PE" and live_color == "green":
        tight_stop = round(prev_bar["high"] + TICK, 2)
        if tight_stop < cur_stop:
            ACTIVE_POSITION["stop"] = tight_stop
            COMFORT_TIGHTENED       = True
            print(f"   🟡 Comfort tighten (PE): 1H HA turned GREEN → "
                  f"stop ₹{cur_stop:.2f} → ₹{tight_stop:.2f} "
                  f"(prev 15m high {prev_bar['high']:.2f})")

# ─────────────────────────────────────────────────────────────────────────────
# MARKET DATA & ORDERS
# ─────────────────────────────────────────────────────────────────────────────

def get_spot() -> Optional[float]:
    try:
        r = requests.get(f"{BASE_URL}/market-quote/ltp", headers=_h(),
                         params={"instrument_key": NIFTY_INDEX_KEY}, timeout=15)
        if r.status_code == 200:
            for v in r.json().get("data", {}).values():
                if v.get("last_price"):
                    return float(v["last_price"])
    except Exception:
        pass
    return None


def get_ltp(key: str) -> Optional[float]:
    try:
        r = requests.get(f"{BASE_URL}/market-quote/ltp", headers=_h(),
                         params={"instrument_key": key}, timeout=15)
        if r.status_code == 200:
            for v in r.json().get("data", {}).values():
                if v.get("last_price"):
                    return float(v["last_price"])
    except Exception:
        pass
    return None


def get_option_chain(option_type: str) -> list:
    global OPTION_CHAIN_CACHE, LAST_CHAIN_FETCH
    stale = (LAST_CHAIN_FETCH is None
             or (datetime.now() - LAST_CHAIN_FETCH).total_seconds() >= OPTION_CHAIN_CACHE_TTL
             or option_type not in OPTION_CHAIN_CACHE)
    if not stale:
        return OPTION_CHAIN_CACHE[option_type]
    try:
        r = requests.get(f"{BASE_URL}/option/contract", headers=_h(),
                         params={"instrument_key": NIFTY_OPTION_KEY}, timeout=20)
        if r.status_code != 200:
            return []
        today = datetime.now().date()
        result = []
        for c in r.json().get("data", []):
            if c.get("instrument_type") != option_type:
                continue
            try:
                exp = datetime.strptime(c["expiry"], "%Y-%m-%d").date()
                if exp > today:
                    c["_exp"] = exp
                    result.append(c)
            except Exception:
                continue
        result.sort(key=lambda x: x["_exp"])
        OPTION_CHAIN_CACHE[option_type] = result
        LAST_CHAIN_FETCH = datetime.now()
        return result
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚠️  option chain: {e}")
        return []


def select_strike(contracts: list, spot: float, opt: str) -> Optional[dict]:
    """ATM or nearest OTM (rounded to 50) for the nearest expiry."""
    if not contracts:
        return None
    exp      = contracts[0]["_exp"]
    nearest  = [c for c in contracts if c["_exp"] == exp]
    atm      = round(spot / 50) * 50
    # Slight OTM: for CE go 50 above ATM, for PE go 50 below
    otm_strike = atm + 50 if opt == "CE" else atm - 50
    # prefer OTM, fallback to ATM
    for strike in [otm_strike, atm]:
        match = [c for c in nearest
                 if abs(c["strike_price"] - strike) < 26]
        if match:
            return min(match, key=lambda c: abs(c["strike_price"] - strike))
    return min(nearest, key=lambda c: abs(c["strike_price"] - spot))


def place_order(key: str, qty: int, side: str, order_type: str,
                price: float = 0, trigger: float = 0) -> Optional[str]:
    try:
        r = requests.post(f"{BASE_URL}/order/place", headers=_oh(), timeout=15,
                          json={"quantity": qty, "product": ORDER_PRODUCT,
                                "validity": "DAY", "price": price,
                                "tag": "ORB_HA_BOT", "instrument_key": key,
                                "order_type": order_type.upper(),
                                "transaction_type": side.upper(),
                                "disclosed_quantity": 0,
                                "trigger_price": trigger, "is_amo": False})
        if DEBUG_MODE:
            print(f"   📤 Order ({r.status_code}): {r.text[:200]}")
        if r.status_code == 200 and r.json().get("status") == "success":
            return r.json().get("data", {}).get("order_id")
    except Exception as e:
        print(f"   ❌ order: {e}")
    return None


def cancel_order(oid: str):
    try:
        requests.delete(f"{BASE_URL}/order/cancel", headers=_oh(),
                        params={"order_id": oid}, timeout=10)
        if DEBUG_MODE:
            print(f"   🗑️  Cancelled {oid}")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def execute_entry(signal: dict):
    global ACTIVE_POSITION, TRADE_TAKEN, COMFORT_TIGHTENED, LAST_HA1H_COLOR

    opt = signal["signal"]
    print(f"\n{'='*60}")
    print(f"🎯 ORB {opt} ENTRY | bar {signal['bar_time'].strftime('%H:%M')} "
          f"| {datetime.now().strftime('%H:%M:%S')}")
    print(f"   ORB H={ORB_HIGH:.2f}  L={ORB_LOW:.2f} | bar close={signal['bar_close']:.2f}")

    spot = get_spot()
    if not spot:
        print("   ❌ Cannot fetch spot")
        return

    contracts = get_option_chain(opt)
    contract  = select_strike(contracts, spot, opt)
    if not contract:
        print(f"   ❌ No {opt} contracts")
        return

    premium = get_ltp(contract["instrument_key"])
    if not premium or premium <= 0:
        print("   ❌ Cannot fetch premium")
        return

    lot_size  = contract.get("lot_size", OPTION_LOT_SIZE)
    qty       = lot_size * ORDER_QUANTITY
    entry_lmt = round(premium * 1.02, 2)
    # Initial SL = ORB level (Low1 for CE, High1 for PE)
    init_stop = ORB_LOW if opt == "CE" else ORB_HIGH
    sl_lmt    = round(init_stop * 0.995 if opt == "CE"
                      else init_stop * 1.005, 2)

    print(f"   Option:  {contract.get('trading_symbol')}")
    print(f"   Spot:    {spot:.0f} | Strike: {contract['strike_price']} | "
          f"Expiry: {contract['expiry']}")
    print(f"   Premium: ₹{premium:.2f} | Initial SL: ₹{init_stop:.2f} (ORB level)")

    TRADE_TAKEN        = True
    COMFORT_TIGHTENED  = False
    LAST_HA1H_COLOR    = signal["ha_color"]

    if not ENABLE_AUTO_TRADING:
        print("   ℹ️  Signal-only mode")
        ACTIVE_POSITION = {
            "key": contract["instrument_key"],
            "symbol": contract.get("trading_symbol"),
            "type": opt, "entry": premium,
            "stop": init_stop, "qty": qty,
            "sl_id": None, "buy_id": "SIGNAL",
            "time": datetime.now(),
            "signal": signal,
        }
        _log_entry(signal, contract, premium, spot, "SIGNAL_ONLY")
        print(f"{'='*60}\n")
        return

    buy_id = place_order(contract["instrument_key"], qty, "BUY", "LIMIT", entry_lmt)
    if not buy_id:
        print("   ❌ BUY order failed")
        TRADE_TAKEN = False
        return
    print(f"   ✅ BUY: {buy_id}")

    sl_id = place_order(contract["instrument_key"], qty, "SELL", "SL", sl_lmt, init_stop)
    if sl_id:
        print(f"   🛡️  Initial SL: {sl_id} (₹{init_stop:.2f})")
    else:
        print("   ⚠️  SL order failed!")

    ACTIVE_POSITION = {
        "key":    contract["instrument_key"],
        "symbol": contract.get("trading_symbol"),
        "type":   opt,
        "entry":  premium,
        "stop":   init_stop,
        "qty":    qty,
        "sl_id":  sl_id,
        "buy_id": buy_id,
        "time":   datetime.now(),
        "signal": signal,
    }
    _log_entry(signal, contract, premium, spot, buy_id)
    print(f"{'='*60}\n")


def update_exchange_sl():
    """Replace the exchange SL order with the current (trailed/tightened) stop."""
    if not ACTIVE_POSITION or not ENABLE_AUTO_TRADING:
        return
    old_id   = ACTIVE_POSITION.get("sl_id")
    opt      = ACTIVE_POSITION["type"]
    stop     = ACTIVE_POSITION["stop"]
    sl_lmt   = round(stop * 0.995 if opt == "CE" else stop * 1.005, 2)
    if old_id:
        cancel_order(old_id)
    new_id = place_order(ACTIVE_POSITION["key"], ACTIVE_POSITION["qty"],
                         "SELL", "SL", sl_lmt, stop)
    if new_id:
        ACTIVE_POSITION["sl_id"] = new_id
        if DEBUG_MODE:
            print(f"   🔄 SL order updated: {new_id} @ ₹{stop:.2f}")
    else:
        print("   ⚠️  SL update failed — stop is in-memory only!")

# ─────────────────────────────────────────────────────────────────────────────
# EXIT
# ─────────────────────────────────────────────────────────────────────────────

def check_exit(df_15m: Optional[pd.DataFrame] = None,
               df_1h:  Optional[pd.DataFrame] = None):
    global ACTIVE_POSITION

    if not ACTIVE_POSITION:
        return

    ltp = get_ltp(ACTIVE_POSITION["key"])
    if not ltp:
        return

    opt    = ACTIVE_POSITION["type"]
    entry  = ACTIVE_POSITION["entry"]
    stop   = ACTIVE_POSITION["stop"]
    qty    = ACTIVE_POSITION["qty"]
    pnl    = (ltp - entry) * qty
    pnl_p  = (ltp - entry) / entry * 100
    t      = datetime.now().strftime("%H:%M")
    reason = None

    # Update trailing stop (15m swings)
    if df_15m is not None:
        prev_stop = ACTIVE_POSITION["stop"]
        update_trailing_stop(df_15m)
        if ACTIVE_POSITION["stop"] != prev_stop:
            update_exchange_sl()

    # HA comfort tightening check
    if df_1h is not None:
        prev_stop = ACTIVE_POSITION["stop"]
        check_comfort_tighten(df_15m, df_1h)
        if ACTIVE_POSITION["stop"] != prev_stop:
            update_exchange_sl()

    stop = ACTIVE_POSITION["stop"]   # refresh after possible update

    # Hard invalidation: candle CLOSES beyond ORB level
    if df_15m is not None and len(df_15m) >= 2:
        last_close = df_15m.iloc[-2]["close"]
        if opt == "CE" and last_close < ORB_LOW:
            reason = "INVALIDATION_CLOSE_BELOW_LOW1"
        elif opt == "PE" and last_close > ORB_HIGH:
            reason = "INVALIDATION_CLOSE_ABOVE_HIGH1"

    # SL hit (LTP-based for intraday responsiveness)
    if not reason:
        if opt == "CE" and ltp <= stop:
            reason = "SL_HIT"
        elif opt == "PE" and ltp >= stop:
            reason = "SL_HIT"

    # EOD
    if not reason and t >= EOD_EXIT_TIME:
        reason = "EOD_EXIT"

    if not reason:
        if DEBUG_MODE:
            print(f"   📊 {ACTIVE_POSITION['symbol']} | "
                  f"LTP ₹{ltp:.2f} | Stop ₹{stop:.2f} | "
                  f"P&L ₹{pnl:+.0f} ({pnl_p:+.1f}%)")
        return

    print(f"\n{'='*60}")
    print(f"🔚 EXIT: {reason} | {ACTIVE_POSITION['symbol']}")
    print(f"   Entry ₹{entry:.2f} → ₹{ltp:.2f} | P&L ₹{pnl:+.0f} ({pnl_p:+.1f}%)")

    if ENABLE_AUTO_TRADING:
        if ACTIVE_POSITION.get("sl_id"):
            cancel_order(ACTIVE_POSITION["sl_id"])
        eid = place_order(ACTIVE_POSITION["key"], qty, "SELL", "MARKET")
        print(f"   {'✅ ' + eid if eid else '⚠️  Exit FAILED — close manually!'}")

    _log_exit(ACTIVE_POSITION, ltp, reason, pnl, pnl_p)
    ACTIVE_POSITION = {}
    print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _log_entry(signal, contract, premium, spot, order_id):
    with open(LOG_FILE, "a") as f:
        f.write(f"\nENTRY {datetime.now()} | {signal['signal']} | "
                f"bar={signal['bar_time'].strftime('%H:%M')} | "
                f"close={signal['bar_close']:.2f} | "
                f"ORB H={ORB_HIGH:.2f} L={ORB_LOW:.2f} | "
                f"{contract.get('trading_symbol')} ₹{premium:.2f} | "
                f"order={order_id}\n")


def _log_exit(pos, exit_px, reason, pnl, pnl_p):
    row = [date.today().isoformat(), pos["type"],
           pos["entry"], pos["time"].strftime("%H:%M:%S"),
           exit_px, datetime.now().strftime("%H:%M:%S"),
           reason, f"{pnl:+.0f}",
           ORB_HIGH, ORB_LOW,
           pos["symbol"], pos.get("buy_id")]
    first = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if first:
            w.writerow(CSV_HEADERS)
        w.writerow(row)
    with open(LOG_FILE, "a") as f:
        f.write(f"EXIT  {datetime.now()} | {reason} | "
                f"₹{pos['entry']:.2f}→₹{exit_px:.2f} | "
                f"P&L ₹{pnl:+.0f} ({pnl_p:+.1f}%)\n")

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────

def banner():
    print("\n" + "="*58)
    print("  NIFTY ORB — HEIKIN-ASHI FILTER  (Live Trading)")
    print("="*58)
    print(f"  Mode          : {'LIVE' if ENABLE_AUTO_TRADING else 'SIGNAL ONLY'}")
    print(f"  Opening range : 09:15–10:15 (first 1H HA candle)")
    print(f"  Entry window  : {ENTRY_START}–{ENTRY_END} (15m breakout)")
    print(f"  HA filter     : Last closed 1H HA must agree with direction")
    print(f"  Initial SL    : ORB Low1 (CE) / High1 (PE)")
    print(f"  Trailing SL   : 15m swing points (lookback {SWING_BARS} bars each side)")
    print(f"  Comfort tight : {'ON' if ENABLE_COMFORT_TIGHTEN else 'OFF'} "
          f"(1H HA flip → tighten to prev 15m candle)")
    print(f"  EOD exit      : {EOD_EXIT_TIME}")
    print(f"  One trade/day : YES")
    print("="*58 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global ORB_HIGH, ORB_LOW, ORB_SET, TRADE_TAKEN
    global ACTIVE_POSITION, COMFORT_TIGHTENED, LAST_HA1H_COLOR
    global OPTION_CHAIN_CACHE, LAST_CHAIN_FETCH, LAST_CHECKED_BAR

    banner()
    scan = 0

    while True:
        try:
            now = datetime.now()
            t   = now.strftime("%H:%M")

            # Outside market hours
            if now.weekday() >= 5 or t < "09:15":
                time.sleep(30)
                continue

            # EOD reset
            if t >= "15:30":
                print(f"📊 Session done")
                ORB_HIGH = ORB_LOW = None; ORB_SET = False
                TRADE_TAKEN = False; ACTIVE_POSITION = {}
                COMFORT_TIGHTENED = False; LAST_HA1H_COLOR = "none"
                OPTION_CHAIN_CACHE.clear(); LAST_CHAIN_FETCH = None
                LAST_CHECKED_BAR = None
                time.sleep(300)
                continue

            scan += 1
            df1   = fetch_1min()
            df15  = get_15m(df1)
            df1h  = get_1h(df1)
            n15   = len(df15) if df15 is not None else 0
            n1h   = len(df1h) if df1h is not None else 0

            print(f"🔍 #{scan} {now.strftime('%H:%M:%S')} | "
                  f"15m={n15} 1H={n1h} | "
                  f"ORB={'SET' if ORB_SET else 'pending'} | "
                  f"trade={'taken' if TRADE_TAKEN else 'available'} | "
                  f"pos={'YES' if ACTIVE_POSITION else 'no'}",
                  flush=True)

            # ── Step 1: Capture ORB after 10:15 ──────────────────────────────
            if t >= RANGE_CAPTURE_TIME and not ORB_SET:
                capture_orb(df1h)

            # ── Step 2/3: Entry scan ──────────────────────────────────────────
            if ORB_SET and not TRADE_TAKEN and not ACTIVE_POSITION:
                sig = scan_entry(df15, df1h)
                if sig:
                    execute_entry(sig)

            # ── Exit management ───────────────────────────────────────────────
            if ACTIVE_POSITION:
                check_exit(df15, df1h)

            # ── EOD force-exit ────────────────────────────────────────────────
            if ACTIVE_POSITION and t >= EOD_EXIT_TIME:
                print("⏰ EOD force-exit triggered")
                check_exit(df15, df1h)

            time.sleep(SCAN_INTERVAL_SECS)

        except KeyboardInterrupt:
            print("\n⛔ Stopped by user")
            if ACTIVE_POSITION and ENABLE_AUTO_TRADING:
                check_exit()
            sys.exit(0)
        except Exception as e:
            print(f"⚠️  {e}")
            if DEBUG_MODE:
                import traceback; traceback.print_exc()
            time.sleep(SCAN_INTERVAL_SECS)


if __name__ == "__main__":
    main()
