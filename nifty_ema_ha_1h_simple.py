#!/usr/bin/env python3
"""
Nifty 50 Options — 1H Heikin-Ashi + EMA Crossover  (Simplified)
=================================================================

STRATEGY (1H chart):
  LONG (CE):  Green HA candle  +  EMA9 crosses above EMA21  +  RSI > 52
  SHORT (PE): Red  HA candle   +  EMA9 crosses below EMA21  +  RSI < 48

EXIT:
  • Fixed SL   : STOPLOSS_PCT % of premium
  • Target     : TARGET_MULTIPLIER × risk
  • HA reversal: opposite HA colour on 1H → exit immediately
  • EOD        : force-exit at MARKET_CLOSE_TIME

FILTERS KEPT (minimal but effective):
  • EMA50 trend filter  (CE above it, PE below it)
  • Daily loss cap
  • Max trades per day

USAGE:
  export UPSTOX_TOKEN="your_token"
  python nifty_ema_ha_1h_simple.py
"""

import os, sys, time, csv
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (all tunables in one place)
# ─────────────────────────────────────────────────────────────────────────────

ACCESS_TOKEN = os.environ.get("UPSTOX_TOKEN")
if not ACCESS_TOKEN:
    raise ValueError(
        "\n❌  Set UPSTOX_TOKEN before running:\n"
        "      export UPSTOX_TOKEN='your_token_here'\n"
    )

# API
NIFTY_INDEX_KEY  = "NSE_INDEX|Nifty 50"
NIFTY_OPTION_KEY = "NSE_INDEX|Nifty 50"
BASE_URL         = "https://api.upstox.com/v2"

# Trade
ORDER_QUANTITY     = 1       # lots
ORDER_PRODUCT      = "I"     # Intraday
STOPLOSS_PCT       = 20.0    # % of premium
TARGET_MULTIPLIER  = 2.0     # target = SL × this
NO_NEW_ENTRY_AFTER = "14:30"
MARKET_CLOSE_TIME  = "15:25"

# Indicators
EMA_FAST   = 9
EMA_SLOW   = 21
EMA_TREND  = 50   # CE only above this; PE only below
RSI_PERIOD = 14
RSI_LONG   = 52   # RSI must be > this for CE
RSI_SHORT  = 48   # RSI must be < this for PE

# Risk
MAX_DAILY_LOSS_ABS = 6000.0
MAX_TRADES_PER_DAY = 4

# Misc
SCAN_INTERVAL_SECS     = 60
OPTION_CHAIN_CACHE_TTL = 600   # seconds
ENABLE_AUTO_TRADING    = True
DEBUG_MODE             = True
LOG_FILE               = "nifty_1h_simple_strategy.txt"
CSV_FILE               = "nifty_1h_simple_trades.csv"

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────────────

ACTIVE_POSITION    = {}
DAILY_PNL          = 0.0
TRADES_TODAY       = 0
LAST_SIGNAL_CANDLE = None          # prevents re-entry on same 1H candle

OPTION_CHAIN_CACHE: dict                = {}
LAST_CHAIN_FETCH:   Optional[datetime] = None

CSV_HEADERS = ["timestamp", "signal", "ema9", "ema21", "rsi",
               "ha_close", "premium", "symbol", "strike", "expiry",
               "spot", "order_id"]

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
# DATA  —  1-min intraday → resample to 1H
# ─────────────────────────────────────────────────────────────────────────────

def fetch_1min() -> Optional[pd.DataFrame]:
    try:
        r = requests.get(
            f"{BASE_URL}/historical-candle/intraday/{NIFTY_INDEX_KEY}/1minute",
            headers=_h(), timeout=20)
        if r.status_code != 200:
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


def get_1h_candles() -> Optional[pd.DataFrame]:
    """Fetch 1-min data and resample to 1H aligned to NSE open (09:15)."""
    df = fetch_1min()
    if df is None:
        return None
    idx = (df.set_index("datetime")[["open", "high", "low", "close", "vol"]]
             .resample("1h", offset="15min")
             .agg({"open": "first", "high": "max", "low": "min",
                   "close": "last", "vol": "sum"})
             .dropna()
             .reset_index())
    idx["datetime"] = pd.to_datetime(idx["datetime"]).dt.tz_localize(None).dt.floor("min")
    return idx

# ─────────────────────────────────────────────────────────────────────────────
# HEIKIN-ASHI
# ─────────────────────────────────────────────────────────────────────────────

def compute_ha(df: pd.DataFrame) -> pd.DataFrame:
    from scipy.signal import lfilter
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
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add HA candles, EMA 9/21/50, and RSI 14 to a 1H DataFrame."""
    df  = compute_ha(df)                                    # adds ha_* columns
    hac = df["ha_close"]
    df["ema9"]  = hac.ewm(span=EMA_FAST,  adjust=False).mean()
    df["ema21"] = hac.ewm(span=EMA_SLOW,  adjust=False).mean()
    df["ema50"] = hac.ewm(span=EMA_TREND, adjust=False).mean()
    # RSI on raw close
    d       = df["close"].diff()
    gain    = d.clip(lower=0).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    loss    = (-d.clip(upper=0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    return df

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL
# ─────────────────────────────────────────────────────────────────────────────

def scan_for_signal(df_1h: pd.DataFrame) -> Optional[dict]:
    """
    Check the last CLOSED 1H candle (iloc[-2]) for entry signals.

    CE:  green HA  +  EMA9 just crossed above EMA21  +  close > EMA50  +  RSI > 52
    PE:  red  HA   +  EMA9 just crossed below EMA21  +  close < EMA50  +  RSI < 48

    Returns signal dict or None.
    """
    if df_1h is None or len(df_1h) < EMA_TREND + 3:
        if DEBUG_MODE:
            n = len(df_1h) if df_1h is not None else 0
            print(f"   ⏳ Warming up indicators ({n}/{EMA_TREND + 3} bars needed)")
        return None

    # Risk guards
    if DAILY_PNL <= -MAX_DAILY_LOSS_ABS:
        print(f"   🛑 Daily loss cap hit (₹{DAILY_PNL:+.0f})")
        return None
    if TRADES_TODAY >= MAX_TRADES_PER_DAY:
        print(f"   🛑 Max trades/day ({MAX_TRADES_PER_DAY}) reached")
        return None

    df = add_indicators(df_1h)

    # c = last CLOSED candle;  p = candle before it (for crossover)
    c  = df.iloc[-2]
    p  = df.iloc[-3]
    ct = c["datetime"]

    # Skip if we already acted on this candle
    if LAST_SIGNAL_CANDLE is not None and ct <= LAST_SIGNAL_CANDLE:
        return None

    # EMA crossover on the closed candle
    bull_cross = (p["ema9"] <= p["ema21"]) and (c["ema9"] > c["ema21"])
    bear_cross = (p["ema9"] >= p["ema21"]) and (c["ema9"] < c["ema21"])

    if DEBUG_MODE:
        print(f"   [{ct.strftime('%H:%M')}] "
              f"HA={c['ha_color']:5s} | "
              f"EMA9={c['ema9']:.0f} EMA21={c['ema21']:.0f} EMA50={c['ema50']:.0f} | "
              f"RSI={c['rsi']:.1f} | "
              f"cross={'BULL' if bull_cross else 'BEAR' if bear_cross else 'none'}")

    # ── CE ────────────────────────────────────────────────────────────────────
    if bull_cross and c["ha_color"] == "green":
        if c["ha_close"] <= c["ema50"]:
            if DEBUG_MODE:
                print(f"   ⛔ CE: close {c['ha_close']:.0f} <= EMA50 {c['ema50']:.0f}")
            return None
        if c["rsi"] <= RSI_LONG:
            if DEBUG_MODE:
                print(f"   ⛔ CE: RSI {c['rsi']:.1f} <= {RSI_LONG}")
            return None
        print(f"   ✅ CE signal | EMA9 {p['ema9']:.0f}→{c['ema9']:.0f} crossed above EMA21 | RSI {c['rsi']:.1f}")
        return {"signal": "CE", "candle_time": ct,
                "ha_close": c["ha_close"], "ema9": c["ema9"],
                "ema21": c["ema21"], "rsi": c["rsi"]}

    # ── PE ────────────────────────────────────────────────────────────────────
    if bear_cross and c["ha_color"] == "red":
        if c["ha_close"] >= c["ema50"]:
            if DEBUG_MODE:
                print(f"   ⛔ PE: close {c['ha_close']:.0f} >= EMA50 {c['ema50']:.0f}")
            return None
        if c["rsi"] >= RSI_SHORT:
            if DEBUG_MODE:
                print(f"   ⛔ PE: RSI {c['rsi']:.1f} >= {RSI_SHORT}")
            return None
        print(f"   ✅ PE signal | EMA9 {p['ema9']:.0f}→{c['ema9']:.0f} crossed below EMA21 | RSI {c['rsi']:.1f}")
        return {"signal": "PE", "candle_time": ct,
                "ha_close": c["ha_close"], "ema9": c["ema9"],
                "ema21": c["ema21"], "rsi": c["rsi"]}

    return None

# ─────────────────────────────────────────────────────────────────────────────
# ORDER HELPERS
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
        age = int((datetime.now() - LAST_CHAIN_FETCH).total_seconds())
        if DEBUG_MODE:
            print(f"   📋 Chain from cache (age {age}s)")
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
        if DEBUG_MODE:
            print(f"   🔄 Chain refreshed ({len(result)} {option_type} contracts)")
        return result
    except Exception as e:
        if DEBUG_MODE:
            print(f"⚠️  option chain error: {e}")
        return []


def select_atm(contracts: list, spot: float) -> Optional[dict]:
    """Nearest-expiry ATM contract."""
    if not contracts:
        return None
    exp = contracts[0]["_exp"]
    nearest = [c for c in contracts if c["_exp"] == exp]
    return min(nearest, key=lambda c: abs(c["strike_price"] - spot))


def place_order(key: str, qty: int, side: str,
                order_type: str, price: float = 0,
                trigger: float = 0) -> Optional[str]:
    try:
        r = requests.post(f"{BASE_URL}/order/place", headers=_oh(), timeout=15,
                          json={"quantity": qty, "product": ORDER_PRODUCT,
                                "validity": "DAY", "price": price,
                                "tag": "EMA_HA_1H", "instrument_key": key,
                                "order_type": order_type.upper(),
                                "transaction_type": side.upper(),
                                "disclosed_quantity": 0,
                                "trigger_price": trigger, "is_amo": False})
        if DEBUG_MODE:
            print(f"   📤 Order ({r.status_code}): {r.text[:200]}")
        if r.status_code == 200:
            d = r.json()
            if d.get("status") == "success":
                return d.get("data", {}).get("order_id")
    except Exception as e:
        print(f"   ❌ place_order: {e}")
    return None


def cancel_order(oid: str):
    try:
        requests.delete(f"{BASE_URL}/order/cancel", headers=_oh(),
                        params={"order_id": oid}, timeout=10)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────────────────────────────────────

def execute_entry(signal: dict):
    global ACTIVE_POSITION, LAST_SIGNAL_CANDLE

    opt = signal["signal"]
    print(f"\n{'='*60}")
    print(f"🎯 {opt} ENTRY | candle {signal['candle_time'].strftime('%H:%M')} "
          f"| {datetime.now().strftime('%H:%M:%S')}")
    print(f"   EMA9={signal['ema9']:.0f} EMA21={signal['ema21']:.0f} RSI={signal['rsi']:.1f}")

    spot = get_spot()
    if not spot:
        print("   ❌ Cannot fetch spot")
        return

    contracts = get_option_chain(opt)
    contract  = select_atm(contracts, spot)
    if not contract:
        print(f"   ❌ No {opt} contracts found")
        return

    premium = get_ltp(contract["instrument_key"])
    if not premium or premium <= 0:
        print("   ❌ Cannot fetch premium")
        return

    lot_size  = contract.get("lot_size", 50)
    qty       = lot_size * ORDER_QUANTITY
    entry_lmt = round(premium * 1.02, 2)          # 2% slippage buffer
    sl        = round(premium * (1 - STOPLOSS_PCT / 100), 2)
    sl_lmt    = round(sl * 0.995, 2)
    target    = round(premium * (1 + (STOPLOSS_PCT / 100) * TARGET_MULTIPLIER), 2)
    risk_amt  = round((premium - sl) * qty, 0)
    rwd_amt   = round((target - premium) * qty, 0)

    print(f"   Option:  {contract.get('trading_symbol')}")
    print(f"   Spot:    {spot:.0f} | Strike: {contract['strike_price']} | Expiry: {contract['expiry']}")
    print(f"   Premium: ₹{premium:.2f} | Lots: {ORDER_QUANTITY} × {lot_size} = {qty} qty")
    print(f"   SL:      ₹{sl:.2f} | Target: ₹{target:.2f}")
    print(f"   Risk:    ₹{risk_amt:,.0f} | Reward: ₹{rwd_amt:,.0f}")

    LAST_SIGNAL_CANDLE = signal["candle_time"]

    if not ENABLE_AUTO_TRADING:
        print("   ℹ️  Signal-only mode — no order placed")
        _log(signal, contract, premium, spot, "SIGNAL_ONLY")
        return

    buy_id = place_order(contract["instrument_key"], qty, "BUY", "LIMIT", entry_lmt)
    if not buy_id:
        print("   ❌ BUY failed")
        return
    print(f"   ✅ BUY: {buy_id}")

    sl_id = place_order(contract["instrument_key"], qty, "SELL", "SL", sl_lmt, sl)
    if sl_id:
        print(f"   🛡️  SL:  {sl_id}")
    else:
        print("   ⚠️  SL order failed — position unprotected!")

    ACTIVE_POSITION = {
        "key":     contract["instrument_key"],
        "symbol":  contract.get("trading_symbol"),
        "type":    opt,
        "entry":   premium,
        "sl":      sl,
        "target":  target,
        "qty":     qty,
        "sl_id":   sl_id,
        "buy_id":  buy_id,
        "time":    datetime.now(),
    }
    _log(signal, contract, premium, spot, buy_id)
    print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# EXIT
# ─────────────────────────────────────────────────────────────────────────────

def ha_reversal(df_1h: Optional[pd.DataFrame]) -> bool:
    """True if the last closed 1H HA candle is opposite to current position."""
    if not ACTIVE_POSITION or df_1h is None or len(df_1h) < 2:
        return False
    last_color = compute_ha(df_1h).iloc[-2]["ha_color"]
    if ACTIVE_POSITION["type"] == "CE" and last_color == "red":
        print("   🔄 Exit: 1H HA turned RED")
        return True
    if ACTIVE_POSITION["type"] == "PE" and last_color == "green":
        print("   🔄 Exit: 1H HA turned GREEN")
        return True
    return False


def check_exit(df_1h: Optional[pd.DataFrame] = None):
    global ACTIVE_POSITION, DAILY_PNL, TRADES_TODAY

    if not ACTIVE_POSITION:
        return

    ltp = get_ltp(ACTIVE_POSITION["key"])
    if not ltp:
        return

    entry   = ACTIVE_POSITION["entry"]
    sl      = ACTIVE_POSITION["sl"]
    target  = ACTIVE_POSITION["target"]
    qty     = ACTIVE_POSITION["qty"]
    opt     = ACTIVE_POSITION["type"]
    pnl     = (ltp - entry) * qty
    pnl_pct = (ltp - entry) / entry * 100
    t       = datetime.now().strftime("%H:%M")
    reason  = None

    # Priority: HA reversal → SL → Target → Time
    if ha_reversal(df_1h):
        reason = "HA_REVERSAL"
    elif opt == "CE" and ltp <= sl:
        reason = "SL_HIT"
    elif opt == "PE" and ltp >= sl:
        reason = "SL_HIT"
    elif opt == "CE" and ltp >= target:
        reason = "TARGET_HIT"
    elif opt == "PE" and ltp <= target:
        reason = "TARGET_HIT"
    elif t >= MARKET_CLOSE_TIME:
        reason = "EOD_EXIT"

    if not reason:
        if DEBUG_MODE:
            print(f"   📊 {ACTIVE_POSITION['symbol']} | "
                  f"LTP ₹{ltp:.2f} | SL ₹{sl:.2f} | Target ₹{target:.2f} | "
                  f"P&L ₹{pnl:+.0f} ({pnl_pct:+.1f}%)")
        return

    print(f"\n{'='*60}")
    print(f"🔚 EXIT: {reason} | {ACTIVE_POSITION['symbol']}")
    print(f"   Entry ₹{entry:.2f} → LTP ₹{ltp:.2f} | P&L ₹{pnl:+.0f} ({pnl_pct:+.1f}%)")

    if ENABLE_AUTO_TRADING:
        if ACTIVE_POSITION.get("sl_id"):
            cancel_order(ACTIVE_POSITION["sl_id"])
        eid = place_order(ACTIVE_POSITION["key"], qty, "SELL", "MARKET")
        print(f"   {'✅ Exit: ' + eid if eid else '⚠️  Exit order FAILED — close manually!'}")

    DAILY_PNL    += pnl
    TRADES_TODAY += 1
    _log_exit(ACTIVE_POSITION, ltp, reason, pnl, pnl_pct)
    ACTIVE_POSITION = {}
    print(f"   Daily P&L ₹{DAILY_PNL:+.0f} | Trades {TRADES_TODAY}/{MAX_TRADES_PER_DAY}")
    print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _log(signal, contract, premium, spot, order_id):
    row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           signal["signal"], f"{signal['ema9']:.1f}", f"{signal['ema21']:.1f}",
           f"{signal['rsi']:.1f}", f"{signal['ha_close']:.1f}",
           premium, contract.get("trading_symbol"),
           contract["strike_price"], contract["expiry"], spot, order_id]
    first = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if first:
            w.writerow(CSV_HEADERS)
        w.writerow(row)
    with open(LOG_FILE, "a") as f:
        f.write(f"\nENTRY {datetime.now()} | {signal['signal']} | "
                f"EMA9={signal['ema9']:.1f} EMA21={signal['ema21']:.1f} "
                f"RSI={signal['rsi']:.1f} | {contract.get('trading_symbol')} "
                f"₹{premium:.2f} | order={order_id}\n")


def _log_exit(pos, exit_px, reason, pnl, pnl_pct):
    with open(LOG_FILE, "a") as f:
        f.write(f"EXIT  {datetime.now()} | {reason} | "
                f"₹{pos['entry']:.2f}→₹{exit_px:.2f} | "
                f"P&L ₹{pnl:+.0f} ({pnl_pct:+.1f}%)\n")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def banner():
    print("\n" + "="*55)
    print("  NIFTY 1H HA + EMA CROSSOVER  (Simplified)")
    print("="*55)
    print(f"  Mode      : {'LIVE' if ENABLE_AUTO_TRADING else 'SIGNAL ONLY'}")
    print(f"  EMA       : {EMA_FAST}/{EMA_SLOW} cross + {EMA_TREND} trend filter")
    print(f"  RSI({RSI_PERIOD})   : CE >{RSI_LONG}  |  PE <{RSI_SHORT}")
    print(f"  SL        : {STOPLOSS_PCT}%  |  Target : {TARGET_MULTIPLIER}× SL")
    print(f"  Exit      : HA reversal / SL / Target / EOD {MARKET_CLOSE_TIME}")
    print(f"  Max trades: {MAX_TRADES_PER_DAY}/day  |  Loss cap: ₹{MAX_DAILY_LOSS_ABS:,.0f}")
    print(f"  Entry     : {EMA_TREND}-bar warmup then {NO_NEW_ENTRY_AFTER} cutoff")
    print("="*55 + "\n")


def main():
    global ACTIVE_POSITION, DAILY_PNL, TRADES_TODAY
    global LAST_SIGNAL_CANDLE, OPTION_CHAIN_CACHE, LAST_CHAIN_FETCH

    banner()
    scan = 0

    while True:
        try:
            now = datetime.now()
            t   = now.strftime("%H:%M")

            # Outside market hours
            if now.weekday() >= 5 or t < "09:15" or t >= MARKET_CLOSE_TIME:
                if t >= MARKET_CLOSE_TIME and t < "16:00":
                    print(f"📊 Session done | P&L ₹{DAILY_PNL:+.0f} | Trades {TRADES_TODAY}")
                    ACTIVE_POSITION = {}; DAILY_PNL = 0.0; TRADES_TODAY = 0
                    LAST_SIGNAL_CANDLE = None
                    OPTION_CHAIN_CACHE.clear(); LAST_CHAIN_FETCH = None
                    time.sleep(300)
                    continue
                time.sleep(30)
                continue

            # Wait for indicator warmup
            if t < "10:30":
                print(f"⏳ {t} — waiting for 10:30 (indicator warmup)")
                time.sleep(30)
                continue

            scan += 1
            df = get_1h_candles()
            n  = len(df) if df is not None else 0

            print(f"🔍 #{scan} {now.strftime('%H:%M:%S')} | "
                  f"bars={n} | trades={TRADES_TODAY}/{MAX_TRADES_PER_DAY} | "
                  f"P&L=₹{DAILY_PNL:+.0f} | pos={'YES' if ACTIVE_POSITION else 'no'}",
                  flush=True)

            # Exit check (always)
            if ACTIVE_POSITION:
                check_exit(df)

            # Entry scan (no position + within window)
            elif t <= NO_NEW_ENTRY_AFTER:
                sig = scan_for_signal(df)
                if sig:
                    execute_entry(sig)

            time.sleep(SCAN_INTERVAL_SECS)

        except KeyboardInterrupt:
            print(f"\n⛔ Stopped | P&L ₹{DAILY_PNL:+.0f}")
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
