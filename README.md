# Nifty 50 Options Trader — Heikin-Ashi Strategy

An intraday options trading bot for **Nifty 50** built on the **Upstox API v2**, using Heikin-Ashi candles across multiple timeframes to identify high-probability reversal entries.

> **Disclaimer:** This is an algorithmic trading tool. Markets carry risk. Use in paper/signal-only mode before going live. The authors take no responsibility for financial losses.

---

## Strategy Overview

1. **No trades before 10:30 AM** — waits for the second 15-minute bar to close.
2. **Initial bias at 10:30** — compares HA-close of the `10:00` and `10:15` 15m bars. Bullish → CE only. Bearish → PE only. Weak delta → skip the day.
3. **Hourly bias refresh** — from 11:15 onwards, the closing 1H HA candle colour updates the bias.
4. **Entry pattern (15m chart)**
   - CE: `RED → GREEN` HA candle, close above prev HA high, small lower shadow
   - PE: `GREEN → RED` HA candle, close below prev HA low, small upper shadow
5. **Exit** — trailing HA stop (ratchet), fixed target (2× risk), or 15:15 time exit.

---

## Filters & Risk Controls (v3)

| Filter | Description |
|--------|-------------|
| **Bias strength grading** | STRONG / NORMAL / WEAK based on HA-close delta at 10:30. WEAK days are skipped entirely. |
| **1H trend alignment** | Last 1H HA close must be moving in bias direction (±5 pt tolerance for minor wiggles). |
| **Volatility / range filter** | Entry blocked if last 15m candle range < 40% of 10-bar average range. |
| **Adaptive structure filter** | STRONG bias → relaxed structure always. NORMAL bias → relaxed inside 11:00–14:00, strict outside. |
| **Dynamic SL modification** | Exchange SL order is cancelled and replaced each time the HA trail ratchets. |
| **Daily loss circuit-breaker** | No new entries if `DAILY_PNL ≤ -₹5,000`. |
| **Max trades per day** | Capped at 3 trades to prevent overtrading on choppy days. |
| **Option chain caching** | Chain endpoint cached for 5 minutes to reduce API calls. |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/frzkhn155-ai/nifty-ha-options-trader.git
cd nifty-ha-options-trader
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Upstox token

```bash
cp .env.example .env
# Edit .env and set UPSTOX_TOKEN=your_token_here
```

Then export it before running:

```bash
export UPSTOX_TOKEN="your_token_here"
```

Or on Windows:

```cmd
set UPSTOX_TOKEN=your_token_here
```

### 4. Run in signal-only mode first

In `nifty_ha_options_v3.py`, set:

```python
ENABLE_AUTO_TRADING = False
```

Then run:

```bash
python nifty_ha_options_v3.py
```

Watch the output through a full session. When you're satisfied with signal quality, flip `ENABLE_AUTO_TRADING = True`.

---

## Key Configuration

```python
# Risk
STOPLOSS_PCT       = 15.0      # % of option premium
TARGET_MULTIPLIER  = 2.0       # target = risk × multiplier
MAX_DAILY_LOSS_ABS = 5000.0    # ₹ — stop trading for the day
MAX_TRADES_PER_DAY = 3

# 1H trend filter
ENABLE_1H_TREND_FILTER  = True
TREND_ALIGN_TOLERANCE   = 5.0  # Nifty points — set 0.0 for strict

# Timing
NO_NEW_ENTRY_AFTER = "15:15"
ENTRY_WINDOW_START = "11:00"   # relaxed structure inside this window
ENTRY_WINDOW_END   = "14:00"

# Bias grading
BIAS_STRONG_MIN_DELTA = 5.0    # pts
BIAS_WEAK_MAX_DELTA   = 1.5    # pts — skip if below this

# Volatility filter
MIN_RANGE_RATIO     = 0.40     # 40% of average range
RANGE_LOOKBACK_BARS = 10

# Option chain cache
OPTION_CHAIN_CACHE_TTL = 300   # seconds
```

---

## Output Files

| File | Contents |
|------|----------|
| `nifty_ha_strategy.txt` | Human-readable trade log (entries, exits, P&L) |
| `nifty_ha_trades.csv` | Machine-readable trade record for analysis |

Both are excluded from git via `.gitignore`.

---

## Version History

| Version | Changes |
|---------|---------|
| v1 | Initial HA strategy, 15m reversal detection, fixed SL |
| v2 | 1H trend filter, dynamic SL modification, bias strength grading, volatility filter, preferred entry window, daily limits, adaptive structure filter |
| v3 | Tolerant 1H alignment (±5 pts), option chain caching, `Optional[...]` type hints (Python 3.8+), `MAX_DAILY_LOSS_ABS` clarity, STRONG bias ignores time window for structure check, hard-fail on missing token |

---

## Requirements

- Python 3.8+
- Upstox trading account with API access enabled
- Active market session (NSE trading hours)
