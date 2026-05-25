import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI
import httpx
import pandas as pd
import ta

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TIMEFRAME        = "15min"
OUTPUTSIZE       = 150
CHECK_INTERVAL   = 5 * 60        # 5 minutes
DAILY_RESET_HR   = 0             # midnight UTC
CAPITAL_AED      = 500.0
RISK_AED         = 10.0
REWARD_AED       = 15.0          # 1:1.5 R:R

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TWELVEDATA_KEY   = os.getenv("TWELVEDATA_API_KEY", "")

MEXC_API_KEY    = os.getenv("MEXC_API_KEY", "")
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "")
MEXC_LEVERAGE   = 5
MEXC_ORDER_VOL  = 1          # contracts per signal
MEXC_BASE       = "https://contract.mexc.com"

ASSETS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "XAU/USD",
    "BTC/USD", "ETH/USD", "BNB/USD", "SOL/USD", "XRP/USD", "ADA/USD",
]

# Mapping: TwelveData symbol → MEXC Futures contract symbol
CRYPTO_TO_MEXC: dict[str, str] = {
    "BTC/USD": "BTC_USDT",
    "ETH/USD": "ETH_USDT",
    "BNB/USD": "BNB_USDT",
    "SOL/USD": "SOL_USDT",
    "XRP/USD": "XRP_USDT",
    "ADA/USD": "ADA_USDT",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── ASSET PARAMS ─────────────────────────────────────────────────────────────
def asset_params(asset: str) -> tuple[float, float]:
    """Returns (unit, sl_multiplier)"""
    if "XAU" in asset: return 0.1,   30.0
    if "BTC" in asset: return 1.0,  100.0
    if "ETH" in asset: return 1.0,   50.0
    if "BNB" in asset: return 0.1,    5.0
    if "SOL" in asset: return 0.01,   1.0
    if "XRP" in asset: return 0.0001, 0.05
    if "ADA" in asset: return 0.0001, 0.02
    if "JPY" in asset: return 0.01,  20.0
    return 0.0001, 15.0

# ─── TRADE STATE ──────────────────────────────────────────────────────────────
@dataclass
class Trade:
    id: str
    asset: str
    direction: str
    entry: float
    sl: float
    tp: float
    opened_at: datetime
    closed_at: Optional[datetime] = None
    result: Optional[str] = None
    pnl: Optional[float] = None
    close_price: Optional[float] = None

trade_counter   = 0
capital         = CAPITAL_AED
open_trades: list[Trade]   = []
closed_trades: list[Trade] = []
last_signals: dict[str, Optional[str]] = {a: None for a in ASSETS}

# Daily tracking (reset each midnight UTC)
daily_signals: list[dict]  = []
daily_closed: list[Trade]  = []

def next_trade_id() -> str:
    global trade_counter
    trade_counter += 1
    return f"T{trade_counter:03d}"

def open_trade(asset, direction, entry, sl, tp) -> Trade:
    t = Trade(
        id=next_trade_id(), asset=asset, direction=direction,
        entry=entry, sl=sl, tp=tp, opened_at=datetime.now(timezone.utc)
    )
    open_trades.append(t)
    daily_signals.append({"asset": asset, "direction": direction,
                          "time": datetime.now(timezone.utc).strftime("%H:%M")})
    return t

def check_close_trades(asset: str, price: float) -> list[Trade]:
    global capital
    closed, remaining = [], []
    for t in open_trades:
        if t.asset != asset:
            remaining.append(t); continue
        result = None
        if t.direction == "BUY":
            if price >= t.tp: result = "WIN"
            elif price <= t.sl: result = "LOSS"
        else:
            if price <= t.tp: result = "WIN"
            elif price >= t.sl: result = "LOSS"
        if result:
            pnl = REWARD_AED if result == "WIN" else -RISK_AED
            capital += pnl
            t.closed_at = datetime.now(timezone.utc)
            t.result = result; t.pnl = pnl; t.close_price = price
            closed_trades.append(t); closed.append(t); daily_closed.append(t)
        else:
            remaining.append(t)
    open_trades.clear(); open_trades.extend(remaining)
    return closed

def get_stats() -> dict:
    wins   = sum(1 for t in closed_trades if t.result == "WIN")
    losses = sum(1 for t in closed_trades if t.result == "LOSS")
    total  = len(closed_trades)
    pnl    = sum(t.pnl for t in closed_trades if t.pnl is not None)
    rate   = round(wins / total * 100) if total else 0
    return {"wins": wins, "losses": losses, "total": total, "pnl": pnl, "rate": rate}

# ─── TELEGRAM ─────────────────────────────────────────────────────────────────
async def send_msg(client: httpx.AsyncClient, chat_id: str, text: str):
    if not TELEGRAM_TOKEN or not chat_id:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        await client.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=10)
    except Exception as e:
        log.error(f"Telegram send error: {e}")

# ─── INDICATORS ───────────────────────────────────────────────────────────────
def check_fvg(df: pd.DataFrame) -> str:
    try:
        if df["High"].iloc[-3] < df["Low"].iloc[-1]:  return "BULLISH_FVG"
        if df["Low"].iloc[-3]  > df["High"].iloc[-1]: return "BEARISH_FVG"
    except: pass
    return "NONE"

def compute_indicators(df: pd.DataFrame) -> Optional[dict]:
    if len(df) < 95:
        return None
    try:
        rsi    = float(ta.momentum.rsi(df["Close"], window=14).iloc[-1])
        ema    = float(ta.trend.ema_indicator(df["Close"], window=50).iloc[-1])
        macd_o = ta.trend.MACD(df["Close"])
        macd   = float(macd_o.macd().iloc[-1])
        macd_s = float(macd_o.macd_signal().iloc[-1])
        stoch  = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=90)
        sk     = float(stoch.stoch().iloc[-1])
        sd     = float(stoch.stoch_signal().iloc[-1])
        fvg    = check_fvg(df)
        return dict(rsi=rsi, ema=ema, macd=macd, macd_sig=macd_s, sk=sk, sd=sd, fvg=fvg)
    except Exception as e:
        log.warning(f"Indicator error: {e}")
        return None

# ─── MEXC FUTURES ─────────────────────────────────────────────────────────────
def _mexc_sign(body_str: str) -> tuple[str, str]:
    ts  = str(int(time.time() * 1000))
    raw = MEXC_API_KEY + ts + body_str
    sig = hmac.new(MEXC_API_SECRET.encode(), raw.encode(), hashlib.sha256).hexdigest()
    return ts, sig

async def place_mexc_order(client: httpx.AsyncClient, symbol: str, direction: str) -> str:
    if not MEXC_API_KEY or not MEXC_API_SECRET:
        return "⚠️ MEXC API keys not configured"
    side = 1 if direction == "BUY" else 3
    body = {
        "symbol":   symbol,
        "price":    0,
        "vol":      MEXC_ORDER_VOL,
        "leverage": MEXC_LEVERAGE,
        "side":     side,
        "type":     5,
        "openType": 1,
    }
    body_str = json.dumps(body, separators=(",", ":"))
    ts, sig  = _mexc_sign(body_str)
    headers  = {
        "ApiKey":       MEXC_API_KEY,
        "Request-Time": ts,
        "Signature":    sig,
        "Content-Type": "application/json",
    }
    try:
        r    = await client.post(f"{MEXC_BASE}/api/v1/private/order/submit",
                                 content=body_str, headers=headers, timeout=10)
        resp = r.json()
        if resp.get("success"):
            oid = resp.get("data", "?")
            return f"✅ Order placed | ID: `{oid}` | {symbol} {direction} {MEXC_LEVERAGE}x"
        return f"❌ MEXC Error: {resp.get('message', resp)}"
    except Exception as e:
        log.error(f"MEXC order error: {e}")
        return f"❌ MEXC Exception: {e}"

async def mexc_balance(client: httpx.AsyncClient) -> str:
    if not MEXC_API_KEY or not MEXC_API_SECRET:
        return "MEXC API keys not set"
    ts, sig = _mexc_sign("")
    headers = {"ApiKey": MEXC_API_KEY, "Request-Time": ts, "Signature": sig}
    try:
        r    = await client.get(f"{MEXC_BASE}/api/v1/private/account/assets",
                                headers=headers, timeout=10)
        resp = r.json()
        if resp.get("success"):
            assets = resp.get("data", [])
            usdt   = next((a for a in assets if a.get("currency") == "USDT"), None)
            if usdt:
                bal = usdt.get("availableBalance", "?")
                return f"💰 MEXC Balance: `{bal} USDT`"
        return f"Could not fetch balance: {resp.get('message', '')}"
    except Exception as e:
        return f"Error: {e}"

# ─── MARKET DATA ──────────────────────────────────────────────────────────────
async def fetch_candles(client: httpx.AsyncClient) -> dict[str, pd.DataFrame]:
    result = {}
    if not TWELVEDATA_KEY:
        log.warning("TWELVEDATA_API_KEY not set"); return result
    params = {
        "symbol": ",".join(ASSETS),
        "interval": TIMEFRAME,
        "outputsize": OUTPUTSIZE,
        "apikey": TWELVEDATA_KEY,
    }
    try:
        r = await client.get("https://api.twelvedata.com/time_series", params=params, timeout=20)
        data = r.json()
    except Exception as e:
        log.error(f"TwelveData error: {e}"); return result

    for asset in ASSETS:
        series = data.get(asset) if len(ASSETS) > 1 else data
        if not series or "values" not in series: continue
        df = pd.DataFrame(series["values"]).iloc[::-1].reset_index(drop=True)
        df[["Close", "High", "Low"]] = df[["close", "high", "low"]].astype(float)
        result[asset] = df
    return result

# ─── SIGNAL CYCLE ─────────────────────────────────────────────────────────────
async def run_cycle(client: httpx.AsyncClient):
    log.info("Checking markets...")
    candles_map = await fetch_candles(client)

    for asset in ASSETS:
        df = candles_map.get(asset)
        if df is None or df.empty: continue
        price = float(df["Close"].iloc[-1])

        for trade in check_close_trades(asset, price):
            icon = "✅" if trade.result == "WIN" else "❌"
            pnl  = trade.pnl or 0
            msg = (
                f"{icon} *Trade Closed: {trade.asset}* [{trade.id}]\n"
                f"Direction: {trade.direction} | Result: *{trade.result}*\n"
                f"Close Price: `{trade.close_price:.5f}`\n"
                f"P&L: `{'+' if pnl >= 0 else ''}{pnl:.2f} AED`"
            )
            await send_msg(client, TELEGRAM_CHAT_ID, msg)

        ind = compute_indicators(df)
        if not ind: continue

        unit, sl_m = asset_params(asset)
        bullish = ind["rsi"] > 35 and price > ind["ema"] and ind["macd"] > ind["macd_sig"] and ind["sk"] > ind["sd"]
        bearish = ind["rsi"] < 75 and price < ind["ema"] and ind["macd"] < ind["macd_sig"] and ind["sk"] < ind["sd"]

        if bullish and last_signals[asset] != "BUY":
            sl, tp  = price - sl_m * unit, price + sl_m * 1.5 * unit
            trade   = open_trade(asset, "BUY", price, sl, tp)
            fvg_tag = "✅ *FVG Detected (Wait for Re-test!)*" if ind["fvg"] == "BULLISH_FVG" else "⚠️ *No FVG found*"
            mexc_line = ""
            if asset in CRYPTO_TO_MEXC:
                result = await place_mexc_order(client, CRYPTO_TO_MEXC[asset], "BUY")
                mexc_line = f"\n\n🏦 *MEXC:* {result}"
            msg = (
                f"🟢 *BUY: {asset}* [{trade.id}]\n"
                f"Entry: `{price:.5f}`\nSL: `{sl:.5f}`\nTP: `{tp:.5f}`\n\n"
                f"{fvg_tag}\n💰 *Risk:* 10 AED → *Reward:* 15 AED{mexc_line}"
            )
            await send_msg(client, TELEGRAM_CHAT_ID, msg)
            last_signals[asset] = "BUY"
            log.info(f"BUY {asset} [{trade.id}] @ {price:.5f}")

        elif bearish and last_signals[asset] != "SELL":
            sl, tp  = price + sl_m * unit, price - sl_m * 1.5 * unit
            trade   = open_trade(asset, "SELL", price, sl, tp)
            fvg_tag = "✅ *FVG Detected (Wait for Re-test!)*" if ind["fvg"] == "BEARISH_FVG" else "⚠️ *No FVG found*"
            mexc_line = ""
            if asset in CRYPTO_TO_MEXC:
                result = await place_mexc_order(client, CRYPTO_TO_MEXC[asset], "SELL")
                mexc_line = f"\n\n🏦 *MEXC:* {result}"
            msg = (
                f"🔴 *SELL: {asset}* [{trade.id}]\n"
                f"Entry: `{price:.5f}`\nSL: `{sl:.5f}`\nTP: `{tp:.5f}`\n\n"
                f"{fvg_tag}\n💰 *Risk:* 10 AED → *Reward:* 15 AED{mexc_line}"
            )
            await send_msg(client, TELEGRAM_CHAT_ID, msg)
            last_signals[asset] = "SELL"
            log.info(f"SELL {asset} [{trade.id}] @ {price:.5f}")

# ─── TELEGRAM COMMANDS ────────────────────────────────────────────────────────
def fmt_daily_summary(label: str = "📅 Daily Summary") -> str:
    stats    = get_stats()
    now      = datetime.now(timezone.utc)
    date_str = now.strftime("%d %b %Y")

    d_wins   = sum(1 for t in daily_closed if t.result == "WIN")
    d_losses = sum(1 for t in daily_closed if t.result == "LOSS")
    d_pnl    = sum(t.pnl for t in daily_closed if t.pnl is not None)
    d_total  = len(daily_closed)
    d_rate   = round(d_wins / d_total * 100) if d_total else 0

    buy_sigs  = [sig for sig in daily_signals if sig["direction"] == "BUY"]
    sell_sigs = [sig for sig in daily_signals if sig["direction"] == "SELL"]

    lines = [
        f"{'━' * 22}",
        f"{label} — {date_str}",
        f"{'━' * 22}",
        f"",
        f"📡 *Signals Fired Today: {len(daily_signals)}*",
        f"  🟢 BUY:  {len(buy_sigs)}    🔴 SELL: {len(sell_sigs)}",
        f"",
        f"📊 *Today's Closed Trades: {d_total}*",
        f"  ✅ Wins: {d_wins}   ❌ Losses: {d_losses}",
        f"  Win Rate: `{d_rate}%`",
        f"  P&L Today: `{'+' if d_pnl >= 0 else ''}{d_pnl:.2f} AED`",
        f"",
        f"💼 *Overall Capital: `{capital:.2f} AED`*",
        f"  Started: 500.00 AED",
        f"  Total P&L: `{'+' if stats['pnl'] >= 0 else ''}{stats['pnl']:.2f} AED`",
        f"  All-time Win Rate: `{stats['rate']}%` ({stats['wins']}W / {stats['losses']}L)",
        f"",
        f"🕐 _Next check in 5 minutes_",
    ]

    if daily_signals:
        lines += ["", "📋 *Today's Signals:*"]
        for sig in daily_signals[-8:]:
            icon = "🟢" if sig["direction"] == "BUY" else "🔴"
            lines.append(f"  {icon} {sig['direction']} {sig['asset']} @ {sig['time']} UTC")

    return "\n".join(lines)

def fmt_start() -> str:
    return (
        "🤖 *MM-Daily Signal Bot*\n\n"
        "SMC Signals: EUR/USD · GBP/USD · USD/JPY · AUD/USD · USD/CHF · XAU/USD\n"
        "MEXC Auto-Trade: BTC · ETH · BNB · SOL · XRP · ADA\n\n"
        "📐 *Strategy:*\n"
        "• Timeframe: 15 minutes\n"
        "• Indicators: EMA(50), RSI(14), MACD, Stoch(90), FVG\n"
        "• Capital: 500 AED | Risk: 10 AED | R:R 1:1.5\n\n"
        "📋 *Commands:*\n"
        "/start — this message\n"
        "/trades — open positions\n"
        "/results — win/loss & P&L\n"
        "/summary — today's full report\n"
        "/mexc — MEXC balance & auto-trade status\n\n"
        "_Signals checked every 5 minutes_"
    )

def fmt_trades() -> str:
    if not open_trades:
        return f"📭 *No open trades right now*\n\n💼 Capital: `{capital:.2f} AED`"
    lines = [f"📂 *Open Trades ({len(open_trades)})*", f"💼 Capital: `{capital:.2f} AED`", ""]
    for t in open_trades:
        age = int((datetime.now(timezone.utc) - t.opened_at).total_seconds() / 60)
        icon = "🟢 BUY" if t.direction == "BUY" else "🔴 SELL"
        lines += [
            f"{icon} *{t.asset}* [{t.id}]",
            f"Entry: `{t.entry:.5f}` | SL: `{t.sl:.5f}` | TP: `{t.tp:.5f}`",
            f"Risk: {RISK_AED} AED → Reward: {REWARD_AED} AED | Open: {age}m ago",
            "",
        ]
    return "\n".join(lines)

def fmt_results() -> str:
    s = get_stats()
    recent = list(reversed(closed_trades[-10:]))
    lines = [
        "📊 *Trade Results*", "",
        f"💼 Capital: `{capital:.2f} AED` (started 500 AED)",
        f"📈 P&L: `{'+' if s['pnl'] >= 0 else ''}{s['pnl']:.2f} AED`",
        f"🏆 Win Rate: `{s['rate']}%` ({s['wins']}W / {s['losses']}L / {s['total']} total)",
        "", f"🕓 *Last {len(recent)} Closed Trades:*",
    ]
    if not recent:
        lines.append("_No closed trades yet_")
    else:
        for t in recent:
            icon = "✅" if t.result == "WIN" else "❌"
            pnl  = t.pnl or 0
            lines.append(f"{icon} {t.direction} *{t.asset}* [{t.id}] → {'+' if pnl >= 0 else ''}{pnl:.2f} AED")
    return "\n".join(lines)

# ─── COMMAND POLLING ──────────────────────────────────────────────────────────
last_update_id = 0

async def poll_commands(client: httpx.AsyncClient):
    global last_update_id
    if not TELEGRAM_TOKEN: return
    try:
        r = await client.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
            params={"offset": last_update_id + 1, "timeout": 20, "limit": 50},
            timeout=25,
        )
        updates = r.json().get("result", [])
        for u in updates:
            last_update_id = u["update_id"]
            msg  = u.get("message", {})
            text = msg.get("text", "").split("@")[0].strip()
            cid  = msg.get("chat", {}).get("id")
            if not cid: continue
            if text == "/start":     await send_msg(client, cid, fmt_start())
            elif text == "/trades":  await send_msg(client, cid, fmt_trades())
            elif text == "/results": await send_msg(client, cid, fmt_results())
            elif text == "/summary": await send_msg(client, cid, fmt_daily_summary("📋 Summary On Demand"))
            elif text == "/mexc":
                bal = await mexc_balance(client)
                pairs = "\n".join(f"  • {k} → {v}" for k, v in CRYPTO_TO_MEXC.items())
                await send_msg(client, cid,
                    f"🏦 *MEXC Futures Status*\n\n"
                    f"{bal}\n\n"
                    f"⚙️ Leverage: `{MEXC_LEVERAGE}x` | Isolated\n"
                    f"📦 Order size: `{MEXC_ORDER_VOL} contract` per signal\n\n"
                    f"📡 *Auto-trading pairs:*\n{pairs}\n\n"
                    f"_Orders fire automatically on each signal_"
                )
    except Exception as e:
        log.warning(f"Poll error: {e}")

# ─── BACKGROUND TASKS ─────────────────────────────────────────────────────────
async def signal_loop(client: httpx.AsyncClient):
    errors = 0
    while True:
        try:
            await run_cycle(client)
            errors = 0
        except Exception as e:
            errors += 1
            wait = min(errors * 30, 600)
            log.error(f"Cycle error ({errors}): {e} — waiting {wait}s")
            await asyncio.sleep(wait)
        await asyncio.sleep(CHECK_INTERVAL)

async def command_loop(client: httpx.AsyncClient):
    while True:
        await poll_commands(client)

async def daily_summary_loop(client: httpx.AsyncClient):
    """Sends a daily summary at 9 PM GMT+4 (17:00 UTC) every day."""
    while True:
        now     = datetime.now(timezone.utc)
        target  = now.replace(hour=17, minute=0, second=0, microsecond=0)
        if now >= target:
            target = target.replace(day=target.day + 1)
        secs = (target - now).total_seconds()
        log.info(f"Daily summary scheduled in {int(secs/60)} minutes")
        await asyncio.sleep(secs)
        await send_msg(client, TELEGRAM_CHAT_ID, fmt_daily_summary())
        log.info("Daily summary sent")

async def daily_reset_loop():
    """Resets signal state and daily counters every day at midnight UTC."""
    while True:
        now  = datetime.now(timezone.utc)
        secs = ((24 - now.hour - 1) * 3600) + ((60 - now.minute - 1) * 60) + (60 - now.second)
        await asyncio.sleep(secs + 60)
        for asset in ASSETS:
            last_signals[asset] = None
        daily_signals.clear()
        daily_closed.clear()
        log.info("Daily signal state and counters reset")

async def keepalive_loop():
    """Pings the health endpoint every 4 minutes to stay awake."""
    await asyncio.sleep(60)
    port = int(os.getenv("PORT", "8000"))
    async with httpx.AsyncClient() as client:
        while True:
            try:
                await client.get(f"http://localhost:{port}/health", timeout=5)
            except Exception:
                pass
            await asyncio.sleep(4 * 60)

# ─── APP ──────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient() as client:
        await send_msg(client, TELEGRAM_CHAT_ID,
            "🤖 *MM-Daily Signal Bot (SMC + MEXC)*\n"
            "Capital: 500 AED | Risk: 10 AED/Trade\n\n"
            "Forex: EUR/USD · GBP/USD · USD/JPY · AUD/USD · USD/CHF · XAU/USD\n"
            "MEXC Futures: BTC · ETH · BNB · SOL · XRP · ADA\n\n"
            "Use /start /trades /results /summary /mexc"
        )
        tasks = [
            asyncio.create_task(signal_loop(client)),
            asyncio.create_task(command_loop(client)),
            asyncio.create_task(daily_summary_loop(client)),
            asyncio.create_task(daily_reset_loop()),
            asyncio.create_task(keepalive_loop()),
        ]
        yield
        for t in tasks:
            t.cancel()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    s = get_stats()
    return {
        "status": "ok",
        "capital_aed": round(capital, 2),
        "open_trades": len(open_trades),
        "closed_trades": s["total"],
        "win_rate_pct": s["rate"],
        "pnl_aed": round(s["pnl"], 2),
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)