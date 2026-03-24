import os
import json
import time
import logging
import threading
import requests
import yfinance as yf
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TELEGRAM_TOKEN    = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID  = os.environ["TELEGRAM_CHAT_ID"]
INITIAL_CAPITAL   = float(os.environ.get("CAPITAL", "200"))
TICKERS = [t.strip() for t in os.environ.get("TICKERS", "NVDA,TSLA,AMD,PLTR,COIN").split(",") if t.strip()]
SCAN_INTERVAL_MIN = int(os.environ.get("SCAN_INTERVAL_MIN", "30"))
MAX_POSITION_PCT  = 0.35
MIN_CONFIDENCE    = 60
MAX_POSITIONS     = 4

MARKET_TZ    = ZoneInfo("America/New_York")
MARKET_OPEN  = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)

portfolio = {
    "cash": INITIAL_CAPITAL, "positions": {}, "history": [],
    "peak_value": INITIAL_CAPITAL, "pending": {},
}
_last_update_id = 0


# ── Helpers ───────────────────────────────────────────────────────────

def portfolio_value() -> float:
    total = portfolio["cash"]
    for pos in portfolio["positions"].values():
        total += pos["shares"] * pos.get("current_price", pos["entry_price"])
    return round(total, 2)

def is_market_hours() -> bool:
    now = datetime.now(MARKET_TZ)
    return now.weekday() < 5 and MARKET_OPEN <= now.time() <= MARKET_CLOSE

def ts() -> str:
    return datetime.now(MARKET_TZ).strftime("%d/%m %H:%M")


# ── Real-time market data via yfinance (no token cost) ────────────────

def fetch_market_data(ticker: str) -> dict:
    """Fetch real-time price + technicals using yfinance."""
    tk   = yf.Ticker(ticker)
    info = tk.fast_info
    hist = tk.history(period="3mo", interval="1d")

    price   = round(float(info.last_price), 2)
    prev    = round(float(info.previous_close), 2)
    chg_pct = round((price - prev) / prev * 100, 2)
    vol     = int(info.three_month_average_volume or 0)

    # Moving averages
    ma20 = round(hist["Close"].tail(20).mean(), 2) if len(hist) >= 20 else price
    ma50 = round(hist["Close"].tail(50).mean(), 2) if len(hist) >= 50 else price

    # RSI (14)
    delta  = hist["Close"].diff()
    gain   = delta.clip(lower=0).tail(14).mean()
    loss   = (-delta.clip(upper=0)).tail(14).mean()
    rsi    = round(100 - (100 / (1 + gain / loss)), 1) if loss != 0 else 50.0

    # Support / resistance (52w high/low)
    high52 = round(float(info.year_high), 2)
    low52  = round(float(info.year_low), 2)
    support    = round(low52 * 1.05, 2)
    resistance = round(high52 * 0.95, 2)

    return {
        "price": price, "prev_close": prev, "chg_pct": chg_pct,
        "ma20": ma20, "ma50": ma50, "rsi": rsi,
        "high52": high52, "low52": low52,
        "support": support, "resistance": resistance,
        "avg_volume": vol,
    }


# ── Claude analysis (no web search — prices already fetched) ─────────

def analyze_ticker(ticker: str, mkt: dict) -> dict:
    pv       = portfolio_value()
    open_pos = list(portfolio["positions"].keys())

    trend = "bullish" if mkt["price"] > mkt["ma20"] > mkt["ma50"] else \
            "bearish" if mkt["price"] < mkt["ma20"] < mkt["ma50"] else "neutral"
    macd_signal = "bullish" if mkt["ma20"] > mkt["ma50"] else \
                  "bearish" if mkt["ma20"] < mkt["ma50"] else "neutral"

    system_prompt = f"""You are a professional-grade portfolio strategist with full discretionary control.

MANDATE: Generate maximum return on this portfolio.
Portfolio value: ${pv:.2f} | Cash: ${portfolio['cash']:.2f} | Open: {open_pos or 'none'}
Max per position: {MAX_POSITION_PCT*100:.0f}% | Profile: AGGRESSIVE.

REAL-TIME MARKET DATA for {ticker} (verified, do NOT override):
- Current price: ${mkt['price']} (prev close: ${mkt['prev_close']}, {mkt['chg_pct']:+.2f}% today)
- MA20: ${mkt['ma20']} | MA50: ${mkt['ma50']} | Trend: {trend}
- RSI(14): {mkt['rsi']} | MACD signal: {macd_signal}
- 52w range: ${mkt['low52']} – ${mkt['high52']}
- Support: ${mkt['support']} | Resistance: ${mkt['resistance']}

Using this real data plus your knowledge of {ticker}'s fundamentals, catalysts, sector, and momentum:
1. Generate a BUY / SELL / HOLD signal
2. Set a structure-based stop loss and take profit
3. Provide a clear, verifiable thesis

Return ONLY raw JSON — no markdown:
{{
  "ticker": string,
  "companyName": string,
  "signal": "BUY" | "SELL" | "HOLD",
  "confidence": number 0-100,
  "currentPrice": {mkt['price']},
  "priceChangePct": {mkt['chg_pct']},
  "riskScore": number 1-10,
  "rsi": {mkt['rsi']},
  "macdSignal": "{macd_signal}",
  "trend": "{trend}",
  "support": {mkt['support']},
  "resistance": {mkt['resistance']},
  "stopLoss": number,
  "takeProfit": number,
  "riskReward": string,
  "conviction": "high" | "medium" | "low",
  "catalysts": string,
  "thesis": string,
  "reasoning": string
}}"""

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 800,
        "system": system_prompt,
        "messages": [{"role": "user", "content": f"Analyze {ticker} and generate signal. Date: {datetime.now(MARKET_TZ).strftime('%Y-%m-%d %H:%M')} ET"}]
    }
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json=payload, timeout=60
    )
    if not resp.ok:
        log.error(f"API {resp.status_code}: {resp.text}")
    resp.raise_for_status()
    data = resp.json()
    text = "".join(b["text"] for b in data.get("content", []) if b.get("type") == "text")
    text = text.replace("```json", "").replace("```", "").strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError(f"No JSON for {ticker}")
    result = json.loads(text[s:e+1])
    # Always enforce real price from yfinance
    result["currentPrice"]   = mkt["price"]
    result["priceChangePct"] = mkt["chg_pct"]
    return result


# ── Position management ───────────────────────────────────────────────

def calc_position_size(signal: dict) -> float:
    pv        = portfolio_value()
    max_alloc = pv * MAX_POSITION_PCT
    conf_mult = signal["confidence"] / 100
    risk_mult = (11 - signal["riskScore"]) / 10
    return round(min(max_alloc * conf_mult * risk_mult, portfolio["cash"]), 2)

def register_buy(ticker: str, shares: float, price: float, signal: dict | None = None) -> str:
    cost = round(shares * price, 2)
    if cost > portfolio["cash"] + 0.01:
        return f"❌ Caixa insuficiente (${portfolio['cash']:.2f}) para ${cost:.2f}"
    if ticker in portfolio["positions"]:
        return f"❌ Posição já aberta em {ticker}."
    stop = signal["stopLoss"]   if signal else round(price * 0.92, 2)
    tgt  = signal["takeProfit"] if signal else round(price * 1.20, 2)
    portfolio["cash"] -= cost
    portfolio["positions"][ticker] = {
        "shares": round(shares, 4), "entry_price": price, "current_price": price,
        "stop_loss": stop, "take_profit": tgt, "opened_at": ts(), "allocated": cost,
    }
    portfolio["pending"].pop(ticker, None)
    pv  = portfolio_value()
    pct = cost / pv * 100
    log.info(f"BUY confirmed: {ticker} {shares:.4f} @ ${price:.2f} (${cost:.2f})")
    return (f"✅ *BUY confirmado — {ticker}*\n"
            f"  {shares:.4f} shares @ ${price:.2f}\n"
            f"  Custo: ${cost:.2f} ({pct:.0f}% do portfólio)\n"
            f"  🛑 Stop: ${stop:.2f}  ✅ Target: ${tgt:.2f}\n"
            f"  💵 Caixa restante: ${portfolio['cash']:.2f}")

def register_sell(ticker: str, shares: float, price: float) -> str:
    pos = portfolio["positions"].get(ticker)
    if not pos:
        return f"❌ Nenhuma posição aberta em {ticker}"
    shares_to_sell = min(shares, pos["shares"])
    proceeds   = shares_to_sell * price
    cost_basis = (shares_to_sell / pos["shares"]) * pos["allocated"]
    pnl        = proceeds - cost_basis
    pnl_pct    = pnl / cost_basis * 100
    if shares_to_sell >= pos["shares"] * 0.999:
        portfolio["positions"].pop(ticker)
        portfolio["history"].append({
            "ticker": ticker, "entry": pos["entry_price"], "exit": price,
            "shares": shares_to_sell, "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2), "reason": "confirmação manual", "closed_at": ts(),
        })
    else:
        pos["shares"]    = round(pos["shares"] - shares_to_sell, 4)
        pos["allocated"] = round(pos["allocated"] - cost_basis, 2)
    portfolio["cash"] += proceeds
    portfolio["pending"].pop(ticker, None)
    log.info(f"SELL confirmed: {ticker} {shares_to_sell:.4f} @ ${price:.2f} PnL ${pnl:+.2f}")
    return (f"{'✅' if pnl >= 0 else '❌'} *SELL confirmado — {ticker}*\n"
            f"  {shares_to_sell:.4f} shares @ ${price:.2f}\n"
            f"  PnL: *{pnl:+.2f}* ({pnl_pct:+.1f}%)\n"
            f"  💵 Caixa: ${portfolio['cash']:.2f}")

def close_position_auto(ticker: str, price: float, reason: str) -> dict | None:
    pos = portfolio["positions"].pop(ticker, None)
    if not pos: return None
    proceeds = pos["shares"] * price
    pnl      = proceeds - pos["allocated"]
    portfolio["cash"] += proceeds
    trade = {"ticker": ticker, "entry": pos["entry_price"], "exit": price,
             "shares": pos["shares"], "pnl": round(pnl, 2),
             "pnl_pct": round(pnl / pos["allocated"] * 100, 2),
             "reason": reason, "closed_at": ts()}
    portfolio["history"].append(trade)
    log.info(f"AUTO-CLOSED {ticker}: PnL ${pnl:+.2f} — {reason}")
    return trade

def check_stops(prices: dict) -> list:
    closed = []
    for ticker, pos in list(portfolio["positions"].items()):
        price = prices.get(ticker, pos["current_price"])
        pos["current_price"] = price
        if price <= pos["stop_loss"]:
            t = close_position_auto(ticker, price, "🛑 stop loss automático")
            if t: closed.append(t)
        elif price >= pos["take_profit"]:
            t = close_position_auto(ticker, price, "✅ take profit automático")
            if t: closed.append(t)
    return closed


# ── Telegram ─────────────────────────────────────────────────────────

def send_telegram(text: str) -> None:
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=15)
    except Exception as e:
        log.error(f"Telegram: {e}")

def fmt_status() -> str:
    pv  = portfolio_value()
    ret = pv - INITIAL_CAPITAL
    pct = ret / INITIAL_CAPITAL * 100
    pk  = portfolio["peak_value"]
    dd  = (pv - pk) / pk * 100 if pv < pk else 0.0
    lines = ["📊 *Status do Portfólio*", f"⏰ {ts()} ET\n",
             f"💰 Valor: *${pv:.2f}*",
             f"{'📈' if ret >= 0 else '📉'} Retorno: *{ret:+.2f}* ({pct:+.1f}%)",
             f"💵 Caixa: ${portfolio['cash']:.2f} ({portfolio['cash']/pv*100:.0f}%)",
             f"📉 Drawdown: {dd:.1f}%\n"]
    if portfolio["positions"]:
        lines.append("📂 *Posições abertas*")
        for tkr, pos in portfolio["positions"].items():
            cp    = pos["current_price"]
            pnl   = (cp - pos["entry_price"]) * pos["shares"]
            pnl_p = (cp - pos["entry_price"]) / pos["entry_price"] * 100
            pct_p = pos["shares"] * cp / pv * 100
            lines.append(f"{'📈' if pnl >= 0 else '📉'} *{tkr}* {pct_p:.0f}% portfólio\n"
                         f"   {pos['shares']:.4f} shares | ${pos['entry_price']:.2f}→${cp:.2f} | PnL {pnl:+.2f} ({pnl_p:+.1f}%)\n"
                         f"   🛑${pos['stop_loss']:.2f}  ✅${pos['take_profit']:.2f}")
    if portfolio["pending"]:
        lines.append("\n⏳ *Aguardando confirmação*")
        for tkr, sig in portfolio["pending"].items():
            alloc  = calc_position_size(sig)
            shares = alloc / sig["currentPrice"]
            lines.append(f"  {sig['signal']} *{tkr}* @ ~${sig['currentPrice']:.2f}\n"
                         f"  → `{sig['signal']} {shares:.4f} {tkr} @ PRECO`")
    if portfolio["history"]:
        wins = sum(1 for t in portfolio["history"] if t["pnl"] > 0)
        wr   = wins / len(portfolio["history"]) * 100
        total_pnl = sum(t["pnl"] for t in portfolio["history"])
        lines.append(f"\n🏆 Win rate: {wr:.0f}% | PnL fechado: {total_pnl:+.2f}")
    lines.append("\n_`BUY 0.1 NVDA @ 880` · `SELL 0.1 NVDA @ 900` · `CANCEL NVDA`_")
    return "\n".join(lines)

def fmt_scan_report(signals: list, closed_trades: list) -> str:
    pv  = portfolio_value()
    ret = pv - INITIAL_CAPITAL
    pct = ret / INITIAL_CAPITAL * 100
    lines = ["🔔 *Novo scan — Portfólio*",
             f"⏰ {ts()} ET | Valor: *${pv:.2f}* ({ret:+.2f} / {pct:+.1f}%)\n"]
    if closed_trades:
        lines.append("🔒 *Fechamentos automáticos*")
        for t in closed_trades:
            lines.append(f"  {'✅' if t['pnl'] > 0 else '❌'} {t['ticker']} {t['pnl']:+.2f} ({t['pnl_pct']:+.1f}%) — {t['reason']}")
        lines.append("")
    if signals:
        lines.append("🚦 *Sinais — aguardando confirmação*")
        for s in signals:
            sig_e  = {"BUY": "🟢", "SELL": "🔴"}.get(s["signal"], "🟡")
            conv_e = {"high": "🔥", "medium": "⚡", "low": "💧"}.get(s.get("conviction","medium"), "⚡")
            alloc  = calc_position_size(s) if s["signal"] == "BUY" else 0
            shares = round(alloc / s["currentPrice"], 4) if alloc else 0
            pos    = portfolio["positions"].get(s["ticker"])
            lines.append(f"{sig_e} *{s['ticker']}* {s['signal']} {conv_e} | Conf: {s['confidence']}% | Risco: {s['riskScore']}/10")
            lines.append(f"   💵 ${s['currentPrice']:.2f} ({s['priceChangePct']:+.1f}% hoje) | R/R: {s.get('riskReward','—')}")
            if s["signal"] == "BUY":
                lines.append(f"   📊 RSI: {s.get('rsi','—')} | Tendência: {s.get('trend','—')}")
                lines.append(f"   💼 Sugestão: {shares:.4f} shares (${alloc:.2f} = {alloc/pv*100:.0f}% portfólio)")
                lines.append(f"   🛑 Stop: ${s['stopLoss']:.2f}  ✅ Target: ${s['takeProfit']:.2f}")
                lines.append(f"   ➡️ `BUY {shares:.4f} {s['ticker']} @ PRECO_EXECUTADO`")
            elif s["signal"] == "SELL" and pos:
                lines.append(f"   ➡️ `SELL {pos['shares']:.4f} {s['ticker']} @ PRECO_EXECUTADO`")
            if s.get("catalysts"):
                lines.append(f"   📰 {s['catalysts'][:130]}")
            if s.get("thesis"):
                lines.append(f"   🎯 _{s['thesis'][:150]}_")
            lines.append("")
    lines.append(f"💵 Caixa: ${portfolio['cash']:.2f} ({portfolio['cash']/pv*100:.0f}%)")
    lines.append("_`STATUS` para ver portfólio completo_")
    return "\n".join(lines)


# ── Telegram listener ─────────────────────────────────────────────────

def parse_command(text: str) -> dict | None:
    text = text.strip().upper()
    if text in ("STATUS", "PORTFOLIO", "/STATUS", "/PORTFOLIO"):
        return {"cmd": "status"}
    if text.startswith("CANCEL "):
        return {"cmd": "cancel", "ticker": text.split()[1]}
    parts = text.replace("@", "").split()
    if len(parts) >= 4 and parts[0] in ("BUY", "SELL"):
        for a, b, c in [(1,2,3),(2,1,3)]:
            try:
                return {"cmd": parts[0], "shares": float(parts[a]),
                        "ticker": parts[b], "price": float(parts[c])}
            except ValueError:
                continue
    return None

def handle_command(text: str):
    parsed = parse_command(text)
    if not parsed:
        send_telegram("❓ *Comandos:*\n`BUY 0.1 NVDA @ 880`\n`SELL 0.1 NVDA @ 900`\n`STATUS`\n`CANCEL NVDA`")
        return
    cmd = parsed["cmd"]
    if cmd == "status":
        send_telegram(fmt_status())
    elif cmd == "cancel":
        tkr = parsed["ticker"]
        portfolio["pending"].pop(tkr, None)
        send_telegram(f"🚫 Sinal de *{tkr}* cancelado.")
    elif cmd in ("BUY", "SELL"):
        sig = portfolio["pending"].get(parsed["ticker"])
        if cmd == "BUY":
            reply = register_buy(parsed["ticker"], parsed["shares"], parsed["price"], sig)
        else:
            reply = register_sell(parsed["ticker"], parsed["shares"], parsed["price"])
        send_telegram(reply)
        pv = portfolio_value()
        if pv > portfolio["peak_value"]:
            portfolio["peak_value"] = pv

def telegram_listener():
    global _last_update_id
    log.info("Telegram listener started")
    while True:
        try:
            resp = requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
                params={"offset": _last_update_id + 1, "timeout": 30}, timeout=40)
            for update in resp.json().get("result", []):
                _last_update_id = update["update_id"]
                msg     = update.get("message", {})
                chat_id = str(msg.get("chat", {}).get("id", ""))
                text    = msg.get("text", "").strip()
                if text and chat_id == TELEGRAM_CHAT_ID:
                    log.info(f"Received: {text!r}")
                    handle_command(text)
        except Exception as e:
            log.error(f"Listener: {e}")
            time.sleep(10)


# ── Main scan ─────────────────────────────────────────────────────────

def scan_all() -> None:
    log.info(f"Scanning: {TICKERS}")
    signals  = {}
    mkt_data = {}

    for ticker in TICKERS:
        try:
            log.info(f"Fetching {ticker}...")
            mkt = fetch_market_data(ticker)
            mkt_data[ticker] = mkt["price"]
            log.info(f"  {ticker} price=${mkt['price']} rsi={mkt['rsi']} trend={'bull' if mkt['price']>mkt['ma20'] else 'bear'}")

            log.info(f"Analyzing {ticker}...")
            sig = analyze_ticker(ticker, mkt)
            signals[ticker] = sig
            log.info(f"  {ticker}: {sig['signal']} conf={sig['confidence']}% risk={sig['riskScore']}")
            time.sleep(5)   # small delay — Claude calls now use <1k tokens each
        except Exception as e:
            log.error(f"Error {ticker}: {e}")
            time.sleep(5)

    if not signals:
        return

    closed_trades = check_stops(mkt_data)

    actionable = []
    for ticker, sig in signals.items():
        if sig["signal"] == "BUY" and ticker not in portfolio["positions"]:
            if sig["confidence"] >= MIN_CONFIDENCE and len(portfolio["positions"]) < MAX_POSITIONS and portfolio["cash"] >= 5:
                portfolio["pending"][ticker] = sig
                actionable.append(sig)
        elif sig["signal"] == "SELL" and ticker in portfolio["positions"]:
            portfolio["pending"][ticker] = sig
            actionable.append(sig)

    pv = portfolio_value()
    if pv > portfolio["peak_value"]:
        portfolio["peak_value"] = pv

    if not closed_trades and not actionable:
        log.info("No actionable signals — skipping report")
        return

    send_telegram(fmt_scan_report(actionable, closed_trades))
    log.info(f"Report sent. Portfolio: ${pv:.2f}")


# ── Entry point ───────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Stock Signal Agent [Portfolio + Confirmation Mode] started ===")
    log.info(f"Tickers: {TICKERS} | Capital: ${INITIAL_CAPITAL} | Interval: {SCAN_INTERVAL_MIN}min")
    threading.Thread(target=telegram_listener, daemon=True).start()
    send_telegram(
        "🚀 *Stock Signal Agent — Modo Portfólio*\n"
        f"Capital: *${INITIAL_CAPITAL:.2f}* | Tickers: {', '.join(TICKERS)}\n"
        f"Preços em tempo real via yfinance ✅\n"
        f"Máx posição: {MAX_POSITION_PCT*100:.0f}% | Confiança mínima: {MIN_CONFIDENCE}%\n\n"
        "📌 *Comandos:*\n"
        "`BUY 0.062 NVDA @ 878.50`\n"
        "`SELL 0.062 NVDA @ 910.00`\n"
        "`STATUS` · `CANCEL NVDA`"
    )
    while True:
        if is_market_hours():
            scan_all()
        else:
            log.info(f"Market closed ({datetime.now(MARKET_TZ).strftime('%a %H:%M')} ET) — sleeping")
        time.sleep(SCAN_INTERVAL_MIN * 60)

if __name__ == "__main__":
    main()
