import os
import json
import time
import logging
import threading
import requests
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TELEGRAM_TOKEN    = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID  = os.environ["TELEGRAM_CHAT_ID"]
INITIAL_CAPITAL   = float(os.environ.get("CAPITAL", "200"))

TICKERS = [t.strip() for t in os.environ.get(
    "TICKERS", "NVDA,TSLA,AMD,PLTR,COIN"
).split(",") if t.strip()]

SCAN_INTERVAL_MIN = int(os.environ.get("SCAN_INTERVAL_MIN", "30"))
MAX_POSITION_PCT  = 0.35
MIN_CONFIDENCE    = 60
MAX_POSITIONS     = 4

MARKET_TZ    = ZoneInfo("America/New_York")
MARKET_OPEN  = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)

# ── Portfolio state ───────────────────────────────────────────────────
portfolio = {
    "cash":       INITIAL_CAPITAL,
    "positions":  {},   # { ticker: {shares, entry_price, current_price, stop_loss, take_profit, ...} }
    "history":    [],
    "peak_value": INITIAL_CAPITAL,
    "pending":    {},   # { ticker: suggested signal } — waiting for user confirmation
}

# ── Telegram polling state ────────────────────────────────────────────
_last_update_id = 0


# ── Helpers ───────────────────────────────────────────────────────────

def portfolio_value() -> float:
    total = portfolio["cash"]
    for pos in portfolio["positions"].values():
        total += pos["shares"] * pos.get("current_price", pos["entry_price"])
    return round(total, 2)


def is_market_hours() -> bool:
    now = datetime.now(MARKET_TZ)
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


def ts() -> str:
    return datetime.now(MARKET_TZ).strftime("%d/%m %H:%M")


# ── Anthropic API ─────────────────────────────────────────────────────

def analyze_ticker(ticker: str) -> dict:
    pv       = portfolio_value()
    open_pos = list(portfolio["positions"].keys())

    system_prompt = f"""You are a professional-grade portfolio strategist with full discretionary control over a real U.S. stock portfolio.

MANDATE: Generate maximum return on this portfolio.
Portfolio value: ${pv:.2f} | Cash available: ${portfolio['cash']:.2f} | Open positions: {open_pos if open_pos else 'none'}
Max allocation per position: {MAX_POSITION_PCT*100:.0f}% of portfolio value.

You have full control over:
- Position sizing: concentrate or diversify based on conviction
- Risk management: stop-loss placement must be structure-based, not arbitrary
- Strategy: short-term catalysts, momentum plays, or long-term thesis — your call
- Priority: highest-conviction ideas get the most capital

Your analysis must be deep and cover ALL of the following:
TECHNICAL — RSI, MACD, moving averages, volume, support/resistance levels, trend structure
FUNDAMENTAL — earnings growth, revenue trajectory, margins, valuation vs sector peers
CATALYST — upcoming earnings dates, product launches, regulatory events, macro tailwinds/headwinds
MOMENTUM — institutional flow signals, short interest, relative strength vs market

Rules:
- Only recommend BUY when you have strong conviction the position will be POSITIVE for the account
- Size positions proportionally to conviction level
- Every trade must have a clear, verifiable thesis in the reasoning field
- SELL when the original thesis is broken or target is reached
- HOLD only when waiting for a better entry or thesis is intact but not yet actionable

Return ONLY raw JSON — no markdown:
{{
  "ticker": string,
  "companyName": string,
  "signal": "BUY" | "SELL" | "HOLD",
  "confidence": number 0-100,
  "currentPrice": number,
  "priceChangePct": number,
  "riskScore": number 1-10,
  "rsi": number,
  "macdSignal": "bullish" | "bearish" | "neutral",
  "trend": "bullish" | "bearish" | "neutral",
  "support": number,
  "resistance": number,
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
        "max_tokens": 1500,
        "system": system_prompt,
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "messages": [{"role": "user", "content": f"Search for the CURRENT live stock price of {ticker} right now on March 24 2026, then analyze it as a portfolio strategist. You MUST use web search to get the real price before generating any signal."}]
    }
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "anthropic-beta": "web-search-2025-03-05",
        },
        json=payload, timeout=120
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
    return json.loads(text[s:e+1])


# ── Position management ───────────────────────────────────────────────

def calc_position_size(signal: dict) -> float:
    pv        = portfolio_value()
    max_alloc = pv * MAX_POSITION_PCT
    conf_mult = signal["confidence"] / 100
    risk_mult = (11 - signal["riskScore"]) / 10
    alloc     = max_alloc * conf_mult * risk_mult
    return round(min(alloc, portfolio["cash"]), 2)


def register_buy(ticker: str, shares: float, price: float, signal: dict | None = None) -> str:
    """Register a confirmed BUY at actual executed price."""
    cost = shares * price
    if cost > portfolio["cash"] + 0.01:
        return f"❌ Caixa insuficiente (${portfolio['cash']:.2f}) para ${cost:.2f}"
    if ticker in portfolio["positions"]:
        return f"❌ Já existe posição aberta em {ticker}. Use SELL para fechar primeiro."

    stop  = signal["stopLoss"]   if signal else round(price * 0.92, 2)
    tgt   = signal["takeProfit"] if signal else round(price * 1.20, 2)

    portfolio["cash"] -= cost
    portfolio["positions"][ticker] = {
        "shares":        round(shares, 4),
        "entry_price":   price,
        "current_price": price,
        "stop_loss":     stop,
        "take_profit":   tgt,
        "opened_at":     ts(),
        "allocated":     round(cost, 2),
    }
    portfolio["pending"].pop(ticker, None)
    pv  = portfolio_value()
    pct = cost / pv * 100
    log.info(f"BUY confirmed: {ticker} {shares:.4f} @ ${price:.2f} (${cost:.2f})")
    return (
        f"✅ *BUY confirmado — {ticker}*\n"
        f"  {shares:.4f} shares @ ${price:.2f}\n"
        f"  Custo: ${cost:.2f} ({pct:.0f}% do portfólio)\n"
        f"  🛑 Stop: ${stop:.2f}  ✅ Target: ${tgt:.2f}\n"
        f"  💵 Caixa restante: ${portfolio['cash']:.2f}"
    )


def register_sell(ticker: str, shares: float, price: float) -> str:
    """Register a confirmed SELL at actual executed price."""
    pos = portfolio["positions"].get(ticker)
    if not pos:
        return f"❌ Nenhuma posição aberta em {ticker}"

    shares_to_sell = min(shares, pos["shares"])
    proceeds = shares_to_sell * price
    cost_basis = (shares_to_sell / pos["shares"]) * pos["allocated"]
    pnl  = proceeds - cost_basis
    pnl_pct = pnl / cost_basis * 100

    if shares_to_sell >= pos["shares"] * 0.999:
        portfolio["positions"].pop(ticker)
        portfolio["history"].append({
            "ticker": ticker, "entry": pos["entry_price"], "exit": price,
            "shares": shares_to_sell, "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2), "reason": "confirmação manual",
            "closed_at": ts(),
        })
    else:
        pos["shares"]    = round(pos["shares"] - shares_to_sell, 4)
        pos["allocated"] = round(pos["allocated"] - cost_basis, 2)

    portfolio["cash"] += proceeds
    portfolio["pending"].pop(ticker, None)
    log.info(f"SELL confirmed: {ticker} {shares_to_sell:.4f} @ ${price:.2f} PnL ${pnl:+.2f}")
    return (
        f"{'✅' if pnl >= 0 else '❌'} *SELL confirmado — {ticker}*\n"
        f"  {shares_to_sell:.4f} shares @ ${price:.2f}\n"
        f"  PnL: *{pnl:+.2f}* ({pnl_pct:+.1f}%)\n"
        f"  💵 Caixa: ${portfolio['cash']:.2f}"
    )


def close_position_auto(ticker: str, price: float, reason: str) -> dict | None:
    pos = portfolio["positions"].pop(ticker, None)
    if not pos: return None
    proceeds = pos["shares"] * price
    pnl      = proceeds - pos["allocated"]
    portfolio["cash"] += proceeds
    trade = {
        "ticker": ticker, "entry": pos["entry_price"], "exit": price,
        "shares": pos["shares"], "pnl": round(pnl, 2),
        "pnl_pct": round(pnl / pos["allocated"] * 100, 2),
        "reason": reason, "closed_at": ts(),
    }
    portfolio["history"].append(trade)
    log.info(f"AUTO-CLOSED {ticker}: PnL ${pnl:+.2f} — {reason}")
    return trade


def check_stops(signals: dict) -> list:
    closed = []
    for ticker, pos in list(portfolio["positions"].items()):
        sig   = signals.get(ticker)
        price = sig["currentPrice"] if sig else pos["current_price"]
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
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=15)
    except Exception as e:
        log.error(f"Telegram send error: {e}")


def parse_trade_command(text: str) -> dict | None:
    """
    Parse commands:
      BUY 0.062 NVDA @ 876.50
      SELL 0.062 TSLA @ 380
      STATUS
      PORTFOLIO
      CANCEL NVDA
    """
    text = text.strip().upper()

    if text in ("STATUS", "PORTFOLIO", "/STATUS", "/PORTFOLIO"):
        return {"cmd": "status"}

    if text.startswith("CANCEL "):
        ticker = text.split()[1]
        return {"cmd": "cancel", "ticker": ticker}

    parts = text.replace("@", "").split()
    # BUY <shares> <ticker> <price>  OR  BUY <ticker> <shares> @ <price>
    if len(parts) >= 4 and parts[0] in ("BUY", "SELL"):
        action = parts[0]
        try:
            # Try: BUY 0.062 NVDA 876
            shares = float(parts[1])
            ticker = parts[2]
            price  = float(parts[3])
            return {"cmd": action, "shares": shares, "ticker": ticker, "price": price}
        except (ValueError, IndexError):
            pass
        try:
            # Try: BUY NVDA 0.062 876
            ticker = parts[1]
            shares = float(parts[2])
            price  = float(parts[3])
            return {"cmd": action, "shares": shares, "ticker": ticker, "price": price}
        except (ValueError, IndexError):
            pass
    return None


def fmt_portfolio_status() -> str:
    pv           = portfolio_value()
    total_return = pv - INITIAL_CAPITAL
    total_pct    = total_return / INITIAL_CAPITAL * 100
    peak         = portfolio["peak_value"]
    drawdown     = (pv - peak) / peak * 100 if pv < peak else 0.0

    lines = [
        "📊 *Status do Portfólio*",
        f"⏰ {ts()} ET\n",
        f"💰 Valor: *${pv:.2f}*",
        f"{'📈' if total_return >= 0 else '📉'} Retorno: *{total_return:+.2f}* ({total_pct:+.1f}%)",
        f"💵 Caixa: ${portfolio['cash']:.2f} ({portfolio['cash']/pv*100:.0f}%)",
        f"📉 Drawdown: {drawdown:.1f}%\n",
    ]

    if portfolio["positions"]:
        lines.append("📂 *Posições abertas*")
        for tkr, pos in portfolio["positions"].items():
            cp    = pos["current_price"]
            pnl   = (cp - pos["entry_price"]) * pos["shares"]
            pnl_p = (cp - pos["entry_price"]) / pos["entry_price"] * 100
            pct   = pos["shares"] * cp / pv * 100
            lines.append(
                f"{'📈' if pnl >= 0 else '📉'} *{tkr}* {pct:.0f}% do portfólio\n"
                f"   {pos['shares']:.4f} shares | ${pos['entry_price']:.2f}→${cp:.2f} | PnL {pnl:+.2f} ({pnl_p:+.1f}%)\n"
                f"   🛑${pos['stop_loss']:.2f}  ✅${pos['take_profit']:.2f}  (desde {pos['opened_at']})"
            )
        lines.append("")

    if portfolio["pending"]:
        lines.append("⏳ *Aguardando sua confirmação*")
        for tkr, sig in portfolio["pending"].items():
            alloc  = calc_position_size(sig)
            shares = alloc / sig["currentPrice"]
            lines.append(
                f"  {sig['signal']} *{tkr}* @ ~${sig['currentPrice']:.2f}\n"
                f"  Sugestão: {shares:.4f} shares (${alloc:.2f})\n"
                f"  Para confirmar: `{sig['signal']} {shares:.4f} {tkr} @ PRECO`"
            )
        lines.append("")

    if portfolio["history"]:
        wins = sum(1 for t in portfolio["history"] if t["pnl"] > 0)
        wr   = wins / len(portfolio["history"]) * 100
        total_pnl = sum(t["pnl"] for t in portfolio["history"])
        lines.append(f"🏆 Win rate: {wr:.0f}% | PnL fechado: {total_pnl:+.2f}")

    lines.append("\n_Comandos: `BUY 0.1 NVDA @ 880` · `SELL 0.1 NVDA @ 900` · `STATUS`_")
    return "\n".join(lines)


def fmt_scan_report(new_signals: list, closed_trades: list) -> str:
    pv           = portfolio_value()
    total_return = pv - INITIAL_CAPITAL
    total_pct    = total_return / INITIAL_CAPITAL * 100

    lines = [
        "🔔 *Novo scan — Portfólio*",
        f"⏰ {ts()} ET | Valor: *${pv:.2f}* ({total_return:+.2f} / {total_pct:+.1f}%)\n",
    ]

    if closed_trades:
        lines.append("🔒 *Fechamentos automáticos*")
        for t in closed_trades:
            lines.append(f"  {'✅' if t['pnl'] > 0 else '❌'} {t['ticker']} {t['pnl']:+.2f} ({t['pnl_pct']:+.1f}%) — {t['reason']}")
        lines.append("")

    if new_signals:
        lines.append("🚦 *Sinais — aguardando sua confirmação*")
        for s in new_signals:
            sig_e  = {"BUY": "🟢", "SELL": "🔴"}.get(s["signal"], "🟡")
            alloc  = calc_position_size(s) if s["signal"] == "BUY" else 0
            shares = alloc / s["currentPrice"] if alloc else 0
            pos    = portfolio["positions"].get(s["ticker"])

            conv_e = {"high": "🔥", "medium": "⚡", "low": "💧"}.get(s.get("conviction","medium"), "⚡")
            lines.append(f"{sig_e} *{s['ticker']}* {s['signal']} {conv_e} | Conf: {s['confidence']}% | Risco: {s['riskScore']}/10")
            lines.append(f"   Preço ref: ${s['currentPrice']:.2f} | R/R: {s.get('riskReward','—')}")
            if s["signal"] == "BUY":
                lines.append(f"   Sugestão: {shares:.4f} shares (${alloc:.2f} = {alloc/pv*100:.0f}% portfólio)")
                lines.append(f"   🛑 Stop: ${s['stopLoss']:.2f}  ✅ Target: ${s['takeProfit']:.2f}")
                lines.append(f"   ➡️ Confirmar: `BUY {shares:.4f} {s['ticker']} @ PRECO_EXECUTADO`")
            elif s["signal"] == "SELL" and pos:
                lines.append(f"   Posição: {pos['shares']:.4f} shares | Entrada ${pos['entry_price']:.2f}")
                lines.append(f"   ➡️ Confirmar: `SELL {pos['shares']:.4f} {s['ticker']} @ PRECO_EXECUTADO`")
            if s.get("catalysts"):
                lines.append(f"   📰 {s['catalysts'][:120]}")
            if s.get("thesis"):
                lines.append(f"   🎯 _{s['thesis'][:150]}_")
            elif s.get("reasoning"):
                lines.append(f"   🎯 _{s['reasoning'][:150]}_")
            lines.append("")

    lines.append(f"💵 Caixa: ${portfolio['cash']:.2f} ({portfolio['cash']/pv*100:.0f}%)")
    lines.append("_Para ver portfólio: `STATUS`_")
    return "\n".join(lines)


# ── Telegram command listener (polling thread) ────────────────────────

def telegram_listener():
    global _last_update_id
    log.info("Telegram listener started")
    while True:
        try:
            url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
            resp = requests.get(url, params={"offset": _last_update_id + 1, "timeout": 30}, timeout=40)
            if not resp.ok:
                time.sleep(5)
                continue
            data = resp.json()
            for update in data.get("result", []):
                _last_update_id = update["update_id"]
                msg = update.get("message", {})
                chat_id = str(msg.get("chat", {}).get("id", ""))
                text    = msg.get("text", "").strip()
                if not text or chat_id != TELEGRAM_CHAT_ID:
                    continue
                log.info(f"Received: {text!r}")
                handle_command(text)
        except Exception as e:
            log.error(f"Listener error: {e}")
            time.sleep(10)


def handle_command(text: str):
    parsed = parse_trade_command(text)
    if not parsed:
        send_telegram(
            "❓ Comando não reconhecido.\n\n"
            "*Comandos disponíveis:*\n"
            "`BUY 0.1 NVDA @ 880` — registrar compra\n"
            "`SELL 0.1 NVDA @ 900` — registrar venda\n"
            "`STATUS` — ver portfólio\n"
            "`CANCEL NVDA` — cancelar sinal pendente"
        )
        return

    cmd = parsed["cmd"]

    if cmd == "status":
        send_telegram(fmt_portfolio_status())

    elif cmd == "cancel":
        ticker = parsed["ticker"]
        if ticker in portfolio["pending"]:
            portfolio["pending"].pop(ticker)
            send_telegram(f"🚫 Sinal de *{ticker}* cancelado.")
        else:
            send_telegram(f"ℹ️ Nenhum sinal pendente para *{ticker}*.")

    elif cmd == "BUY":
        sig = portfolio["pending"].get(parsed["ticker"])
        reply = register_buy(parsed["ticker"], parsed["shares"], parsed["price"], sig)
        send_telegram(reply)
        pv = portfolio_value()
        if pv > portfolio["peak_value"]:
            portfolio["peak_value"] = pv

    elif cmd == "SELL":
        reply = register_sell(parsed["ticker"], parsed["shares"], parsed["price"])
        send_telegram(reply)
        pv = portfolio_value()
        if pv > portfolio["peak_value"]:
            portfolio["peak_value"] = pv


# ── Main scan ─────────────────────────────────────────────────────────

def scan_all() -> None:
    log.info(f"Scanning: {TICKERS}")
    signals = {}

    for ticker in TICKERS:
        try:
            log.info(f"Analyzing {ticker}...")
            sig = analyze_ticker(ticker)
            signals[ticker] = sig
            log.info(f"{ticker}: {sig['signal']} conf={sig['confidence']}% risk={sig['riskScore']}")
            time.sleep(2)
        except Exception as e:
            log.error(f"Error {ticker}: {e}")
            time.sleep(5)

    if not signals:
        return

    # Auto check stops
    closed_trades = check_stops(signals)

    # Collect actionable signals (BUY/SELL only)
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

    report = fmt_scan_report(actionable, closed_trades)
    send_telegram(report)
    log.info(f"Report sent. Portfolio: ${pv:.2f}")


# ── Entry point ───────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Stock Signal Agent [Portfolio + Confirmation Mode] started ===")
    log.info(f"Tickers: {TICKERS} | Capital: ${INITIAL_CAPITAL} | Interval: {SCAN_INTERVAL_MIN}min")

    # Start Telegram listener in background thread
    t = threading.Thread(target=telegram_listener, daemon=True)
    t.start()

    send_telegram(
        "🚀 *Stock Signal Agent — Modo Portfólio*\n"
        f"Capital inicial: *${INITIAL_CAPITAL:.2f}*\n"
        f"Tickers: {', '.join(TICKERS)}\n"
        f"Máx por posição: {MAX_POSITION_PCT*100:.0f}% | Máx posições simultâneas: {MAX_POSITIONS}\n"
        f"Confiança mínima para sinal: {MIN_CONFIDENCE}%\n\n"
        "📌 *Como usar:*\n"
        "Quando receber um sinal, execute no homebroker e confirme:\n"
        "`BUY 0.062 NVDA @ 878.50`\n"
        "`SELL 0.062 NVDA @ 910.00`\n"
        "`STATUS` — ver portfólio completo\n"
        "`CANCEL NVDA` — ignorar sinal"
    )

    while True:
        if is_market_hours():
            scan_all()
        else:
            now = datetime.now(MARKET_TZ)
            log.info(f"Market closed ({now.strftime('%a %H:%M')} ET) — sleeping")
        time.sleep(SCAN_INTERVAL_MIN * 60)


if __name__ == "__main__":
    main()
