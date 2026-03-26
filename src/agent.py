import os, json, time, logging, threading, requests
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time, timedelta
from zoneinfo import ZoneInfo
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.environ["ANTHROPIC_API_KEY"]
TELEGRAM_TOKEN     = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]
ALPACA_API_KEY     = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET      = os.environ["ALPACA_SECRET"]
INITIAL_CAPITAL    = float(os.environ.get("CAPITAL", "200"))
TICKERS            = [t.strip() for t in os.environ.get("TICKERS", "NVDA,TSLA,AMD,PLTR,COIN").split(",") if t.strip()]
SCAN_INTERVAL_MIN  = int(os.environ.get("SCAN_INTERVAL_MIN", "30"))
MAX_POSITION_PCT   = 0.35
MIN_CONFIDENCE     = 60
MAX_POSITIONS      = 4
CONFIRM_TIMEOUT_S  = 120   # 2 minutes to confirm

MARKET_TZ    = ZoneInfo("America/New_York")
MARKET_OPEN  = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)

# ── Alpaca clients ────────────────────────────────────────────────────
alpaca      = TradingClient(ALPACA_API_KEY, ALPACA_SECRET, paper=True)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET)

# ── Portfolio state ───────────────────────────────────────────────────
portfolio = {
    "cash": INITIAL_CAPITAL, "positions": {}, "history": [],
    "peak_value": INITIAL_CAPITAL,
    "pending": {},    # { ticker: {signal, expires_at} }
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

def calc_position_size(signal: dict) -> float:
    pv        = portfolio_value()
    max_alloc = pv * MAX_POSITION_PCT
    conf_mult = signal["confidence"] / 100
    risk_mult = (11 - signal["riskScore"]) / 10
    return round(min(max_alloc * conf_mult * risk_mult, portfolio["cash"]), 2)


# ── Market data via Alpaca Data API ──────────────────────────────────

def fetch_market_data(ticker: str) -> dict:
    """Fetch real-time price + technicals directly from Alpaca."""

    # Latest quote via IEX (free plan)
    quote_req = StockLatestQuoteRequest(symbol_or_symbols=ticker, feed=DataFeed.IEX)
    quote     = data_client.get_stock_latest_quote(quote_req)[ticker]
    price     = round(float((quote.bid_price + quote.ask_price) / 2), 2)

    # Daily bars via IEX — last 90 days for technicals
    end   = datetime.now(ZoneInfo("UTC"))
    start = end - timedelta(days=90)
    bars_req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=start, end=end,
        feed=DataFeed.IEX
    )
    bars  = data_client.get_stock_bars(bars_req)[ticker]
    close = pd.Series([b.close for b in bars])

    prev    = round(float(close.iloc[-2]) if len(close) >= 2 else price, 2)
    chg_pct = round((price - prev) / prev * 100, 2) if prev else 0.0
    ma20    = round(float(close.tail(20).mean()), 2) if len(close) >= 20 else price
    ma50    = round(float(close.tail(50).mean()), 2) if len(close) >= 50 else price
    high52  = round(float(close.tail(252).max()), 2) if len(close) >= 10 else price
    low52   = round(float(close.tail(252).min()), 2) if len(close) >= 10 else price

    # RSI(14)
    delta = close.diff()
    gain  = delta.clip(lower=0).tail(14).mean()
    loss  = (-delta.clip(upper=0)).tail(14).mean()
    rsi   = round(float(100 - (100 / (1 + gain / loss))), 1) if loss and loss != 0 else 50.0

    return {
        "price": price, "prev_close": prev, "chg_pct": chg_pct,
        "ma20": ma20, "ma50": ma50, "rsi": rsi,
        "high52": high52, "low52": low52,
        "support":    round(low52 * 1.05, 2),
        "resistance": round(high52 * 0.95, 2),
    }


# ── Claude analysis ───────────────────────────────────────────────────

def analyze_ticker(ticker: str, mkt: dict) -> dict:
    pv       = portfolio_value()
    open_pos = list(portfolio["positions"].keys())
    trend    = "bullish" if mkt["price"] > mkt["ma20"] > mkt["ma50"] else \
               "bearish" if mkt["price"] < mkt["ma20"] < mkt["ma50"] else "neutral"
    macd_sig = "bullish" if mkt["ma20"] > mkt["ma50"] else \
               "bearish" if mkt["ma20"] < mkt["ma50"] else "neutral"

    # Build open positions context with original strategy
    positions_context = ""
    if portfolio["positions"]:
        positions_context = "\nOPEN POSITIONS & ORIGINAL STRATEGY:\n"
        for t, p in portfolio["positions"].items():
            pnl_pct = (p["current_price"] - p["entry_price"]) / p["entry_price"] * 100
            positions_context += (
                f"- {t}: {p['shares']:.4f} shares @ ${p['entry_price']:.2f} entry "
                f"(now ${p['current_price']:.2f}, {pnl_pct:+.1f}%)\n"
                f"  Stop: ${p['stop_loss']:.2f} | Target: ${p['take_profit']:.2f} | R/R: {p.get('risk_reward','-')}\n"
                f"  Entry date: {p.get('strategy_date', p.get('opened_at', '-'))}\n"
                f"  Original thesis: {p.get('thesis', 'N/A')}\n"
                f"  Catalysts: {p.get('catalysts', 'N/A')}\n"
                f"  Conviction at entry: {p.get('conviction','-')} ({p.get('original_conf','-')}% confidence)\n"
            )

    system_prompt = f"""You are a professional-grade portfolio strategist with full discretionary control.

MANDATE: Generate maximum return on this portfolio.
Portfolio value: ${pv:.2f} | Cash: ${portfolio['cash']:.2f} | Open positions: {open_pos or 'none'}
Max per position: {MAX_POSITION_PCT*100:.0f}% | Profile: AGGRESSIVE.
{positions_context}
When analyzing a ticker that is already in the portfolio, evaluate whether the ORIGINAL THESIS
is still valid. Only recommend SELL if the thesis is broken, target is reached, or risk has
materially changed. Do NOT recommend SELL just because of short-term volatility if the thesis holds.

VERIFIED REAL-TIME DATA for {ticker}:
- Price: ${mkt['price']} ({mkt['chg_pct']:+.2f}% today)
- MA20: ${mkt['ma20']} | MA50: ${mkt['ma50']} | Trend: {trend}
- RSI(14): {mkt['rsi']} | MACD: {macd_sig}
- 52w: ${mkt['low52']}–${mkt['high52']}
- Support: ${mkt['support']} | Resistance: ${mkt['resistance']}

Rules:
- Stop loss must be structure-based (not arbitrary %)
- Only BUY with strong conviction and clear thesis
- SELL when thesis broken or target reached

Return ONLY raw JSON:
{{
  "ticker": string, "companyName": string,
  "signal": "BUY"|"SELL"|"HOLD",
  "confidence": number 0-100,
  "currentPrice": {mkt['price']},
  "priceChangePct": {mkt['chg_pct']},
  "riskScore": number 1-10,
  "rsi": {mkt['rsi']},
  "macdSignal": "{macd_sig}",
  "trend": "{trend}",
  "support": {mkt['support']},
  "resistance": {mkt['resistance']},
  "stopLoss": number,
  "takeProfit": number,
  "limitPrice": number,
  "riskReward": string,
  "conviction": "high"|"medium"|"low",
  "catalysts": string,
  "thesis": string
}}"""

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 800,
        "system": system_prompt,
        "messages": [{"role": "user", "content": f"Analyze {ticker}. Date: {datetime.now(MARKET_TZ).strftime('%Y-%m-%d %H:%M')} ET"}]
    }
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json=payload, timeout=60
    )
    if not resp.ok:
        log.error(f"API {resp.status_code}: {resp.text}")
    resp.raise_for_status()
    text = "".join(b["text"] for b in resp.json().get("content", []) if b.get("type") == "text")
    text = text.replace("```json", "").replace("```", "").strip()
    s, e = text.find("{"), text.rfind("}")
    result = json.loads(text[s:e+1])
    result["currentPrice"]   = mkt["price"]
    result["priceChangePct"] = mkt["chg_pct"]
    return result


# ── Alpaca order execution ────────────────────────────────────────────

def execute_buy(ticker: str, signal: dict) -> dict:
    """Place simple limit order on Alpaca (paper).
    Bracket orders don't support fractional shares — stops monitored internally."""
    alloc       = calc_position_size(signal)
    limit_price = round(signal.get("limitPrice", signal["currentPrice"] * 1.001), 2)
    qty         = round(alloc / limit_price, 6)

    order_req = LimitOrderRequest(
        symbol        = ticker,
        qty           = qty,
        side          = OrderSide.BUY,
        time_in_force = TimeInForce.DAY,
        limit_price   = limit_price,
    )
    order = alpaca.submit_order(order_req)
    log.info(f"Alpaca BUY submitted: {ticker} {qty:.6f} @ ${limit_price} | order_id={order.id}")
    return {"order_id": str(order.id), "qty": qty, "limit_price": limit_price, "alloc": alloc}


def execute_sell(ticker: str, signal: dict, pos: dict) -> dict:
    """Place limit sell order on Alpaca — verifies position exists first."""
    # Verify position exists in Alpaca before selling
    try:
        alpaca_pos = alpaca.get_open_position(ticker)
        qty = float(alpaca_pos.qty)
    except Exception:
        raise ValueError(f"No open position for {ticker} in Alpaca — cannot sell.")

    limit_price = round(signal.get("currentPrice", pos["entry_price"]) * 0.999, 2)
    order_req = LimitOrderRequest(
        symbol        = ticker,
        qty           = qty,
        side          = OrderSide.SELL,
        time_in_force = TimeInForce.DAY,
        limit_price   = limit_price,
    )
    order = alpaca.submit_order(order_req)
    log.info(f"Alpaca SELL submitted: {ticker} {qty:.4f} @ ${limit_price} | order_id={order.id}")
    return {"order_id": str(order.id), "qty": qty, "limit_price": limit_price}


def get_alpaca_account() -> dict:
    acc = alpaca.get_account()
    return {
        "buying_power": float(acc.buying_power),
        "portfolio_value": float(acc.portfolio_value),
        "cash": float(acc.cash),
    }

def sync_with_alpaca() -> str:
    """Sync portfolio state from Alpaca on startup — only real positions."""
    try:
        acc       = alpaca.get_account()
        positions = alpaca.get_all_positions()

        portfolio["cash"] = float(acc.cash)

        # Real tickers in Alpaca
        real_tickers = {pos.symbol for pos in positions}

        # Remove phantom positions (in agent memory but not in Alpaca)
        phantoms = [t for t in list(portfolio["positions"].keys()) if t not in real_tickers]
        for t in phantoms:
            log.warning(f"Removing phantom position: {t}")
            portfolio["positions"].pop(t, None)

        synced = []
        for pos in positions:
            ticker = pos.symbol
            price  = float(pos.current_price or pos.avg_entry_price)
            entry  = float(pos.avg_entry_price)
            shares = float(pos.qty)
            cost   = entry * shares

            if ticker not in portfolio["positions"]:
                portfolio["positions"][ticker] = {
                    "shares":        round(shares, 6),
                    "entry_price":   round(entry, 2),
                    "current_price": round(price, 2),
                    "stop_loss":     round(entry * 0.92, 2),
                    "take_profit":   round(entry * 1.20, 2),
                    "opened_at":     "sync",
                    "allocated":     round(cost, 2),
                }
                synced.append(ticker)
                log.info(f"Synced from Alpaca: {ticker} {shares:.4f} @ ${entry:.2f}")
            else:
                # Update qty and price from Alpaca (source of truth)
                portfolio["positions"][ticker]["shares"]        = round(shares, 6)
                portfolio["positions"][ticker]["current_price"] = round(price, 2)

        if portfolio["peak_value"] < portfolio_value():
            portfolio["peak_value"] = portfolio_value()

        if synced:
            return (f"🔄 *Portfólio sincronizado com Alpaca*\n"
                    f"Posições importadas: {', '.join(synced)}\n"
                    f"⚠️ Stop/target com valores padrão (8%/20%). "
                    f"Use `SETSTOP COIN 170 200` para ajustar.")
        return ""
    except Exception as e:
        log.error(f"Sync error: {e}")
        return f"⚠️ Erro ao sincronizar com Alpaca: {e}"


# ── Portfolio tracking ────────────────────────────────────────────────

def record_buy(ticker: str, signal: dict, order_info: dict) -> None:
    alloc = order_info["alloc"]
    portfolio["cash"] -= alloc
    portfolio["positions"][ticker] = {
        "shares":        order_info["qty"],
        "entry_price":   order_info["limit_price"],
        "current_price": order_info["limit_price"],
        "stop_loss":     signal["stopLoss"],
        "take_profit":   signal["takeProfit"],
        "opened_at":     ts(),
        "allocated":     alloc,
        "order_id":      order_info["order_id"],
        # Strategy context — preserved for all future decisions
        "thesis":        signal.get("thesis", ""),
        "catalysts":     signal.get("catalysts", ""),
        "conviction":    signal.get("conviction", "medium"),
        "original_conf": signal.get("confidence", 0),
        "risk_reward":   signal.get("riskReward", ""),
        "strategy_date": ts(),
    }

def record_sell(ticker: str, order_info: dict) -> dict | None:
    pos = portfolio["positions"].pop(ticker, None)
    if not pos: return None
    proceeds   = order_info["qty"] * order_info["limit_price"]
    pnl        = proceeds - pos["allocated"]
    pnl_pct    = pnl / pos["allocated"] * 100
    portfolio["cash"] += proceeds
    trade = {"ticker": ticker, "entry": pos["entry_price"], "exit": order_info["limit_price"],
             "shares": order_info["qty"], "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
             "reason": "sinal SELL", "closed_at": ts()}
    portfolio["history"].append(trade)
    return trade

def execute_auto_sell(ticker: str, price: float, reason: str) -> dict | None:
    """Execute sell order on Alpaca automatically — no confirmation needed."""
    pos = portfolio["positions"].get(ticker)
    if not pos:
        return None
    try:
        # Verify position exists in Alpaca
        alpaca_pos = alpaca.get_open_position(ticker)
        qty = float(alpaca_pos.qty)
        order_req = LimitOrderRequest(
            symbol        = ticker,
            qty           = qty,
            side          = OrderSide.SELL,
            time_in_force = TimeInForce.DAY,
            limit_price   = round(price * 0.999, 2),
        )
        order = alpaca.submit_order(order_req)
        log.info(f"AUTO SELL {ticker} @ ${price:.2f} order_id={order.id}")
    except Exception as e:
        log.error(f"Auto sell error {ticker}: {e}")

    pos_copy = portfolio["positions"].pop(ticker)
    proceeds = pos_copy["shares"] * price
    pnl      = proceeds - pos_copy["allocated"]
    portfolio["cash"] += proceeds
    trade = {
        "ticker": ticker, "entry": pos_copy["entry_price"], "exit": price,
        "shares": pos_copy["shares"], "pnl": round(pnl, 2),
        "pnl_pct": round(pnl / pos_copy["allocated"] * 100, 2),
        "reason": reason, "closed_at": ts(),
    }
    portfolio["history"].append(trade)
    return trade


def check_stops(prices: dict) -> list:
    closed = []
    for ticker, pos in list(portfolio["positions"].items()):
        price = prices.get(ticker, pos["current_price"])
        pos["current_price"] = price

        if price <= pos["stop_loss"]:
            log.info(f"STOP HIT {ticker} @ ${price:.2f}")
            trade = execute_auto_sell(ticker, price, "🛑 stop loss automático")
            if trade:
                closed.append(trade)
                pnl = trade["pnl"]
                send_telegram(
                    f"🛑 *Stop loss executado — {ticker}*\n"
                    f"  Vendido automaticamente @ ${price:.2f}\n"
                    f"  PnL: *{pnl:+.2f}* ({trade['pnl_pct']:+.1f}%)\n"
                    f"  💵 Caixa: ${portfolio['cash']:.2f}"
                )

        elif price >= pos["take_profit"]:
            log.info(f"TARGET HIT {ticker} @ ${price:.2f}")
            trade = execute_auto_sell(ticker, price, "✅ take profit automático")
            if trade:
                closed.append(trade)
                pnl = trade["pnl"]
                send_telegram(
                    f"✅ *Take profit executado — {ticker}*\n"
                    f"  Vendido automaticamente @ ${price:.2f}\n"
                    f"  PnL: *{pnl:+.2f}* ({trade['pnl_pct']:+.1f}%)\n"
                    f"  💵 Caixa: ${portfolio['cash']:.2f}"
                )
    return closed


# ── Telegram ─────────────────────────────────────────────────────────

def send_telegram(text: str, buttons: list = None) -> None:
    """Send message with optional inline keyboard buttons.
    buttons = [ [{"text": "label", "callback_data": "data"}, ...], ... ]
    """
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    if buttons:
        payload["reply_markup"] = {"inline_keyboard": buttons}
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json=payload, timeout=15)
    except Exception as e:
        log.error(f"Telegram send: {e}")

def answer_callback(callback_id: str, text: str = "") -> None:
    """Acknowledge a button press (removes loading spinner)."""
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery",
            json={"callback_query_id": callback_id, "text": text}, timeout=10)
    except Exception as e:
        log.error(f"Telegram callback: {e}")

def fmt_status() -> str:
    pv  = portfolio_value()
    ret = pv - INITIAL_CAPITAL
    pct = ret / INITIAL_CAPITAL * 100
    pk  = portfolio["peak_value"]
    dd  = (pv - pk) / pk * 100 if pv < pk else 0.0

    # Sync with Alpaca account
    try:
        acc = get_alpaca_account()
        alpaca_line = f"🏦 Alpaca paper: ${acc['portfolio_value']:.2f} (cash ${acc['cash']:.2f})"
    except:
        alpaca_line = ""

    lines = ["📊 *Status do Portfólio*", f"⏰ {ts()} ET\n",
             f"💰 Valor: *${pv:.2f}*",
             f"{'📈' if ret >= 0 else '📉'} Retorno: *{ret:+.2f}* ({pct:+.1f}%)",
             f"💵 Caixa: ${portfolio['cash']:.2f} ({portfolio['cash']/pv*100:.0f}%)",
             f"📉 Drawdown: {dd:.1f}%"]
    if alpaca_line:
        lines.append(alpaca_line)
    lines.append("")

    if portfolio["positions"]:
        lines.append("📂 *Posições abertas*")
        for tkr, pos in portfolio["positions"].items():
            cp    = pos["current_price"]
            pnl   = (cp - pos["entry_price"]) * pos["shares"]
            pnl_p = (cp - pos["entry_price"]) / pos["entry_price"] * 100
            conv_e = {"high": "🔥", "medium": "⚡", "low": "💧"}.get(pos.get("conviction","medium"), "⚡")
            thesis_line = f"\n   🎯 _{pos['thesis'][:120]}_" if pos.get("thesis") else ""
            lines.append(
                f"{'📈' if pnl >= 0 else '📉'} *{tkr}* {conv_e} | {pos['shares']:.4f} shares\n"
                f"   Entrada: ${pos['entry_price']:.2f} em {pos.get('strategy_date', pos.get('opened_at','-'))}\n"
                f"   Atual: ${cp:.2f} | PnL {pnl:+.2f} ({pnl_p:+.1f}%)\n"
                f"   🛑${pos['stop_loss']:.2f}  ✅${pos['take_profit']:.2f}  R/R: {pos.get('risk_reward','-')}"
                f"{thesis_line}"
            )

    if portfolio["pending"]:
        lines.append("\n⏳ *Aguardando confirmação (2 min)*")
        now = time.time()
        for tkr, p in list(portfolio["pending"].items()):
            remaining = max(0, int(p["expires_at"] - now))
            sig = p["signal"]
            alloc  = calc_position_size(sig)
            shares = round(alloc / sig["currentPrice"], 4)
            lines.append(f"  {sig['signal']} *{tkr}* @ ${sig['currentPrice']:.2f} | ⏱ {remaining}s\n"
                         f"  → `CONFIRM {tkr}` ou `CANCEL {tkr}`")

    if portfolio["history"]:
        wins = sum(1 for t in portfolio["history"] if t["pnl"] > 0)
        wr   = wins / len(portfolio["history"]) * 100
        total_pnl = sum(t["pnl"] for t in portfolio["history"])
        lines.append(f"\n🏆 Win rate: {wr:.0f}% | PnL fechado: {total_pnl:+.2f}")

    lines.append("\n_`CONFIRM NVDA` · `CANCEL NVDA` · `STATUS`_")
    return "\n".join(lines)

def fmt_scan_report(signals: list, closed_trades: list) -> str:
    pv  = portfolio_value()
    ret = pv - INITIAL_CAPITAL
    lines = ["🔔 *Novo scan — Portfólio*",
             f"⏰ {ts()} ET | *${pv:.2f}* ({ret:+.2f})\n"]

    if closed_trades:
        lines.append("🔒 *Fechamentos automáticos*")
        for t in closed_trades:
            lines.append(f"  {'✅' if t['pnl'] > 0 else '❌'} {t['ticker']} {t['pnl']:+.2f} ({t['pnl_pct']:+.1f}%) — {t['reason']}")
        lines.append("")

    if signals:
        lines.append(f"🚦 *Sinais — você tem {CONFIRM_TIMEOUT_S//60} min para confirmar*")
        for s in signals:
            sig_e  = {"BUY": "🟢", "SELL": "🔴"}.get(s["signal"], "🟡")
            conv_e = {"high": "🔥", "medium": "⚡", "low": "💧"}.get(s.get("conviction","medium"), "⚡")
            alloc  = calc_position_size(s) if s["signal"] == "BUY" else 0
            shares = round(alloc / s["currentPrice"], 4) if alloc else 0
            lp     = round(s.get("limitPrice", s["currentPrice"]), 2)
            pos    = portfolio["positions"].get(s["ticker"])

            lines.append(f"{sig_e} *{s['ticker']}* {s['signal']} {conv_e} | Conf: {s['confidence']}% | Risco: {s['riskScore']}/10")
            lines.append(f"   💵 ${s['currentPrice']:.2f} ({s['priceChangePct']:+.1f}%) | RSI: {s.get('rsi','—')} | R/R: {s.get('riskReward','—')}")
            if s["signal"] == "BUY":
                lines.append(f"   📋 Limit: ${lp:.2f} | {shares:.4f} shares (${alloc:.2f} = {alloc/pv*100:.0f}% portfólio)")
                lines.append(f"   🛑 Stop: ${s['stopLoss']:.2f}  ✅ Target: ${s['takeProfit']:.2f}")
                lines.append(f"   ✅ Executar: `CONFIRM {s['ticker']}`")
                lines.append(f"   ❌ Ignorar:  `CANCEL {s['ticker']}`")
            elif s["signal"] == "SELL" and pos:
                lines.append(f"   Posição atual: {pos['shares']:.4f} shares | entrada ${pos['entry_price']:.2f}")
                lines.append(f"   ✅ Vender: `CONFIRM {s['ticker']}`")
                lines.append(f"   ❌ Manter: `CANCEL {s['ticker']}`")
            if s.get("catalysts"):
                lines.append(f"   📰 {s['catalysts'][:130]}")
            if s.get("thesis"):
                lines.append(f"   🎯 _{s['thesis'][:150]}_")
            lines.append("")

    lines.append(f"💵 Caixa: ${portfolio['cash']:.2f} | `STATUS` para portfólio completo")
    return "\n".join(lines)

def build_signal_buttons(signals: list) -> list:
    rows = []
    for s in signals:
        ticker = s["ticker"]
        action = s["signal"]
        rows.append([
            {"text": f"✅ {action} {ticker}", "callback_data": f"CONFIRM_{ticker}"},
            {"text": f"❌ Ignorar {ticker}",  "callback_data": f"CANCEL_{ticker}"},
        ])
    return rows


# ── Pending expiry watchdog ───────────────────────────────────────────

def expiry_watchdog():
    """Background thread: cancel pending signals after timeout."""
    while True:
        now = time.time()
        expired = [tkr for tkr, p in list(portfolio["pending"].items()) if now > p["expires_at"]]
        for tkr in expired:
            portfolio["pending"].pop(tkr, None)
            send_telegram(f"⏱ Sinal *{tkr}* expirou sem confirmação — cancelado automaticamente.")
            log.info(f"Signal {tkr} expired")
        time.sleep(10)


# ── Telegram command listener ─────────────────────────────────────────

def parse_command(text: str) -> dict | None:
    text = text.strip().upper()
    if text in ("STATUS", "/STATUS", "PORTFOLIO"):
        return {"cmd": "status"}
    if text.startswith("CONFIRM ") and len(text.split()) == 2:
        return {"cmd": "confirm", "ticker": text.split()[1]}
    if text.startswith("CANCEL ") and len(text.split()) == 2:
        return {"cmd": "cancel", "ticker": text.split()[1]}
    # SETSTOP COIN 170 200
    parts = text.split()
    if parts[0] == "SETSTOP" and len(parts) == 4:
        try:
            return {"cmd": "setstop", "ticker": parts[1],
                    "stop": float(parts[2]), "target": float(parts[3])}
        except ValueError:
            pass
    # SETTHESIS COIN RSI oversold bounce with crypto institutional demand
    if parts[0] == "SETTHESIS" and len(parts) >= 3:
        return {"cmd": "setthesis", "ticker": parts[1], "thesis": " ".join(parts[2:])}
    return None

def handle_command(text: str):
    parsed = parse_command(text)
    if not parsed:
        send_telegram(
            "❓ *Comandos disponíveis:*\n"
            "`CONFIRM NVDA` — executar ordem na Alpaca\n"
            "`CANCEL NVDA` — ignorar sinal\n"
            "`STATUS` — ver portfólio completo"
        )
        return

    cmd = parsed["cmd"]

    if cmd == "status":
        send_telegram(fmt_status())

    elif cmd == "setstop":
        tkr  = parsed["ticker"]
        pos  = portfolio["positions"].get(tkr)
        if not pos:
            send_telegram(f"❌ Nenhuma posição aberta em *{tkr}*.")
        else:
            pos["stop_loss"]   = parsed["stop"]
            pos["take_profit"] = parsed["target"]
            send_telegram(
                f"✅ *{tkr}* atualizado\n"
                f"  🛑 Stop: ${parsed['stop']:.2f}\n"
                f"  ✅ Target: ${parsed['target']:.2f}"
            )

    elif cmd == "setthesis":
        tkr  = parsed["ticker"]
        pos  = portfolio["positions"].get(tkr)
        if not pos:
            send_telegram(f"❌ Nenhuma posição aberta em *{tkr}*.")
        else:
            pos["thesis"]       = parsed["thesis"]
            pos["strategy_date"] = pos.get("strategy_date", ts())
            send_telegram(
                f"✅ *{tkr}* — tese registrada:\n"
                f"  🎯 _{parsed['thesis']}_"
            )

    elif cmd == "cancel":
        tkr = parsed["ticker"]
        portfolio["pending"].pop(tkr, None)
        send_telegram(f"🚫 Sinal de *{tkr}* cancelado.")

    elif cmd == "confirm":
        tkr     = parsed["ticker"]
        pending = portfolio["pending"].get(tkr)
        if not pending:
            send_telegram(f"⚠️ Nenhum sinal pendente para *{tkr}*.\n_O sinal pode ter expirado (2 min)._")
            return
        if time.time() > pending["expires_at"]:
            portfolio["pending"].pop(tkr, None)
            send_telegram(f"⏱ Sinal de *{tkr}* já expirou. Aguarde o próximo scan.")
            return

        sig = pending["signal"]
        send_telegram(f"⚙️ Executando ordem *{sig['signal']} {tkr}* na Alpaca...")

        try:
            if sig["signal"] == "BUY":
                order_info = execute_buy(tkr, sig)
                record_buy(tkr, sig, order_info)
                portfolio["pending"].pop(tkr, None)
                pv = portfolio_value()
                if pv > portfolio["peak_value"]: portfolio["peak_value"] = pv
                send_telegram(
                    f"✅ *BUY executado — {tkr}*\n"
                    f"  {order_info['qty']:.4f} shares\n"
                    f"  Limit: ${order_info['limit_price']:.2f}\n"
                    f"  Valor: ${order_info['alloc']:.2f}\n"
                    f"  🛑 Stop: ${sig['stopLoss']:.2f}  ✅ Target: ${sig['takeProfit']:.2f}\n"
                    f"  🏦 Order ID: `{order_info['order_id'][:16]}...`\n"
                    f"  💵 Caixa restante: ${portfolio['cash']:.2f}"
                )
            elif sig["signal"] == "SELL":
                pos = portfolio["positions"].get(tkr)
                if not pos:
                    send_telegram(f"❌ Nenhuma posição aberta em *{tkr}*.")
                    return
                order_info = execute_sell(tkr, sig, pos)
                trade = record_sell(tkr, order_info)
                portfolio["pending"].pop(tkr, None)
                pnl = trade["pnl"] if trade else 0
                send_telegram(
                    f"{'✅' if pnl >= 0 else '❌'} *SELL executado — {tkr}*\n"
                    f"  {order_info['qty']:.4f} shares @ ${order_info['limit_price']:.2f}\n"
                    f"  PnL: *{pnl:+.2f}* ({trade['pnl_pct']:+.1f}%)\n"
                    f"  🏦 Order ID: `{order_info['order_id'][:16]}...`\n"
                    f"  💵 Caixa: ${portfolio['cash']:.2f}"
                )
        except Exception as e:
            log.error(f"Alpaca order error: {e}")
            send_telegram(f"❌ Erro ao executar ordem em *{tkr}*:\n`{str(e)[:200]}`")

def telegram_listener():
    global _last_update_id
    log.info("Telegram listener started")
    while True:
        try:
            resp = requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
                params={"offset": _last_update_id + 1, "timeout": 30}, timeout=40)
            for update in resp.json().get("result", []):
                _last_update_id = update["update_id"]

                # Handle inline button press
                cb = update.get("callback_query")
                if cb:
                    chat_id = str(cb.get("from", {}).get("id", ""))
                    data    = cb.get("data", "")
                    cb_id   = cb.get("id", "")
                    if chat_id == TELEGRAM_CHAT_ID and data:
                        log.info(f"Button pressed: {data!r}")
                        answer_callback(cb_id)
                        handle_command(data.replace("CONFIRM_", "CONFIRM ").replace("CANCEL_", "CANCEL "))
                    continue

                # Handle text message
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

    # ── Step 1: Sync real positions from Alpaca (always first) ──
    try:
        acc       = alpaca.get_account()
        positions = alpaca.get_all_positions()
        portfolio["cash"] = float(acc.cash)
        real_tickers = {pos.symbol for pos in positions}

        # Remove phantom positions
        for t in list(portfolio["positions"].keys()):
            if t not in real_tickers:
                log.warning(f"Removing phantom position: {t}")
                portfolio["positions"].pop(t, None)

        # Update real positions (qty + price from Alpaca)
        for pos in positions:
            ticker = pos.symbol
            price  = float(pos.current_price or pos.avg_entry_price)
            shares = float(pos.qty)
            entry  = float(pos.avg_entry_price)
            if ticker in portfolio["positions"]:
                portfolio["positions"][ticker]["shares"]        = round(shares, 6)
                portfolio["positions"][ticker]["current_price"] = round(price, 2)
            else:
                portfolio["positions"][ticker] = {
                    "shares": round(shares, 6), "entry_price": round(entry, 2),
                    "current_price": round(price, 2),
                    "stop_loss":  round(entry * 0.92, 2),
                    "take_profit": round(entry * 1.20, 2),
                    "opened_at": "sync", "allocated": round(entry * shares, 2),
                }
                log.info(f"New position from Alpaca: {ticker} {shares:.4f} @ ${entry:.2f}")

        pv = portfolio_value()
        if pv > portfolio["peak_value"]:
            portfolio["peak_value"] = pv
        log.info(f"Alpaca sync OK — cash=${portfolio['cash']:.2f} positions={list(portfolio['positions'].keys())}")
    except Exception as e:
        log.error(f"Alpaca sync failed: {e}")

    signals  = {}
    prices   = {}

    for ticker in TICKERS:
        try:
            log.info(f"Fetching {ticker}...")
            mkt = fetch_market_data(ticker)
            prices[ticker] = mkt["price"]
            log.info(f"  {ticker} ${mkt['price']} rsi={mkt['rsi']}")
            sig = analyze_ticker(ticker, mkt)
            signals[ticker] = sig
            log.info(f"  → {sig['signal']} conf={sig['confidence']}% risk={sig['riskScore']}")
            time.sleep(5)
        except Exception as e:
            log.error(f"Error {ticker}: {e}")
            time.sleep(5)

    if not signals:
        return

    closed_trades = check_stops(prices)
    actionable    = []

    for ticker, sig in signals.items():
        if sig["signal"] == "BUY" and ticker not in portfolio["positions"]:
            if sig["confidence"] >= MIN_CONFIDENCE and len(portfolio["positions"]) < MAX_POSITIONS and portfolio["cash"] >= 5:
                portfolio["pending"][ticker] = {
                    "signal":     sig,
                    "expires_at": time.time() + CONFIRM_TIMEOUT_S
                }
                actionable.append(sig)
        elif sig["signal"] == "SELL" and ticker in portfolio["positions"]:
            portfolio["pending"][ticker] = {
                "signal":     sig,
                "expires_at": time.time() + CONFIRM_TIMEOUT_S
            }
            actionable.append(sig)

    pv = portfolio_value()
    if pv > portfolio["peak_value"]:
        portfolio["peak_value"] = pv

    if not closed_trades and not actionable:
        log.info("No actionable signals")
        return

    report  = fmt_scan_report(actionable, closed_trades)
    buttons = build_signal_buttons(actionable) if actionable else None
    send_telegram(report, buttons)
    log.info(f"Report sent. Portfolio: ${pv:.2f}")


# ── Entry point ───────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Stock Signal Agent [Alpaca Paper + Confirmation Mode] started ===")
    log.info(f"Tickers: {TICKERS} | Capital: ${INITIAL_CAPITAL} | Interval: {SCAN_INTERVAL_MIN}min")

    threading.Thread(target=telegram_listener, daemon=True).start()
    threading.Thread(target=expiry_watchdog,   daemon=True).start()

    try:
        acc = get_alpaca_account()
        alpaca_status = f"✅ Alpaca Paper conectada\n💵 Buying power: ${acc['buying_power']:.2f}"
    except Exception as e:
        alpaca_status = f"⚠️ Alpaca: {e}"

    send_telegram(
        "🚀 *Stock Signal Agent — Alpaca Paper Trading*\n"
        f"Capital: *${INITIAL_CAPITAL:.2f}* | Tickers: {', '.join(TICKERS)}\n"
        f"{alpaca_status}\n\n"
        "📌 *Como funciona:*\n"
        "Ao receber um sinal, você tem *2 minutos* para confirmar.\n"
        "A ordem é executada automaticamente na Alpaca.\n\n"
        "✅ Confirmar: `CONFIRM NVDA`\n"
        "❌ Ignorar: `CANCEL NVDA`\n"
        "📊 Portfólio: `STATUS`"
    )

    # Sync existing Alpaca positions on every startup
    sync_msg = sync_with_alpaca()
    if sync_msg:
        send_telegram(sync_msg)

    while True:
        if is_market_hours():
            scan_all()
        else:
            log.info(f"Market closed ({datetime.now(MARKET_TZ).strftime('%a %H:%M')} ET) — sleeping")
        time.sleep(SCAN_INTERVAL_MIN * 60)

if __name__ == "__main__":
    main()
