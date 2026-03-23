import os
import json
import time
import logging
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
TELEGRAM_TOKEN   = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
CAPITAL          = float(os.environ.get("CAPITAL", "200"))

TICKERS = [t.strip() for t in os.environ.get(
    "TICKERS", "NVDA,TSLA,AMD,PLTR,COIN,SMCI,MSTR,RKLB"
).split(",") if t.strip()]

SCAN_INTERVAL_MIN = int(os.environ.get("SCAN_INTERVAL_MIN", "30"))

MARKET_TZ   = ZoneInfo("America/New_York")
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)

# Keeps last signal per ticker to avoid duplicate alerts
_last_signal: dict[str, str] = {}


def is_market_hours() -> bool:
    now = datetime.now(MARKET_TZ)
    if now.weekday() >= 5:          # Saturday / Sunday
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


def analyze_ticker(ticker: str) -> dict:
    system_prompt = f"""You are an aggressive stock trading signal agent.
Use web search to find the CURRENT price, today's change, recent news,
technicals (RSI, MACD, moving averages, support/resistance) and analyst ratings.
Capital to allocate: ${CAPITAL}. Profile: AGGRESSIVE.

Return ONLY a raw JSON object — no markdown backticks, no explanation.
{{
  "ticker": string,
  "companyName": string,
  "signal": "BUY" | "SELL" | "HOLD",
  "confidence": number 0-100,
  "currentPrice": number,
  "priceChange": number,
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
  "suggestedInvestment": number,
  "suggestedShares": number,
  "topCatalyst": string,
  "reasoning": string
}}"""

    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1200,
        "system": system_prompt,
        "messages": [{"role": "user", "content": f"Analyze stock ticker: {ticker}. Today's date: {datetime.now(MARKET_TZ).strftime('%Y-%m-%d')}. Use your knowledge of this company, recent trends, technical analysis principles, and fundamental data to generate the signal."}]
    }

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=payload,
        timeout=90
    )
    resp.raise_for_status()
    data = resp.json()

    text = "".join(b["text"] for b in data.get("content", []) if b.get("type") == "text")
    text = text.replace("```json", "").replace("```", "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in response for {ticker}")
    return json.loads(text[start:end + 1])


def format_message(d: dict) -> str:
    signal = d.get("signal", "HOLD")
    emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(signal, "⚪")
    trend_e = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}.get(d.get("trend",""), "")
    chg = d.get("priceChangePct", 0)
    chg_str = f"+{chg:.2f}%" if chg >= 0 else f"{chg:.2f}%"
    conf = d.get("confidence", 0)
    risk = d.get("riskScore", 5)
    risk_bar = "█" * risk + "░" * (10 - risk)

    msg = (
        f"{emoji} *{signal} — {d.get('ticker')}*\n"
        f"_{d.get('companyName', '')}_\n\n"
        f"💵 Preço: *${d.get('currentPrice', 0):.2f}* ({chg_str} hoje)\n"
        f"{trend_e} Tendência: {d.get('trend','').capitalize()}\n"
        f"📊 RSI: {d.get('rsi', 0):.1f}  ·  MACD: {d.get('macdSignal','').capitalize()}\n\n"
        f"🎯 Confiança: {conf}%\n"
        f"⚠️ Risco: {risk}/10  `{risk_bar}`\n\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"💼 *Dimensionamento*\n"
        f"  Investir: *${d.get('suggestedInvestment', 0):.2f}*\n"
        f"  Ações: {d.get('suggestedShares', 0):.2f}\n"
        f"  Entrada: ${d.get('currentPrice', 0):.2f}\n"
        f"  🛑 Stop: ${d.get('stopLoss', 0):.2f}\n"
        f"  ✅ Target: ${d.get('takeProfit', 0):.2f}\n"
        f"  R/R: {d.get('riskReward','—')}\n\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📰 {d.get('topCatalyst','')}\n\n"
        f"🤖 _{d.get('reasoning','')}_\n\n"
        f"⏰ {datetime.now(MARKET_TZ).strftime('%d/%m %H:%M')} ET"
    )
    return msg


def send_telegram(text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }, timeout=15)


def scan_all() -> None:
    log.info(f"Starting scan: {TICKERS}")
    for ticker in TICKERS:
        try:
            log.info(f"Analyzing {ticker}...")
            d = analyze_ticker(ticker)
            signal = d.get("signal", "HOLD")
            prev   = _last_signal.get(ticker)

            changed = signal != prev
            is_action = signal in ("BUY", "SELL")

            if is_action and changed:
                log.info(f"{ticker}: {prev} → {signal} (sending alert)")
                msg = format_message(d)
                send_telegram(msg)
                _last_signal[ticker] = signal
            elif is_action and not changed:
                log.info(f"{ticker}: {signal} (same as before, skipping)")
            else:
                log.info(f"{ticker}: HOLD (no alert)")
                _last_signal[ticker] = signal

            time.sleep(3)   # polite delay between tickers
        except Exception as e:
            log.error(f"Error analyzing {ticker}: {e}")
            time.sleep(5)


def main() -> None:
    log.info("=== Stock Signal Agent started ===")
    log.info(f"Tickers: {TICKERS}")
    log.info(f"Capital: ${CAPITAL} | Interval: {SCAN_INTERVAL_MIN}min")
    send_telegram(
        "🚀 *Stock Signal Agent iniciado!*\n"
        f"Monitorando: {', '.join(TICKERS)}\n"
        f"Capital: ${CAPITAL} · Perfil: Agressivo\n"
        f"Intervalo: {SCAN_INTERVAL_MIN} minutos (horário de mercado)"
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
