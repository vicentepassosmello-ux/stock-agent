# Stock Signal Agent

Agente autônomo que monitora o mercado americano e envia alertas de BUY/SELL via Telegram.

## Arquitetura
- **Motor**: Claude claude-sonnet-4-20250514 com web search em tempo real
- **Deploy**: Render (worker service — roda 24/7)
- **Alertas**: Bot Telegram
- **Estratégias**: Técnica (RSI/MACD/MAs) + Fundamentalista + Momentum + Notícias

## Variáveis de ambiente (configurar no Render)

| Variável | Obrigatória | Descrição |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Chave da API Anthropic |
| `TELEGRAM_BOT_TOKEN` | ✅ | Token do bot (BotFather) |
| `TELEGRAM_CHAT_ID` | ✅ | Seu Chat ID no Telegram |
| `CAPITAL` | — | Capital em USD (default: 200) |
| `TICKERS` | — | Tickers separados por vírgula |
| `SCAN_INTERVAL_MIN` | — | Intervalo entre scans (default: 30) |

## Lógica de alertas
- Só envia alerta quando o sinal **muda** para BUY ou SELL (sem spam)
- HOLD não gera notificação
- Só roda em horário de mercado (9h30–16h ET, seg–sex)
- Fora do horário, o agente dorme e acorda no próximo ciclo

## Estrutura
```
stock-agent/
├── src/
│   └── agent.py       # Agente principal
├── requirements.txt
├── render.yaml        # Config do Render
└── README.md
```
