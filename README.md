# Earnings Volatility Strategy — live scanner & dashboard

Live dashboard: **https://markandrewjenkins.github.io/earnings-volatility-strategy/**

Sells overpriced earnings implied volatility — long ATM calendar spreads
(preferred) or short ATM straddles — entered ~15 minutes before the close
preceding an earnings announcement and exited ~15 minutes after the next open.

Strategy basis: the Volatility Vibes earnings IV-crush research
(72,500 earnings events across 4,500 stocks, 2007–present).

## How it works

1. `scanner.py` pulls the Nasdaq earnings calendar (today → +4 days),
   keeps US stocks ≥ $1B market cap, and for each candidate pulls the
   Yahoo Finance option chain to compute:
   - `ts_slope_0_45` — ATM IV term-structure slope (front expiry → 45d)
   - `iv30_rv30` — 30d implied vol vs Yang-Zhang realized vol
   - `avg_volume30` — 30-day average share volume
   - expected move (front ATM straddle / price), calendar-spread legs and
     estimated debit, ATM spread quality, and the stock's average realized
     move over its last ~8 earnings
2. Events are rated **RECOMMENDED / CONSIDER / AVOID** using the original
   research thresholds, plus a stricter **TIER 1 ★** rating that adds
   execution-quality and premium-richness filters (see the dashboard's
   Improvements tab).
3. Results are written to `scan_results.json`; `index.html` renders it.

## Refresh schedule

`.github/workflows/scan.yml` runs the scan roughly every 15 minutes during
US market hours (GitHub cron is best-effort, so expect 15–30 min in
practice), plus a late-evening run to roll the trade date. Manual refresh:
Actions tab → "Earnings volatility scan" → Run workflow.

## Run locally

```bash
pip install -r requirements.txt
python scanner.py                 # full calendar scan
python scanner.py --tickers AAPL  # debug a single name
python -m http.server 8000        # then open http://localhost:8000/
```

## Disclaimer

Educational/research project — not investment advice. Options involve
substantial risk. Quotes are delayed; verify everything at your broker.
