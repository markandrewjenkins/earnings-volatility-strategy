# Earnings Volatility Strategy — live scanner & dashboard

Live dashboard: **https://markandrewjenkins.github.io/earnings-volatility-strategy/**

Sells overpriced earnings implied volatility — long ATM calendar spreads
(preferred) or short ATM straddles — entered ~15 minutes before the close
preceding an earnings announcement and exited ~15 minutes after the next open.

Strategy basis: the Volatility Vibes earnings IV-crush research
(72,500 earnings events across 4,500 stocks, 2007–present).

## How it works

1. `scanner.py` pulls the Nasdaq earnings calendar (today → +30 days; the
   far half is cached ~12h in `calendar_cache.json`), keeps US stocks
   ≥ $1B market cap, and buckets events:
   - **now** — reports after today's close or before the next open
     (enter before today's close): full options analysis
   - **week** — within 7 days: full options analysis as a preview
   - **watch** — 8–30 days out: light history-only analysis with a
     HIGH/MEDIUM/LOW "likely to qualify" grade (volume & price persist
     week-to-week; the slope and IV/RV filters only form in the final days)

   Earnings dates come from Nasdaq and are **cross-checked against Yahoo's
   per-ticker earnings date** — a >1-day disagreement sets `date_mismatch`
   and shows a ⚠ on the dashboard (dates do shift; confirm before trading).

   For each fully-analyzed candidate the option chain provides:
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
4. `backtest_mc.py` (run manually, not on the schedule) generates
   `sim_results.json` — a 10-year, 4,000-path Monte Carlo calibrated to the
   published research moments, with each proposed improvement modeled as an
   explicitly stated assumption. Rendered on the dashboard's Simulation tab.

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
