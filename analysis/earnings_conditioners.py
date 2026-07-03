"""
Earnings Volatility Strategy — conditioner study

Question: do market-level and history-level conditions change how BIG
earnings reactions are? The strategy is short the earnings move, so any
condition under which stocks move MORE than usual is a condition under
which the IV-crush edge gets eaten by gamma losses.

We can't backtest option P&L (no free historical chains), but we CAN measure
the realized-move half of the trade across thousands of past earnings events:

  move_z = |reaction-day log return| / trailing 21d daily return std
           (each stock's reaction normalized by its own recent volatility)

Bigger move_z = worse for the short-vol trade. We bucket move_z by:

  1. FOMC proximity      — announce/reaction day within ±1 trading day of a
                           Fed decision day
  2. VIX level           — <15 / 15-20 / 20-28 / >=28 (day before reaction)
  3. VIX term structure  — VIX/VIX3M <0.9 (contango) / 0.9-1.0 / >=1.0 (inverted)
  4. SPY trend           — above/below its 200-day MA (bull/bear)
  5. Beat history        — stock's EPS-beat rate over its prior events
  6. Sector cohort heat  — avg move_z of same-sector reporters in the prior
                           3 weeks of the same season

Outputs analysis/conditioner_study.json + a console table. For each bucket:
n, median move_z, mean move_z, and P(move_z > 2) ("blowthrough rate" — the
reactions most likely to overwhelm an expected-move short) with a bootstrap
95% CI on the median difference vs the baseline.

Universe: ~100 liquid optionable US large/mid caps across all GICS sectors.
Data: yfinance earnings dates (with EPS surprise) + daily prices, ^VIX,
^VIX3M, SPY. Roughly the last 4-6 years per name (Yahoo's limit).
"""

import json
import math
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "conditioner_study.json")

FOMC_DECISIONS = [
    # source: federalreserve.gov FOMC calendars (decision/statement days)
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
    "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
    "2026-09-16", "2026-10-28", "2026-12-09",
]

UNIVERSE = {
    # ticker: GICS-ish sector
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AMD": "Tech", "INTC": "Tech",
    "CRM": "Tech", "ORCL": "Tech", "ADBE": "Tech", "QCOM": "Tech", "AVGO": "Tech",
    "MU": "Tech", "TXN": "Tech", "AMAT": "Tech", "NOW": "Tech", "SNOW": "Tech",
    "CSCO": "Tech", "IBM": "Tech", "PLTR": "Tech", "PANW": "Tech",
    "GOOGL": "Comm", "META": "Comm", "NFLX": "Comm", "DIS": "Comm", "CMCSA": "Comm",
    "AMZN": "ConsDisc", "TSLA": "ConsDisc", "HD": "ConsDisc", "NKE": "ConsDisc",
    "MCD": "ConsDisc", "SBUX": "ConsDisc", "LOW": "ConsDisc", "TGT": "ConsDisc",
    "LULU": "ConsDisc", "CMG": "ConsDisc", "ABNB": "ConsDisc", "UBER": "ConsDisc",
    "CCL": "ConsDisc", "DKNG": "ConsDisc", "ETSY": "ConsDisc",
    "WMT": "Staples", "COST": "Staples", "PG": "Staples", "KO": "Staples",
    "PEP": "Staples", "PM": "Staples", "MO": "Staples", "CL": "Staples",
    "JPM": "Fins", "BAC": "Fins", "WFC": "Fins", "C": "Fins", "GS": "Fins",
    "MS": "Fins", "SCHW": "Fins", "AXP": "Fins", "BLK": "Fins", "V": "Fins",
    "MA": "Fins", "PYPL": "Fins", "COF": "Fins", "COIN": "Fins",
    "UNH": "Health", "JNJ": "Health", "PFE": "Health", "MRK": "Health",
    "ABBV": "Health", "LLY": "Health", "TMO": "Health", "ABT": "Health",
    "BMY": "Health", "AMGN": "Health", "GILD": "Health", "CVS": "Health",
    "ISRG": "Health", "MRNA": "Health",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "OXY": "Energy", "DVN": "Energy", "HAL": "Energy",
    "BA": "Indus", "CAT": "Indus", "DE": "Indus", "UPS": "Indus", "FDX": "Indus",
    "GE": "Indus", "HON": "Indus", "LMT": "Indus", "RTX": "Indus", "UNP": "Indus",
    "DAL": "Indus", "UAL": "Indus",
    "LIN": "Mater", "FCX": "Mater", "NEM": "Mater", "DOW": "Mater",
    "PLD": "REIT", "AMT": "REIT", "NEE": "Util", "DUK": "Util",
}


def fetch_market():
    vix = yf.Ticker("^VIX").history(period="7y")["Close"]
    vix3m = yf.Ticker("^VIX3M").history(period="7y")["Close"]
    spy = yf.Ticker("SPY").history(period="8y")["Close"]
    spy200 = spy.rolling(200).mean()
    return (vix.tz_localize(None), vix3m.tz_localize(None),
            spy.tz_localize(None), spy200.tz_localize(None))


def asof(series, d):
    """Last value at or before date d."""
    s = series.loc[:pd.Timestamp(d)]
    return float(s.iloc[-1]) if len(s) else None


def collect_events():
    vix, vix3m, spy, spy200 = fetch_market()
    fomc = set(pd.Timestamp(d).date() for d in FOMC_DECISIONS)
    fomc_pm1 = set()
    for d in fomc:
        for k in (-1, 0, 1):
            fomc_pm1.add(d + timedelta(days=k))

    events = []
    fails = []
    for i, (sym, sector) in enumerate(UNIVERSE.items()):
        try:
            tk = yf.Ticker(sym)
            ed = tk.get_earnings_dates(limit=24)
            hist = tk.history(period="7y", auto_adjust=True)
            if ed is None or ed.empty or len(hist) < 300:
                fails.append(sym)
                continue
            closes = hist["Close"].tz_localize(None)
            rets = np.log(closes / closes.shift(1))
            sigma21 = rets.rolling(21).std()
            dates = list(closes.index)

            today = datetime.today().date()
            prior_events = []  # (date, beat) chronological
            rows = []
            for ts in sorted(ed.index):
                ann_date = ts.date() if hasattr(ts, "date") else ts
                if ann_date >= today:
                    continue
                is_amc = ts.hour >= 12
                # reaction day: next trading day for AMC, same day for BMO
                after = [d for d in dates if d.date() > ann_date] if is_amc else \
                        [d for d in dates if d.date() >= ann_date]
                if not after:
                    continue
                rd = after[0]
                ri = dates.index(rd)
                if ri < 25 or ri >= len(dates):
                    continue
                move = float(rets.iloc[ri])
                sig = float(sigma21.iloc[ri - 3]) if not math.isnan(sigma21.iloc[ri - 3]) else None
                if not sig or sig <= 0:
                    continue
                move_z = abs(move) / sig

                surprise = None
                try:
                    s = ed.loc[ts].get("Surprise(%)")
                    if s is not None and not (isinstance(s, float) and math.isnan(s)):
                        surprise = float(s)
                except Exception:
                    pass

                v = asof(vix, rd - pd.Timedelta(days=1))
                v3 = asof(vix3m, rd - pd.Timedelta(days=1))
                sp = asof(spy, rd - pd.Timedelta(days=1))
                sp200 = asof(spy200, rd - pd.Timedelta(days=1))

                beat_rate = None
                prior = [b for (pd_, b) in prior_events if b is not None]
                if len(prior) >= 4:
                    beat_rate = sum(prior) / len(prior)

                rows.append({
                    "ticker": sym, "sector": sector,
                    "announce": ann_date.isoformat(), "reaction": rd.date().isoformat(),
                    "amc": is_amc, "move_z": round(move_z, 3),
                    "abs_move_pct": round(abs(math.expm1(move)) * 100, 2),
                    "surprise_pct": surprise,
                    "vix": v, "vix_ratio": (v / v3) if (v and v3) else None,
                    "spy_bull": bool(sp and sp200 and sp >= sp200),
                    "fomc_window": (ann_date in fomc_pm1) or (rd.date() in fomc_pm1),
                    "beat_rate_prior": round(beat_rate, 3) if beat_rate is not None else None,
                })
                prior_events.append((ann_date, None if surprise is None else surprise > 0))
            events.extend(rows)
            print(f"[{i+1:3d}/{len(UNIVERSE)}] {sym:6s} {len(rows):3d} events")
        except Exception as e:
            fails.append(sym)
            print(f"[{i+1:3d}/{len(UNIVERSE)}] {sym:6s} FAILED: {e}")
        time.sleep(0.35)
    return events, fails


def add_cohort_heat(events):
    """Avg move_z of same-sector reporters in the 21 days before each event."""
    by_sector = {}
    for e in events:
        by_sector.setdefault(e["sector"], []).append(e)
    for sec, evs in by_sector.items():
        evs.sort(key=lambda x: x["reaction"])
    for e in events:
        rd = datetime.strptime(e["reaction"], "%Y-%m-%d").date()
        lo, hi = rd - timedelta(days=21), rd - timedelta(days=2)
        peers = [x["move_z"] for x in by_sector[e["sector"]]
                 if x is not e and lo <= datetime.strptime(x["reaction"], "%Y-%m-%d").date() <= hi]
        e["cohort_n"] = len(peers)
        e["cohort_heat"] = round(float(np.mean(peers)), 3) if len(peers) >= 3 else None


def boot_median_diff(a, b, n=3000, rng=None):
    """Bootstrap 95% CI for median(a) - median(b)."""
    rng = rng or np.random.default_rng(11)
    a, b = np.array(a), np.array(b)
    diffs = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b)))
             for _ in range(n)]
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def bucketize(events):
    z = [e["move_z"] for e in events]
    baseline = {
        "n": len(z), "median_z": round(float(np.median(z)), 3),
        "mean_z": round(float(np.mean(z)), 3),
        "p_blowthrough": round(float(np.mean(np.array(z) > 2)), 4),
    }

    def stats(subset, complement):
        zs = [e["move_z"] for e in subset]
        zc = [e["move_z"] for e in complement]
        if len(zs) < 30:
            return None
        lo, hi = boot_median_diff(zs, zc)
        return {
            "n": len(zs),
            "median_z": round(float(np.median(zs)), 3),
            "mean_z": round(float(np.mean(zs)), 3),
            "p_blowthrough": round(float(np.mean(np.array(zs) > 2)), 4),
            "median_diff_vs_rest": round(float(np.median(zs) - np.median(zc)), 3),
            "diff_ci95": [round(lo, 3), round(hi, 3)],
            "significant": bool(lo > 0 or hi < 0),
        }

    out = {"baseline": baseline, "conditioners": {}}

    def run(name, buckets):
        res = {}
        for label, pred in buckets:
            sub = [e for e in events if pred(e)]
            comp = [e for e in events if not pred(e)]
            s = stats(sub, comp)
            if s:
                res[label] = s
        out["conditioners"][name] = res

    run("fomc_window", [
        ("within +/-1td of FOMC decision", lambda e: e["fomc_window"]),
        ("no FOMC nearby", lambda e: not e["fomc_window"]),
    ])
    run("vix_level", [
        ("VIX < 15", lambda e: e["vix"] is not None and e["vix"] < 15),
        ("VIX 15-20", lambda e: e["vix"] is not None and 15 <= e["vix"] < 20),
        ("VIX 20-28", lambda e: e["vix"] is not None and 20 <= e["vix"] < 28),
        ("VIX >= 28", lambda e: e["vix"] is not None and e["vix"] >= 28),
    ])
    run("vix_term_structure", [
        ("contango (VIX/VIX3M < 0.9)", lambda e: e["vix_ratio"] is not None and e["vix_ratio"] < 0.9),
        ("flat (0.9-1.0)", lambda e: e["vix_ratio"] is not None and 0.9 <= e["vix_ratio"] < 1.0),
        ("inverted (>= 1.0)", lambda e: e["vix_ratio"] is not None and e["vix_ratio"] >= 1.0),
    ])
    run("spy_trend", [
        ("SPY above 200dma (bull)", lambda e: e["spy_bull"]),
        ("SPY below 200dma (bear)", lambda e: not e["spy_bull"]),
    ])
    run("beat_history", [
        ("serial beater (>=75% prior beats)", lambda e: e["beat_rate_prior"] is not None and e["beat_rate_prior"] >= 0.75),
        ("mixed (50-75%)", lambda e: e["beat_rate_prior"] is not None and 0.5 <= e["beat_rate_prior"] < 0.75),
        ("inconsistent (<50%)", lambda e: e["beat_rate_prior"] is not None and e["beat_rate_prior"] < 0.5),
    ])
    run("sector_cohort_heat", [
        ("quiet cohort (avg peer z < 1.0)", lambda e: e["cohort_heat"] is not None and e["cohort_heat"] < 1.0),
        ("normal cohort (1.0-1.5)", lambda e: e["cohort_heat"] is not None and 1.0 <= e["cohort_heat"] < 1.5),
        ("hot cohort (>= 1.5)", lambda e: e["cohort_heat"] is not None and e["cohort_heat"] >= 1.5),
    ])
    run("sector", [(s, (lambda e, s=s: e["sector"] == s)) for s in
                   sorted(set(e["sector"] for e in events))])
    return out


def main():
    events, fails = collect_events()
    print(f"\ncollected {len(events)} earnings events "
          f"({len(UNIVERSE) - len(fails)}/{len(UNIVERSE)} tickers)")
    if len(events) < 500:
        print("too few events — aborting without writing")
        return 1
    add_cohort_heat(events)
    result = bucketize(events)
    result["generated_utc"] = datetime.utcnow().isoformat(timespec="seconds")
    result["universe_size"] = len(UNIVERSE) - len(fails)
    result["failed_tickers"] = fails
    result["methodology"] = (
        "move_z = |reaction-day log return| / trailing 21d daily std (per stock). "
        "Bigger move_z is worse for short-earnings-vol. p_blowthrough = share of "
        "events with move_z > 2. diff_ci95 = bootstrap 95% CI of the median "
        "difference vs all other events; 'significant' = CI excludes zero. "
        "Measures the realized-move half of the trade only — IV-crush behavior "
        "(especially back-month) is NOT captured.")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, allow_nan=False)
    print(f"wrote {OUT}\n")

    b = result["baseline"]
    print(f"BASELINE: n={b['n']}  median_z={b['median_z']}  P(z>2)={b['p_blowthrough']*100:.1f}%\n")
    for cname, buckets in result["conditioners"].items():
        print(f"— {cname} —")
        for label, s in buckets.items():
            sig = " ***" if s.get("significant") else ""
            print(f"  {label:38s} n={s['n']:5d}  med={s['median_z']:5.2f}  "
                  f"P(z>2)={s['p_blowthrough']*100:4.1f}%  "
                  f"dmed={s['median_diff_vs_rest']:+.2f} CI[{s['diff_ci95'][0]:+.2f},{s['diff_ci95'][1]:+.2f}]{sig}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
