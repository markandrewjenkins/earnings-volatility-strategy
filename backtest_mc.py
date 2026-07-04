"""
Earnings Volatility Strategy — 10-year Monte Carlo simulation

Free historical option-chain data does not exist, so a tick-level backtest of
this strategy is not possible here. Instead we Monte Carlo 10 years of trading
from per-trade return distributions CALIBRATED TO THE PUBLISHED RESEARCH
(Volatility Vibes, 72,500 earnings events, 2007+):

  Calendar (filtered model): mean +7.3%/trade, sigma 28%, max loss -105%
  Straddle (filtered model): mean +9.0%/trade, sigma 48%, fat left tail
  10y MC @ 10% Kelly calendar: ~66% win rate, mean max DD ~20%, Sharpe ~3.5

Each proposed improvement is modeled as an EXPLICIT ASSUMPTION (documented in
VARIANTS below) applied to those distributions — e.g. "execution filters save
1.5pp of slippage per trade but cut trade count 25%". The output shows how the
10-year outcome distribution shifts under each assumption. These are
assumption-driven simulations, not evidence the improvements work; they rank
which levers matter most and by roughly how much.

Writes sim_results.json (rendered by the dashboard's Simulation tab).
"""

import json
import math
import os
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sim_results.json")

RNG = np.random.default_rng(42)
N_PATHS = 4000
YEARS = 10
START_EQUITY = 10_000.0


def beta_params(mean, sd, lo, hi):
    """Beta distribution on [lo, hi] with the given mean/sd."""
    span = hi - lo
    m = (mean - lo) / span
    s = sd / span
    common = m * (1 - m) / (s * s) - 1
    a = m * common
    b = (1 - m) * common
    if a <= 0 or b <= 0:
        raise ValueError(f"infeasible beta: mean={mean}, sd={sd}, range=({lo},{hi})")
    return a, b


# ── Per-trade return distributions (in % of premium/debit) ───────────────

def calendar_returns(n, rng):
    """Mixture calibrated to mean +7.3, sigma 28, win rate 66%, floor -105.
    Wins: Beta on (0,150] mean 22 sd 20. Losses: Beta on [-105,0) mean -21.2 sd 17.2."""
    wins = rng.random(n) < 0.66
    out = np.empty(n)
    aw, bw = beta_params(22.0, 20.0, 0.0, 150.0)
    al, bl = beta_params(21.2, 17.2, 0.0, 105.0)
    out[wins] = rng.beta(aw, bw, wins.sum()) * 150.0
    out[~wins] = -rng.beta(al, bl, (~wins).sum()) * 105.0
    return out


def straddle_returns(n, rng):
    """Mixture calibrated to mean +9, sigma ~48, fat left tail (capped -500).
    Win rate 70% is an assumption (not published). Wins: Beta on (0,100]
    mean 20 sd 28. Losses: lognormal mean ~16.7 sd ~70, capped at 500."""
    wins = rng.random(n) < 0.70
    out = np.empty(n)
    aw, bw = beta_params(20.0, 28.0, 0.0, 100.0)
    out[wins] = rng.beta(aw, bw, wins.sum()) * 100.0
    nl = (~wins).sum()
    # tail sd parameter 120 with a -2000% cap reproduces the published
    # moments post-truncation (research's worst single loss was -9200%)
    sig2 = math.log(1 + (120.0 / 16.7) ** 2)
    mu = math.log(16.7) - sig2 / 2
    losses = np.minimum(rng.lognormal(mu, math.sqrt(sig2), nl), 2000.0)
    out[~wins] = -losses
    return out


# ── Improvement effects (THE ASSUMPTIONS) ────────────────────────────────

def trim_tail(returns, rng, worse_than=-60.0, keep_prob=0.5):
    """Improvements E/F (skip binary names, VIX regime filter): assume half of
    the deep losses (< -60%) are avoided; avoided trades are re-drawn from the
    mild-loss region (-5..-40%)."""
    deep = returns < worse_than
    resample = deep & (rng.random(len(returns)) > keep_prob)
    returns[resample] = -rng.uniform(5.0, 40.0, resample.sum())
    return returns


VARIANTS = [
    {
        "id": "baseline_calendar",
        "label": "Calendar — research filters (baseline)",
        "dist": "calendar", "frac": 0.06, "trades_yr": 120,
        "mean_shift": 0.0, "trim": False,
        "assumption": "Per-trade distribution calibrated to the published filtered-model stats "
                      "(mean +7.3%, σ28%, 66% win, floor −105%). ~120 trades/yr (one per "
                      "qualifying day, earnings seasons only). 6% of account per trade (10% Kelly).",
    },
    {
        "id": "baseline_straddle",
        "label": "Straddle — research filters (baseline)",
        "dist": "straddle", "frac": 0.02, "trades_yr": 120,
        "mean_shift": 0.0, "trim": False,
        "assumption": "Calibrated to mean +9%, σ48%, fat left tail capped −2000%. Win rate 70% "
                      "assumed (not published). 2% of account premium per trade (30% Kelly).",
    },
    {
        "id": "tier1_execution",
        "label": "Calendar + Tier 1 execution filters (A/B/C)",
        "dist": "calendar", "frac": 0.06, "trades_yr": 90,
        "mean_shift": 1.5, "trim": False,
        "assumption": "ATM spread ≤10%, price ≥$20, richness ≥1.15× — assume +1.5pp/trade saved "
                      "slippage & better setups, at the cost of 25% fewer trades (90/yr).",
    },
    {
        "id": "ranked_top1",
        "label": "Calendar + rank by decile score (D)",
        "dist": "calendar", "frac": 0.06, "trades_yr": 72,
        "mean_shift": 2.0, "trim": False,
        "assumption": "Only the best-ranked candidate per day (slope + IV/RV deciles are "
                      "monotonic in the research) — assume +2pp/trade, 40% fewer trades (72/yr).",
    },
    {
        "id": "tail_trimmed",
        "label": "Calendar + tail filters (E/F)",
        "dist": "calendar", "frac": 0.06, "trades_yr": 108,
        "mean_shift": -0.3, "trim": True,
        "assumption": "Skip binary-catalyst names + stand aside in VIX spikes — assume half of "
                      "sub-−60% losses avoided (re-drawn as mild losses), −0.3pp mean cost, "
                      "10% fewer trades (108/yr).",
    },
    {
        "id": "all_combined",
        "label": "Calendar + all improvements (A–F)",
        "dist": "calendar", "frac": 0.06, "trades_yr": 66,
        "mean_shift": 3.2, "trim": True,
        "assumption": "Execution filters + ranking + tail trims combined: +3.2pp/trade, deep-loss "
                      "halving, but only ~66 trades/yr after all filters stack.",
    },
]


def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return dd.min()


def longest_dd(equity):
    peak = np.maximum.accumulate(equity)
    at_high = equity >= peak
    longest = cur = 0
    for x in at_high:
        cur = 0 if x else cur + 1
        longest = max(longest, cur)
    return longest


def simulate(variant):
    n_tr = int(variant["trades_yr"] * YEARS)
    frac = variant["frac"]
    draw = calendar_returns if variant["dist"] == "calendar" else straddle_returns

    all_r = draw(N_PATHS * n_tr, RNG).reshape(N_PATHS, n_tr)
    if variant["trim"]:
        flat = all_r.ravel()
        trim_tail(flat, RNG)
        all_r = flat.reshape(N_PATHS, n_tr)
    all_r += variant["mean_shift"]

    # equity paths: account return per trade = frac × trade return (floor at
    # -100% of the allocated fraction × loss multiple for straddles)
    acct_r = np.maximum(frac * all_r / 100.0, -0.95)
    equity = START_EQUITY * np.cumprod(1.0 + acct_r, axis=1)
    equity = np.concatenate([np.full((N_PATHS, 1), START_EQUITY), equity], axis=1)

    terminal = equity[:, -1]
    cagr = (terminal / START_EQUITY) ** (1.0 / YEARS) - 1.0
    mdd = np.array([max_drawdown(e) for e in equity])
    ldd = np.array([longest_dd(e) for e in equity]) / variant["trades_yr"]  # years
    per_trade_mean = all_r.mean()
    per_trade_sd = all_r.std()
    win_rate = (all_r > 0).mean()
    sharpe = (acct_r.mean() / acct_r.std()) * math.sqrt(variant["trades_yr"]) if acct_r.std() > 0 else 0

    def pct(a, q):
        return float(np.percentile(a, q))

    # equity band for the chart: percentiles of equity over time (52 samples)
    idx = np.linspace(0, equity.shape[1] - 1, 53).astype(int)
    band = {
        "t_years": [round(i / variant["trades_yr"], 2) for i in idx],
        "p5":  [round(pct(equity[:, i], 5)) for i in idx],
        "p25": [round(pct(equity[:, i], 25)) for i in idx],
        "p50": [round(pct(equity[:, i], 50)) for i in idx],
        "p75": [round(pct(equity[:, i], 75)) for i in idx],
        "p95": [round(pct(equity[:, i], 95)) for i in idx],
    }

    # per-trade return histogram (for the distribution chart)
    hist_counts, hist_edges = np.histogram(all_r[:200].ravel(), bins=np.arange(-110, 160, 10))
    return {
        "id": variant["id"],
        "label": variant["label"],
        "assumption": variant["assumption"],
        "params": {"dist": variant["dist"], "frac": variant["frac"],
                   "trades_yr": variant["trades_yr"], "mean_shift": variant["mean_shift"],
                   "tail_trim": variant["trim"]},
        "per_trade": {"mean_pct": round(float(per_trade_mean), 2),
                      "sd_pct": round(float(per_trade_sd), 2),
                      "win_rate": round(float(win_rate), 4)},
        "terminal": {"median": round(pct(terminal, 50)), "mean": round(float(terminal.mean())),
                     "p5": round(pct(terminal, 5)), "p25": round(pct(terminal, 25)),
                     "p75": round(pct(terminal, 75)), "p95": round(pct(terminal, 95))},
        "cagr": {"median": round(pct(cagr, 50), 4), "p5": round(pct(cagr, 5), 4),
                 "p95": round(pct(cagr, 95), 4)},
        "max_dd": {"mean": round(float(mdd.mean()), 4), "median": round(pct(mdd, 50), 4),
                   "p95_worst": round(pct(mdd, 5), 4)},
        "longest_dd_years": {"median": round(pct(ldd, 50), 2), "p95": round(pct(ldd, 95), 2)},
        "sharpe": round(float(sharpe), 2),
        "prob": {"loss_after_10y": round(float((terminal < START_EQUITY).mean()), 4),
                 "halved_at_any_point": round(float((mdd <= -0.5).mean()), 4),
                 "x10_after_10y": round(float((terminal >= 10 * START_EQUITY).mean()), 4)},
        "equity_band": band,
        "ret_hist": {"edges": [int(x) for x in hist_edges.tolist()],
                     "counts": [int(c) for c in hist_counts.tolist()]},
    }


# ── Position sizing study ────────────────────────────────────────────────

def kelly_fraction(draw, mean_shift=0.0, n=300_000):
    """Numeric full Kelly: argmax_f E[log(1 + f*r)] on the actual (fat-tailed)
    distribution — NOT mean/variance, which understates tail risk."""
    rng = np.random.default_rng(5)
    r = (draw(n, rng) + mean_shift) / 100.0
    fs = np.linspace(0.01, 0.95, 95)
    growth = [float(np.mean(np.log1p(np.maximum(f * r, -0.999)))) for f in fs]
    i = int(np.argmax(growth))
    return round(float(fs[i]), 2)


def path_stats(acct_r, trades_yr):
    eq = np.cumprod(1.0 + acct_r, axis=1)
    eq = np.concatenate([np.ones((acct_r.shape[0], 1)), eq], axis=1)
    terminal = eq[:, -1]
    yrs = acct_r.shape[1] / trades_yr
    cagr = terminal ** (1.0 / yrs) - 1.0
    mdd = np.array([max_drawdown(e) for e in eq])
    return {
        "cagr_median": round(float(np.percentile(cagr, 50)), 4),
        "cagr_p5": round(float(np.percentile(cagr, 5)), 4),
        "cagr_p95": round(float(np.percentile(cagr, 95)), 4),
        "mdd_mean": round(float(mdd.mean()), 4),
        "mdd_p95_worst": round(float(np.percentile(mdd, 5)), 4),
        "p_halved": round(float((mdd <= -0.5).mean()), 4),
        "p_dd80": round(float((mdd <= -0.8).mean()), 4),
        "p_loss_10y": round(float((terminal < 1.0).mean()), 4),
    }


def sizing_study(trades_yr=120, years=10, n_paths=2500):
    """Sweep the per-trade fraction for the baseline calendar distribution,
    plus two-tier 'confidence-scaled' sizing (Tier 1 setups get more)."""
    rng = np.random.default_rng(99)
    n_tr = trades_yr * years
    full_kelly = kelly_fraction(calendar_returns)

    sweep = []
    for f in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30):
        r = calendar_returns(n_paths * n_tr, rng).reshape(n_paths, n_tr) / 100.0
        s = path_stats(np.maximum(f * r, -0.95), trades_yr)
        s["frac"] = f
        s["kelly_mult"] = round(f / full_kelly, 2)
        sweep.append(s)

    # Two-tier sizing: assume 40% of RECOMMENDED trades are Tier 1 with
    # +1.5pp better mean (the execution-filter assumption). Same ~6% average
    # exposure in both variants — only the allocation differs.
    t1_share = 0.40
    def tiered(f_rec, f_t1):
        r = calendar_returns(n_paths * n_tr, rng).reshape(n_paths, n_tr) / 100.0
        is_t1 = rng.random((n_paths, n_tr)) < t1_share
        r = np.where(is_t1, r + 0.015, r)
        f = np.where(is_t1, f_t1, f_rec)
        return path_stats(np.maximum(f * r, -0.95), trades_yr)

    flat = tiered(0.06, 0.06)
    conf = tiered(0.043, 0.086)       # avg exposure = 0.6*4.3 + 0.4*8.6 ≈ 6.0%
    conf_up = tiered(0.06, 0.12)      # avg ≈ 8.4% — sized up AND tilted

    return {
        "full_kelly_frac": full_kelly,
        "note": ("Full Kelly computed numerically on the calibrated fat-tailed "
                 "distribution. Sweep uses the baseline calendar distribution "
                 f"({trades_yr} trades/yr, {years}y, {n_paths} paths). Two-tier "
                 "variants assume 40% of trades are Tier 1 with +1.5pp mean "
                 "(the execution-filter assumption) — same distribution "
                 "otherwise."),
        "sweep": sweep,
        "two_tier": {
            "flat_6pct": flat,
            "confidence_scaled_same_exposure": conf,
            "confidence_scaled_sized_up": conf_up,
            "labels": {
                "flat_6pct": "Flat 6% on every trade",
                "confidence_scaled_same_exposure": "REC 4.3% / Tier 1 8.6% (same ~6% avg exposure)",
                "confidence_scaled_sized_up": "REC 6% / Tier 1 12% (~8.4% avg exposure)",
            },
        },
    }


def main():
    results = []
    for v in VARIANTS:
        r = simulate(v)
        results.append(r)
        print(f"{r['label']:52s} med.term=${r['terminal']['median']:>12,} "
              f"CAGR={r['cagr']['median']*100:5.1f}% win={r['per_trade']['win_rate']*100:4.1f}% "
              f"meanDD={r['max_dd']['mean']*100:5.1f}% Sharpe={r['sharpe']}")
    sizing = sizing_study()
    print(f"\nfull Kelly (numeric, fat-tailed): {sizing['full_kelly_frac']*100:.0f}% per trade")
    for s in sizing["sweep"]:
        print(f"  f={s['frac']*100:4.1f}%  medCAGR={s['cagr_median']*100:6.1f}%  "
              f"meanDD={s['mdd_mean']*100:5.1f}%  P(halve)={s['p_halved']*100:4.1f}%  "
              f"P(DD>80)={s['p_dd80']*100:4.1f}%")
    tt = sizing["two_tier"]
    for k in ("flat_6pct", "confidence_scaled_same_exposure", "confidence_scaled_sized_up"):
        s = tt[k]
        print(f"  {tt['labels'][k]:48s} medCAGR={s['cagr_median']*100:6.1f}%  "
              f"meanDD={s['mdd_mean']*100:5.1f}%  P(halve)={s['p_halved']*100:4.1f}%")

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_paths": N_PATHS, "years": YEARS, "start_equity": START_EQUITY,
        "sizing": sizing,
        "methodology": (
            "Monte Carlo, not an options-level backtest: free historical option chains do not "
            "exist, so per-trade return distributions are calibrated to the published Volatility "
            "Vibes research (72,500 events, 2007+). Improvement variants apply explicitly stated "
            "assumptions (mean shifts, trade-count cuts, tail trims) to the baseline distribution. "
            "Treat differences BETWEEN variants as the signal; absolute dollar outcomes inherit "
            "all the assumptions."),
        "variants": results,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    print(f"\nWrote {OUT}")

    # calibration check vs published targets
    chk = calendar_returns(200_000, np.random.default_rng(7))
    print(f"calendar calib: mean={chk.mean():.2f} (target 7.3)  sd={chk.std():.2f} (target 28)  "
          f"win={(chk>0).mean()*100:.1f}% (target 66)  min={chk.min():.0f} (target >= -105)")
    chks = straddle_returns(200_000, np.random.default_rng(7))
    print(f"straddle calib: mean={chks.mean():.2f} (target 9)  sd={chks.std():.2f} (target ~48)  "
          f"win={(chks>0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
