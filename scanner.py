"""
Earnings Volatility Strategy — scanner

Scans US stocks with upcoming earnings (today after-close and next trading
day pre-open = positions you would enter before today's close), pulls their
option chains, and computes the three edge filters from the Volatility Vibes
research (72,500 earnings events, 2007+):

  1. ts_slope_0_45  <= -0.00406   (IV term structure in backwardation)
  2. iv30 / rv30    >=  1.25      (implied rich vs Yang-Zhang realized vol)
  3. avg volume 30d >=  1,500,000 (liquidity / price-insensitive flow)

Tiers (same as the original calculator):
  RECOMMENDED  all three pass
  CONSIDER     slope passes + exactly one of the other two
  AVOID        slope fails, or fewer than two pass

Enhanced tier (our additions, see STRATEGY.md):
  TIER 1 = RECOMMENDED plus execution-quality and premium-richness checks
           (tight ATM spreads, price >= $20, expected move rich vs the
           stock's own historical earnings moves).

Writes scan_results.json for the dashboard. Never leaves a half-written
file: output is written atomically, and a total calendar failure keeps the
previous file intact.

DISCLAIMER: educational/research use only. Not investment advice.
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone, date

import numpy as np
import pandas as pd
import requests
from scipy.interpolate import interp1d

try:
    from zoneinfo import ZoneInfo
except ImportError:  # py<3.9
    ZoneInfo = None

import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "scan_results.json")
CAL_CACHE_PATH = os.path.join(HERE, "calendar_cache.json")

# ── Strategy constants (from the research) ──────────────────────────────
SLOPE_THRESHOLD = -0.00406
IVRV_THRESHOLD = 1.25
VOLUME_THRESHOLD = 1_500_000
CALENDAR_GAP_DAYS = 30          # front/back expiry gap for the calendar
KELLY = {
    # fraction of account per trade (see video: 10% Kelly calendar, 30% straddle)
    "calendar_frac": 0.06,      # 10% Kelly of 60%  -> 6% of account as debit
    "straddle_frac": 0.02,      # 30% Kelly of 6.5% -> 2% of account as premium
}
# Enhanced-filter thresholds (our additions — TIER 1 = RECOMMENDED plus ALL of
# these; every one is about turning the theoretical edge into a realized one)
MAX_ATM_SPREAD_PCT = 0.10       # ATM bid/ask spread <= 10% of mid
MIN_PRICE = 20.0                # avoid wide-relative-spread cheap stocks
RICHNESS_MIN = 1.15             # expected move >= 1.15x avg historical move
MIN_ATM_OI = 500                # ATM open interest (min of call/put side)
                                # — thin OI means bad fills at both ends

MIN_MARKET_CAP = 1_000_000_000  # only scan liquid-ish names (the 1.5M-share
                                # volume filter would reject most below this
                                # anyway; keeps API usage sane in peak season)

CAL_HORIZON_DAYS = 30           # how far ahead we watch the earnings calendar
NEAR_WINDOW_DAYS = 7            # full options analysis inside this window
WATCH_MAX = 40                  # watchlist names given light (history-only) analysis
CACHE_PATH = None               # set below (depends on HERE)
CACHE_MAX_AGE_H = 12            # re-fetch the far calendar this often

NASDAQ_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}


def now_et():
    if ZoneInfo:
        return datetime.now(ZoneInfo("America/New_York"))
    return datetime.utcnow() - timedelta(hours=5)


def next_trading_day(d: date) -> date:
    d = d + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


# ── Earnings calendar (Nasdaq public API) ───────────────────────────────

def fetch_calendar_day(day: date, session: requests.Session):
    url = f"https://api.nasdaq.com/api/calendar/earnings?date={day.isoformat()}"
    r = session.get(url, headers=NASDAQ_HEADERS, timeout=25)
    r.raise_for_status()
    payload = r.json()
    rows = (payload.get("data") or {}).get("rows") or []
    out = []
    for row in rows:
        sym = (row.get("symbol") or "").strip().upper()
        if not sym or not sym.isalpha():
            continue  # skip units/warrants/foreign dotted tickers
        mc_raw = (row.get("marketCap") or "").replace("$", "").replace(",", "").strip()
        try:
            mcap = float(mc_raw)
        except ValueError:
            mcap = None
        t = row.get("time") or ""
        when = {"time-after-hours": "AMC", "time-pre-market": "BMO"}.get(t, "TNS")
        out.append({
            "ticker": sym,
            "name": (row.get("name") or "").strip(),
            "date": day.isoformat(),
            "when": when,
            "market_cap": mcap,
            "eps_est": row.get("epsForecast") or None,
        })
    return out


def fetch_calendar_days(days, session=None, max_retries=3):
    session = session or requests.Session()
    events = []
    for day in days:
        last_err = None
        for attempt in range(max_retries):
            try:
                events.extend(fetch_calendar_day(day, session))
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(3 * (attempt + 1))
        if last_err is not None:
            raise RuntimeError(f"calendar fetch failed for {day}: {last_err}")
        time.sleep(0.5)
    return events


def fetch_calendar(today):
    """Near days (0..NEAR_WINDOW) are always fetched fresh. The far calendar
    (NEAR_WINDOW+1 .. CAL_HORIZON) is cached in calendar_cache.json and only
    re-fetched every CACHE_MAX_AGE_H hours — earnings dates that far out move
    slowly, and 30 Nasdaq requests per 15-minute run would invite blocks."""
    session = requests.Session()
    near_days = [today + timedelta(days=i) for i in range(0, NEAR_WINDOW_DAYS + 1)
                 if (today + timedelta(days=i)).weekday() < 5]
    events = fetch_calendar_days(near_days, session)

    far_days = [today + timedelta(days=i)
                for i in range(NEAR_WINDOW_DAYS + 1, CAL_HORIZON_DAYS + 1)
                if (today + timedelta(days=i)).weekday() < 5]
    cache = None
    if os.path.exists(CAL_CACHE_PATH):
        try:
            with open(CAL_CACHE_PATH, encoding="utf-8") as f:
                cache = json.load(f)
            fetched = datetime.fromisoformat(cache["fetched_utc"])
            age_h = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
            if age_h > CACHE_MAX_AGE_H or cache.get("horizon_start") != far_days[0].isoformat():
                cache = None
        except Exception:
            cache = None
    if cache is None:
        far_events = fetch_calendar_days(far_days, session)
        cache = {
            "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "horizon_start": far_days[0].isoformat() if far_days else None,
            "events": far_events,
        }
        try:
            write_atomic(cache, CAL_CACHE_PATH)
        except Exception:
            pass
    events.extend(cache["events"])
    return events


def yahoo_earnings_date(stock, nasdaq_date):
    """Cross-check Nasdaq's date against Yahoo's per-ticker earnings date.
    Nasdaq mixes confirmed dates with projections, and projected dates are
    often shifted (last year's date + ~91 days). Returns (yahoo_date_str,
    mismatch_bool) — mismatch when they differ by more than 1 day."""
    try:
        cal = stock.calendar
        eds = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if not eds:
            return None, None
        nd = datetime.strptime(nasdaq_date, "%Y-%m-%d").date()
        yds = []
        for x in eds:
            if hasattr(x, "date"):
                x = x.date() if isinstance(x, datetime) else x
            yds.append(x)
        nearest = min(yds, key=lambda y: abs((y - nd).days))
        return nearest.isoformat(), abs((nearest - nd).days) > 1
    except Exception:
        return None, None


# ── Options math (identical to the original calculator) ─────────────────

def filter_dates(dates):
    """Keep expirations up to and including the first one >= 45 days out."""
    today = datetime.today().date()
    cutoff = today + timedelta(days=45)
    sorted_dates = sorted(datetime.strptime(d, "%Y-%m-%d").date() for d in dates)
    arr = []
    for i, d in enumerate(sorted_dates):
        if d >= cutoff:
            arr = [x.strftime("%Y-%m-%d") for x in sorted_dates[:i + 1]]
            break
    if arr:
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr
    raise ValueError("No expiration 45+ days out")


def yang_zhang(price_data, window=30, trading_periods=252):
    log_ho = (price_data["High"] / price_data["Open"]).apply(np.log)
    log_lo = (price_data["Low"] / price_data["Open"]).apply(np.log)
    log_co = (price_data["Close"] / price_data["Open"]).apply(np.log)
    log_oc = (price_data["Open"] / price_data["Close"].shift(1)).apply(np.log)
    log_cc = (price_data["Close"] / price_data["Close"].shift(1)).apply(np.log)
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    close_vol = (log_cc ** 2).rolling(window=window).sum() * (1.0 / (window - 1.0))
    open_vol = (log_oc ** 2).rolling(window=window).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(window=window).sum() * (1.0 / (window - 1.0))
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * math.sqrt(trading_periods)
    return float(result.iloc[-1])


def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)
    idx = days.argsort()
    days, ivs = days[idx], ivs[idx]
    spline = interp1d(days, ivs, kind="linear", fill_value="extrapolate")

    def term(dte):
        if dte < days[0]:
            return float(ivs[0])
        if dte > days[-1]:
            return float(ivs[-1])
        return float(spline(dte))

    return term


def mid(bid, ask):
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return None
    if bid <= 0 and ask <= 0:
        return None
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return max(bid, ask)


# ── Historical earnings moves (for the richness filter) ─────────────────

def historical_earnings_moves(stock, history, earnings_date):
    """Avg abs close-to-close move over past earnings (up to 8)."""
    try:
        ed = stock.get_earnings_dates(limit=20)
        if ed is None or ed.empty:
            return None
        past = [d.date() for d in ed.index if d.date() < datetime.today().date()]
    except Exception:
        return None
    closes = history["Close"]
    dates = [d.date() for d in closes.index]
    moves = []
    for pdte in past[:8]:
        # earnings AMC on day D -> move shows close(D) -> close(D+1)
        # earnings BMO on day D -> move shows close(D-1) -> close(D)
        # We don't know past timing; measure the larger of the two candidate
        # gaps around the date, which captures the reaction day either way.
        try:
            i = next(k for k, x in enumerate(dates) if x >= pdte)
        except StopIteration:
            continue
        cands = []
        for j in (i, i + 1):
            if 0 < j < len(closes):
                prev, cur = float(closes.iloc[j - 1]), float(closes.iloc[j])
                if prev > 0:
                    cands.append(abs(cur / prev - 1.0))
        if cands:
            moves.append(max(cands) * 100.0)
    if not moves:
        return None
    return {"avg_abs_move_pct": round(float(np.mean(moves)), 2),
            "n": len(moves),
            "moves_pct": [round(m, 2) for m in moves]}


# ── Per-ticker analysis ──────────────────────────────────────────────────

def analyze_ticker(symbol, earnings_date):
    stock = yf.Ticker(symbol)
    exp_all = list(stock.options)
    if not exp_all:
        raise ValueError("no options")
    exp_dates = filter_dates(exp_all)

    history = stock.history(period="2y", auto_adjust=True)
    if history is None or len(history) < 40:
        raise ValueError("not enough price history")
    price = float(history["Close"].iloc[-1])

    today = datetime.today().date()
    atm_iv = {}
    chains = {}
    straddle = None
    atm_strike = None
    spread_pct = None
    front_exp = None
    atm_oi = None
    strike_width = None

    for i, exp in enumerate(exp_dates):
        chain = stock.option_chain(exp)
        calls, puts = chain.calls, chain.puts
        if calls.empty or puts.empty:
            continue
        chains[exp] = chain
        c_idx = (calls["strike"] - price).abs().idxmin()
        p_idx = (puts["strike"] - price).abs().idxmin()
        c_iv = float(calls.loc[c_idx, "impliedVolatility"])
        p_iv = float(puts.loc[p_idx, "impliedVolatility"])
        if not (np.isfinite(c_iv) and np.isfinite(p_iv)) or c_iv <= 0 or p_iv <= 0:
            continue  # Yahoo sometimes returns NaN/zero IV on illiquid expiries
        atm_iv[exp] = (c_iv + p_iv) / 2.0

        if straddle is None:  # first usable expiry = front month
            front_exp = exp
            atm_strike = float(calls.loc[c_idx, "strike"])
            try:
                atm_oi = int(min(float(calls.loc[c_idx, "openInterest"] or 0),
                                 float(puts.loc[p_idx, "openInterest"] or 0)))
            except Exception:
                atm_oi = None
            try:  # strike spacing around ATM (coarse strikes = off-model fills)
                ks = sorted(set(calls["strike"].astype(float)))
                ki = ks.index(atm_strike)
                gaps = [ks[j + 1] - ks[j] for j in range(max(0, ki - 1), min(len(ks) - 1, ki + 1))]
                strike_width = float(min(gaps)) if gaps else None
            except Exception:
                strike_width = None
            c_mid = mid(calls.loc[c_idx, "bid"], calls.loc[c_idx, "ask"])
            p_mid = mid(puts.loc[p_idx, "bid"], puts.loc[p_idx, "ask"])
            if c_mid and p_mid:
                straddle = c_mid + p_mid
                try:
                    c_spr = float(calls.loc[c_idx, "ask"]) - float(calls.loc[c_idx, "bid"])
                    p_spr = float(puts.loc[p_idx, "ask"]) - float(puts.loc[p_idx, "bid"])
                    if c_mid > 0 and p_mid > 0 and c_spr >= 0 and p_spr >= 0:
                        spread_pct = ((c_spr / (2 * c_mid)) + (p_spr / (2 * p_mid))) / 2.0
                except Exception:
                    spread_pct = None
        time.sleep(0.15)

    if not atm_iv:
        raise ValueError("no usable ATM IV")

    dtes, ivs, term_points = [], [], []
    for exp, iv in atm_iv.items():
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
        dtes.append(dte)
        ivs.append(iv)
        term_points.append({"exp": exp, "dte": dte, "iv": round(iv, 4)})
    term_points.sort(key=lambda x: x["dte"])

    term = build_term_structure(dtes, ivs)
    dte0 = min(dtes)
    ts_slope = (term(45) - term(dte0)) / (45 - dte0)
    rv30 = yang_zhang(history.tail(90))
    iv30 = term(30)
    ivrv = iv30 / rv30 if rv30 > 0 else None
    avg_vol = float(history["Volume"].rolling(30).mean().dropna().iloc[-1])
    exp_move_pct = (straddle / price * 100.0) if straddle else None

    # calendar construction: sell front ATM, buy the expiry nearest front+30d
    back_exp, cal_debit = None, None
    if front_exp:
        f_date = datetime.strptime(front_exp, "%Y-%m-%d").date()
        target = f_date + timedelta(days=CALENDAR_GAP_DAYS)
        cands = [e for e in exp_all
                 if datetime.strptime(e, "%Y-%m-%d").date() > f_date]
        if cands:
            back_exp = min(cands, key=lambda e: abs(
                (datetime.strptime(e, "%Y-%m-%d").date() - target).days))
            try:
                bchain = chains.get(back_exp) or stock.option_chain(back_exp)
                bcalls = bchain.calls
                b_idx = (bcalls["strike"] - atm_strike).abs().idxmin()
                b_mid = mid(bcalls.loc[b_idx, "bid"], bcalls.loc[b_idx, "ask"])
                fcalls = chains[front_exp].calls
                f_idx = (fcalls["strike"] - atm_strike).abs().idxmin()
                f_mid = mid(fcalls.loc[f_idx, "bid"], fcalls.loc[f_idx, "ask"])
                if b_mid and f_mid and b_mid > f_mid:
                    cal_debit = b_mid - f_mid
            except Exception:
                cal_debit = None

    hist = historical_earnings_moves(stock, history, earnings_date)
    y_date, mismatch = yahoo_earnings_date(stock, earnings_date)

    pass_slope = ts_slope <= SLOPE_THRESHOLD
    pass_ivrv = bool(ivrv and ivrv >= IVRV_THRESHOLD)
    pass_vol = avg_vol >= VOLUME_THRESHOLD
    if pass_slope and pass_ivrv and pass_vol:
        tier = "RECOMMENDED"
    elif pass_slope and (pass_ivrv != pass_vol):
        tier = "CONSIDER"
    else:
        tier = "AVOID"

    # Enhanced (our) filters — execution quality + premium richness
    richness = None
    if exp_move_pct and hist and hist["avg_abs_move_pct"] > 0:
        richness = exp_move_pct / hist["avg_abs_move_pct"]
    enh_checks = {
        "tight_spread": bool(spread_pct is not None and spread_pct <= MAX_ATM_SPREAD_PCT),
        "price_ok": price >= MIN_PRICE,
        "premium_rich": bool(richness is None or richness >= RICHNESS_MIN),
        "richness_known": richness is not None,
        "oi_ok": bool(atm_oi is not None and atm_oi >= MIN_ATM_OI),
        "date_confirmed": bool(mismatch is not True),  # None (no Yahoo data) tolerated
        "em_vs_width": bool(straddle and strike_width and straddle >= strike_width),
    }
    tier1 = (tier == "RECOMMENDED" and enh_checks["tight_spread"] and
             enh_checks["price_ok"] and enh_checks["premium_rich"] and
             enh_checks["oi_ok"] and enh_checks["date_confirmed"])

    return {
        "price": round(price, 2),
        "yahoo_date": y_date,
        "date_mismatch": mismatch,
        "front_exp": front_exp,
        "back_exp": back_exp,
        "atm_strike": atm_strike,
        "straddle_mid": round(straddle, 2) if straddle else None,
        "calendar_debit_est": round(cal_debit, 2) if cal_debit else None,
        "expected_move_pct": round(exp_move_pct, 2) if exp_move_pct else None,
        "ts_slope_0_45": round(float(ts_slope), 5),
        "iv30": round(float(iv30), 4),
        "rv30": round(float(rv30), 4),
        "iv30_rv30": round(float(ivrv), 3) if ivrv else None,
        "avg_volume30": int(avg_vol),
        "atm_spread_pct": round(float(spread_pct), 4) if spread_pct is not None else None,
        "atm_oi": atm_oi,
        "strike_width": strike_width,
        "term_structure": term_points,
        "hist_moves": hist,
        "richness": round(float(richness), 2) if richness else None,
        "pass_slope": bool(pass_slope),
        "pass_ivrv": bool(pass_ivrv),
        "pass_volume": bool(pass_vol),
        "tier": tier,
        "tier1": bool(tier1),
        "enhanced": enh_checks,
    }


# ── Watchlist (light) analysis — no option chains ───────────────────────

def analyze_watch_ticker(symbol, earnings_date):
    """History-only look at a name reporting 8–30 days out. The slope and
    IV/RV filters only form in the final days before the event, so here we
    grade the *persistent* criteria and the stock's earnings history."""
    stock = yf.Ticker(symbol)
    history = stock.history(period="2y", auto_adjust=True)
    if history is None or len(history) < 40:
        raise ValueError("not enough price history")
    price = float(history["Close"].iloc[-1])
    avg_vol = float(history["Volume"].rolling(30).mean().dropna().iloc[-1])
    hist = historical_earnings_moves(stock, history, earnings_date)
    y_date, mismatch = yahoo_earnings_date(stock, earnings_date)

    pass_vol = avg_vol >= VOLUME_THRESHOLD
    price_ok = price >= MIN_PRICE
    # Likelihood the name will rate RECOMMENDED on its earnings day, based on
    # what is observable now: volume is stable week to week (the strongest
    # persistent signal); price and options-liquidity proxy fill it out.
    # Slope/IV richness can only be confirmed in the final days.
    if pass_vol and price_ok:
        likelihood = "HIGH"
    elif avg_vol >= VOLUME_THRESHOLD * 0.5 and price_ok:
        likelihood = "MEDIUM"
    else:
        likelihood = "LOW"
    return {
        "price": round(price, 2),
        "avg_volume30": int(avg_vol),
        "pass_volume": bool(pass_vol),
        "price_ok": bool(price_ok),
        "hist_moves": hist,
        "yahoo_date": y_date,
        "date_mismatch": mismatch,
        "likelihood": likelihood,
    }


# ── Main scan ────────────────────────────────────────────────────────────

def run_scan(max_analyze=45, tickers_override=None):
    et = now_et()
    today = et.date()
    next_td = next_trading_day(today)

    if tickers_override:
        events = [{"ticker": t, "name": t, "date": today.isoformat(),
                   "when": "AMC", "market_cap": None, "eps_est": None}
                  for t in tickers_override]
    else:
        events = fetch_calendar(today)

    for ev in events:
        d = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        if (d == today and ev["when"] in ("AMC", "TNS")) or \
           (d == next_td and ev["when"] == "BMO"):
            ev["bucket"] = "now"       # enter before today's close
        elif d <= today:
            ev["bucket"] = "past"      # already reported (yesterday, or today BMO)
        elif (d - today).days <= NEAR_WINDOW_DAYS:
            ev["bucket"] = "week"      # full metrics, preview
        else:
            ev["bucket"] = "watch"     # 30-day watchlist, light metrics
        # keep old field name for anything that still reads it
        ev["trade_window"] = "today" if ev["bucket"] == "now" else "upcoming"

    def cap(ev):
        return ev["market_cap"] or 0

    scannable = [e for e in events if cap(e) >= MIN_MARKET_CAP or tickers_override]
    now_evs = sorted([e for e in scannable if e["bucket"] == "now"], key=cap, reverse=True)
    week_evs = sorted([e for e in scannable if e["bucket"] == "week"],
                      key=lambda e: (e["date"], -cap(e)))
    watch_evs = sorted([e for e in scannable if e["bucket"] == "watch"], key=cap, reverse=True)

    errors = []
    analyzed = []

    # Full options analysis: everything tradeable now, then this week, within budget
    for ev in (now_evs + week_evs)[:max_analyze]:
        sym = ev["ticker"]
        try:
            metrics = analyze_ticker(sym, ev["date"])
            analyzed.append({**ev, **metrics, "error": None})
            print(f"  {sym:6s} [{ev['bucket']:5s}] {metrics['tier']:12s} "
                  f"slope={metrics['ts_slope_0_45']} ivrv={metrics['iv30_rv30']} "
                  f"vol={metrics['avg_volume30']:,}")
        except Exception as e:
            errors.append({"ticker": sym, "error": str(e)})
            print(f"  {sym:6s} [{ev['bucket']:5s}] ERROR: {e}")
        time.sleep(0.6)

    # Light analysis for the 30-day watchlist (largest caps first)
    watchlist = []
    for ev in watch_evs[:WATCH_MAX]:
        sym = ev["ticker"]
        try:
            metrics = analyze_watch_ticker(sym, ev["date"])
            watchlist.append({**ev, **metrics, "error": None})
            print(f"  {sym:6s} [watch] {metrics['likelihood']:6s} "
                  f"vol={metrics['avg_volume30']:,}")
        except Exception as e:
            errors.append({"ticker": sym, "error": str(e)})
            print(f"  {sym:6s} [watch] ERROR: {e}")
        time.sleep(0.4)

    analyzed_syms = {a["ticker"] for a in analyzed} | {w["ticker"] for w in watchlist}
    skipped = [e for e in events
               if e["ticker"] not in analyzed_syms and e["bucket"] != "past"]

    counts = {
        "calendar_events": len(events),
        "analyzed": len(analyzed),
        "watchlist": len(watchlist),
        "recommended": sum(1 for a in analyzed if a["tier"] == "RECOMMENDED"),
        "consider": sum(1 for a in analyzed if a["tier"] == "CONSIDER"),
        "avoid": sum(1 for a in analyzed if a["tier"] == "AVOID"),
        "tier1": sum(1 for a in analyzed if a.get("tier1")),
        "watch_high": sum(1 for w in watchlist if w["likelihood"] == "HIGH"),
        "errors": len(errors),
    }

    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_et": et.isoformat(timespec="seconds"),
        "scan_date": today.isoformat(),
        "next_trading_day": next_td.isoformat(),
        "horizon_days": CAL_HORIZON_DAYS,
        "date_source": "Nasdaq earnings calendar API, cross-checked per ticker "
                       "against Yahoo Finance (date_mismatch flags a >1 day gap)",
        "thresholds": {
            "ts_slope_0_45": SLOPE_THRESHOLD,
            "iv30_rv30": IVRV_THRESHOLD,
            "avg_volume30": VOLUME_THRESHOLD,
            "max_atm_spread_pct": MAX_ATM_SPREAD_PCT,
            "min_price": MIN_PRICE,
            "richness_min": RICHNESS_MIN,
            "min_atm_oi": MIN_ATM_OI,
            "calendar_gap_days": CALENDAR_GAP_DAYS,
            "min_market_cap": MIN_MARKET_CAP,
        },
        "sizing": KELLY,
        "counts": counts,
        "events": analyzed,
        "watchlist": watchlist,
        "skipped": [{k: e[k] for k in ("ticker", "name", "date", "when",
                                       "market_cap", "bucket")}
                    for e in skipped][:300],
        "errors": errors,
    }
    return result


def sanitize(obj):
    """NaN/Inf are invalid JSON and would break the dashboard's JSON.parse."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_atomic(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sanitize(obj), f, indent=1, allow_nan=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=45, help="max tickers to analyze")
    ap.add_argument("--tickers", type=str, default=None,
                    help="comma-separated override (debug), e.g. AAPL,MSFT")
    args = ap.parse_args()

    override = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    try:
        result = run_scan(max_analyze=args.max, tickers_override=override)
    except Exception:
        traceback.print_exc()
        # keep previous scan_results.json intact; mark it stale if it exists
        if os.path.exists(OUT_PATH):
            try:
                with open(OUT_PATH, encoding="utf-8") as f:
                    prev = json.load(f)
                prev["stale"] = True
                prev["stale_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                write_atomic(prev, OUT_PATH)
                print("Scan failed — previous results kept (marked stale).")
                return 0
            except Exception:
                pass
        return 1

    write_atomic(result, OUT_PATH)
    c = result["counts"]
    print(f"\nWrote {OUT_PATH}")
    print(f"calendar={c['calendar_events']} analyzed={c['analyzed']} "
          f"recommended={c['recommended']} consider={c['consider']} "
          f"avoid={c['avoid']} tier1={c['tier1']} errors={c['errors']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
