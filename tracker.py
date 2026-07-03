"""
Earnings Volatility Strategy — automated paper-trade tracker

Runs after scanner.py on the same schedule and maintains trades_log.json:

  ENTRY  — during the last ~40 minutes of the US session (15:20–16:00 ET),
           every RECOMMENDED candidate in today's trade window gets a
           hypothetical ATM call-calendar opened at current (delayed) mid
           prices. Tier 1 status is recorded on each trade so the dashboard
           can compare "all RECOMMENDED" vs "Tier 1 only" performance.
           TNS (time-not-supplied) reporters are NOT tracked — they may have
           already reported, which would poison the sample.

  EXIT   — the first scan at/after 09:40 ET on the reaction day (the first
           session after the announcement) closes the trade at current mids:
           the same "jump" exit the research prescribes.

  P&L    — % return on debit per trade, plus an idealized fractional account:
           $10,000 start, 6% of current equity allocated per trade (matching
           the Monte Carlo sizing; fractional contracts, no rounding). Both
           per-trade and compounded equity are recorded.

Quotes are ~15-minute delayed Yahoo mids — good enough to grade the strategy,
not good enough to grade your fills. Compare against real executions.

DISCLAIMER: educational/research use only. Not investment advice.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone, date

import yfinance as yf

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN_PATH = os.path.join(HERE, "scan_results.json")
LOG_PATH = os.path.join(HERE, "trades_log.json")

START_EQUITY = 10_000.0
SIZING_FRAC = 0.06          # fraction of current equity per trade (10% Kelly)
ENTRY_START = (15, 20)      # ET
ENTRY_END = (16, 0)
EXIT_AFTER = (9, 40)        # ET, on/after the reaction day


def now_et():
    if ZoneInfo:
        return datetime.now(ZoneInfo("America/New_York"))
    return datetime.utcnow() - timedelta(hours=5)


def next_trading_day(d: date) -> date:
    d = d + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def mid(bid, ask):
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return None
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if bid > 0 or ask > 0:
        return max(bid, ask)
    return None


def leg_mids(ticker, front_exp, back_exp, strike):
    """Fresh call mids for both calendar legs + spot. Returns dict or None."""
    stock = yf.Ticker(ticker)
    try:
        hist = stock.history(period="1d")
        spot = float(hist["Close"].iloc[-1]) if len(hist) else None
    except Exception:
        spot = None
    out = {"spot": spot}
    for label, exp in (("front", front_exp), ("back", back_exp)):
        try:
            calls = stock.option_chain(exp).calls
            idx = (calls["strike"].astype(float) - strike).abs().idxmin()
            row = calls.loc[idx]
            if abs(float(row["strike"]) - strike) > 1e-6:
                return None  # exact strike vanished (splits etc.) — skip
            m = mid(row["bid"], row["ask"])
            if m is None:
                last = float(row.get("lastPrice") or 0)
                m = last if last > 0 else None
            if m is None:
                return None
            out[label] = round(m, 3)
        except Exception:
            return None
        time.sleep(0.2)
    return out


def load_log():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "account": {"start": START_EQUITY, "equity": START_EQUITY,
                    "sizing_frac": SIZING_FRAC, "mode": "fractional (idealized)"},
        "open": [],
        "closed": [],
    }


def save_log(log):
    log["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp = LOG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1, allow_nan=False)
    os.replace(tmp, LOG_PATH)


def reaction_date(ev_date: str, when: str) -> str:
    d = datetime.strptime(ev_date, "%Y-%m-%d").date()
    if when == "BMO":
        return d.isoformat()                 # reports pre-open -> exit same day
    return next_trading_day(d).isoformat()   # AMC -> exit next session


def in_entry_window(et, force=False):
    if force:
        return True
    if et.weekday() >= 5:
        return False
    t = (et.hour, et.minute)
    return ENTRY_START <= t < ENTRY_END


def exit_eligible(trade, et, force=False):
    if force:
        return True
    rd = datetime.strptime(trade["reaction_date"], "%Y-%m-%d").date()
    if et.date() < rd:
        return False
    if et.date() == rd:
        return (et.hour, et.minute) >= EXIT_AFTER and et.weekday() < 5
    return True  # past the reaction day (weekend runs, missed scans) — close ASAP


def try_entries(log, scan, et, force=False):
    if not in_entry_window(et, force):
        print(f"entry window closed (ET {et:%H:%M}) — skipping entries")
        return 0
    known = {t["id"] for t in log["open"]} | {t["id"] for t in log["closed"]}
    opened = 0
    for ev in scan.get("events", []):
        if ev.get("bucket") != "now" or ev.get("tier") != "RECOMMENDED":
            continue
        if ev.get("when") == "TNS":
            continue  # may already have reported — untrackable cleanly
        if not (ev.get("front_exp") and ev.get("back_exp") and ev.get("atm_strike")):
            continue
        tid = f"{ev['ticker']}-{ev['date'].replace('-', '')}"
        if tid in known:
            continue
        legs = leg_mids(ev["ticker"], ev["front_exp"], ev["back_exp"], float(ev["atm_strike"]))
        if not legs or legs.get("front") is None or legs.get("back") is None:
            print(f"  {ev['ticker']}: no fresh leg quotes — skipped")
            continue
        debit = round(legs["back"] - legs["front"], 3)
        if debit <= 0:
            print(f"  {ev['ticker']}: non-positive calendar debit ({debit}) — skipped")
            continue
        log["open"].append({
            "id": tid,
            "ticker": ev["ticker"],
            "name": ev.get("name"),
            "earnings_date": ev["date"],
            "when": ev["when"],
            "tier": ev["tier"],
            "tier1": bool(ev.get("tier1")),
            "structure": "call calendar",
            "strike": float(ev["atm_strike"]),
            "front_exp": ev["front_exp"],
            "back_exp": ev["back_exp"],
            "reaction_date": reaction_date(ev["date"], ev["when"]),
            "entry": {
                "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "et": et.isoformat(timespec="minutes"),
                "stock_price": legs["spot"],
                "front_mid": legs["front"],
                "back_mid": legs["back"],
                "debit": debit,
                "expected_move_pct": ev.get("expected_move_pct"),
                "iv30_rv30": ev.get("iv30_rv30"),
                "ts_slope_0_45": ev.get("ts_slope_0_45"),
            },
        })
        opened += 1
        print(f"  OPENED {tid}: {ev['front_exp']}/{ev['back_exp']} @ {ev['atm_strike']} "
              f"debit {debit} (tier1={bool(ev.get('tier1'))})")
    return opened


def try_exits(log, et, force=False):
    closed_n = 0
    still_open = []
    for tr in log["open"]:
        if not exit_eligible(tr, et, force):
            still_open.append(tr)
            continue
        legs = leg_mids(tr["ticker"], tr["front_exp"], tr["back_exp"], tr["strike"])
        if not legs or legs.get("front") is None or legs.get("back") is None:
            print(f"  {tr['id']}: no exit quotes yet — will retry next run")
            still_open.append(tr)
            continue
        value = round(legs["back"] - legs["front"], 3)
        debit = tr["entry"]["debit"]
        pnl_pct = (value - debit) / debit * 100.0
        ep = tr["entry"].get("stock_price")
        actual_move = (abs(legs["spot"] / ep - 1.0) * 100.0) if (legs.get("spot") and ep) else None

        eq_before = log["account"]["equity"]
        alloc = eq_before * SIZING_FRAC
        pnl_usd = alloc * pnl_pct / 100.0
        log["account"]["equity"] = round(eq_before + pnl_usd, 2)

        tr["exit"] = {
            "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "et": et.isoformat(timespec="minutes"),
            "stock_price": legs.get("spot"),
            "front_mid": legs["front"],
            "back_mid": legs["back"],
            "value": value,
            "actual_move_pct": round(actual_move, 2) if actual_move is not None else None,
        }
        tr["result"] = {
            "pnl_pct": round(pnl_pct, 2),
            "win": pnl_pct > 0,
            "alloc_usd": round(alloc, 2),
            "pnl_usd": round(pnl_usd, 2),
            "equity_after": log["account"]["equity"],
        }
        log["closed"].append(tr)
        closed_n += 1
        print(f"  CLOSED {tr['id']}: value {value} vs debit {debit} -> {pnl_pct:+.1f}% "
              f"(${pnl_usd:+,.2f}, equity ${log['account']['equity']:,.2f})")
    log["open"] = still_open
    return closed_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-entry", action="store_true", help="ignore the entry time window (testing)")
    ap.add_argument("--force-exit", action="store_true", help="close all open trades now (testing)")
    ap.add_argument("--dry", action="store_true", help="don't write trades_log.json")
    args = ap.parse_args()

    if not os.path.exists(SCAN_PATH):
        print("no scan_results.json — run scanner.py first")
        return 1
    with open(SCAN_PATH, encoding="utf-8") as f:
        scan = json.load(f)

    et = now_et()
    log = load_log()
    closed = try_exits(log, et, force=args.force_exit)
    opened = try_entries(log, scan, et, force=args.force_entry)

    if not args.dry:
        save_log(log)
    print(f"tracker: opened={opened} closed={closed} open_now={len(log['open'])} "
          f"closed_total={len(log['closed'])} equity=${log['account']['equity']:,.2f}"
          f"{' (dry run)' if args.dry else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
