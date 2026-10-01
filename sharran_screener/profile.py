"""Per-business profile: what the company does, how it earns, and its
balance sheet / earnings track record.

Powers the 'business card' view - Sharran's rule #1 is only investing in
companies you understand, so every shortlisted name gets a one-page
infographic for that judgment call.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import yfinance as yf

from .market_data import SESSION, _fresh_session, _load_cache, _save_cache

log = logging.getLogger(__name__)


def _row(df: pd.DataFrame | None, names: list[str]) -> pd.Series | None:
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = df.loc[n]
            if isinstance(s, pd.DataFrame):  # duplicate row labels
                s = s.iloc[0]
            s = s.dropna()
            if len(s):
                return s
    return None


def _latest(df: pd.DataFrame | None, names: list[str]) -> float:
    s = _row(df, names)
    if s is None:
        return np.nan
    try:
        return float(s.iloc[0])
    except (TypeError, ValueError):
        return np.nan


def _yearly(df: pd.DataFrame | None, names: list[str], n: int = 4) -> dict:
    """Latest n fiscal years of a line item, {2023: value, ...} newest last."""
    s = _row(df, names)
    if s is None:
        return {}
    out = {}
    for ts, v in list(s.items())[:n]:
        try:
            out[pd.Timestamp(ts).year] = float(v)
        except (TypeError, ValueError):
            continue
    return dict(sorted(out.items()))


def fetch_business_profile(sym: str, force: bool = False) -> dict | None:
    """Annual financials + company descriptor for one ticker.
    Cached per calendar day; returns None if Yahoo refuses us."""
    cached = _load_cache("profiles") or {}
    if not force and cached.get(sym):
        return cached[sym]

    prof = None
    for attempt in range(3):
        sess = SESSION if attempt == 0 else _fresh_session()
        try:
            t = yf.Ticker(sym, session=sess)
            info = t.get_info() or {}
            inc = t.income_stmt
            bs = t.balance_sheet
            cf = t.cashflow

            assets = _latest(bs, ["Total Assets"])
            debt = _latest(bs, ["Total Debt", "Total Debt And Capital Lease Obligation"])
            cash = _latest(bs, [
                "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments"])
            equity = _latest(bs, ["Stockholders Equity", "Common Stock Equity",
                                  "Total Equity Gross Minority Interest"])
            cur_a = _latest(bs, ["Current Assets"])
            cur_l = _latest(bs, ["Current Liabilities"])

            prof = {
                "name": info.get("shortName") or info.get("longName") or sym,
                "summary": info.get("longBusinessSummary") or "",
                "sector": info.get("sector") or "",
                "industry": info.get("industry") or "",
                "website": info.get("website") or "",
                "employees": info.get("fullTimeEmployees"),
                "hq": ", ".join(x for x in [info.get("city"),
                                           info.get("country")] if x),
                "revenue_yr": _yearly(inc, ["Total Revenue", "Operating Revenue"]),
                "net_income_yr": _yearly(inc, ["Net Income",
                                               "Net Income Common Stockholders"]),
                "eps_yr": _yearly(inc, ["Diluted EPS", "Basic EPS"]),
                "fcf_yr": _yearly(cf, ["Free Cash Flow"]),
                "ocf_yr": _yearly(cf, ["Operating Cash Flow"]),
                "assets": assets,
                "debt": debt if not np.isnan(debt) else 0.0,
                "cash": cash,
                "equity": equity,
                "current_ratio": (cur_a / cur_l
                                  if cur_a > 0 and cur_l > 0 else np.nan),
                "debt_to_assets": (debt / assets
                                   if assets > 0 and not np.isnan(debt)
                                   else np.nan),
                "debt_to_equity": (debt / equity
                                   if equity and not np.isnan(debt)
                                   else np.nan),
                "fetched_ok": bool(info.get("shortName") or info.get("longName")),
            }
            if prof["fetched_ok"]:
                break
        except Exception as e:  # noqa: BLE001
            log.warning("profile %s attempt %d failed: %s", sym, attempt + 1, e)
            time.sleep(5)

    if prof is not None:
        cached[sym] = prof
        _save_cache("profiles", cached)
    return prof
