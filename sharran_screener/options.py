"""Long-call contract selection.

Sharran trades options in ~30-45 day cycles (covered calls). For *buying*
long calls the logic inverts: buy time, own the upside. We scan the real
chain for slightly ITM calls with 60-180 DTE, delta ~0.60, tight spreads
and real open interest, then rank them.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime
from statistics import NormalDist

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

NORM = NormalDist()
RISK_FREE = 0.043          # ~10y UST; used only for delta estimation
MIN_DTE, MAX_DTE = 45, 200
TARGET_DELTA = 0.62
DELTA_BAND = (0.45, 0.78)
MIN_OI = 100
MIN_VOL = 5
MAX_SPREAD_PCT = 0.15


def _bs_call_delta(S: float, K: float, T: float, sigma: float, r: float) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    return NORM.cdf(d1)


def _dte(expiry: str) -> int:
    return (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days


def scan_long_calls(
    ticker: str,
    spot: float,
    expiries: tuple[str, ...],
    chain_fetch,
    max_expiries: int = 3,
) -> list[dict]:
    """Return ranked candidate contracts for `ticker`.

    `chain_fetch(expiry)` must return a yfinance option chain (.calls df).
    """
    usable = [e for e in expiries if MIN_DTE <= _dte(e) <= MAX_DTE]
    usable.sort(key=lambda e: abs(_dte(e) - 105))  # prefer ~3-4 months out
    candidates: list[dict] = []

    for exp in usable[:max_expiries]:
        try:
            chain = chain_fetch(exp)
        except Exception as e:  # noqa: BLE001
            log.warning("%s %s chain failed: %s", ticker, exp, e)
            continue
        calls = chain.calls.copy()
        if calls.empty:
            continue
        T = _dte(exp) / 365.0
        calls["mid"] = (calls["bid"] + calls["ask"]) / 2
        calls["spread_pct"] = np.where(
            calls["mid"] > 0, (calls["ask"] - calls["bid"]) / calls["mid"], np.inf
        )
        calls["delta"] = calls.apply(
            lambda r: _bs_call_delta(
                spot, r["strike"], T, max(r.get("impliedVolatility") or 0, 0.05),
                RISK_FREE,
            ),
            axis=1,
        )
        m = calls[
            (calls["strike"] >= spot * 0.95)
            & (calls["strike"] <= spot * 1.12)
            & (calls["delta"].between(*DELTA_BAND))
            & (calls["openInterest"].fillna(0) >= MIN_OI)
            & (calls["volume"].fillna(0) >= MIN_VOL)
            & (calls["spread_pct"] <= MAX_SPREAD_PCT)
            & (calls["mid"] > 0)
        ]
        for _, c in m.iterrows():
            be = c["strike"] + c["mid"]
            candidates.append({
                "ticker": ticker,
                "contract": c.get("contractSymbol", ""),
                "expiry": exp,
                "dte": _dte(exp),
                "strike": round(float(c["strike"]), 2),
                "mid": round(float(c["mid"]), 2),
                "delta": round(float(c["delta"]), 2),
                "iv": round(float(c.get("impliedVolatility") or 0), 3),
                "oi": int(c.get("openInterest") or 0),
                "vol": int(c.get("volume") or 0),
                "spread_pct": round(float(c["spread_pct"]) * 100, 1),
                "breakeven": round(be, 2),
                "be_vs_spot": round(be / spot - 1, 3),
            })

    for c in candidates:
        c["rank"] = (
            (1 - abs(c["delta"] - TARGET_DELTA) / 0.5) * 40
            + (1 - min(c["spread_pct"] / 100, MAX_SPREAD_PCT) / MAX_SPREAD_PCT) * 25
            + min(math.log10(max(c["oi"], 1)) / 5, 1) * 20
            + min(math.log10(max(c["vol"], 1)) / 4, 1) * 15
        )
    candidates.sort(key=lambda c: c["rank"], reverse=True)
    return candidates
