"""Technical indicators computed from daily OHLCV frames."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> float:
    """Wilder RSI on a close-price series; returns the latest value."""
    if len(close) < period + 1:
        return float("nan")
    delta = close.diff().dropna()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    val = 100 - (100 / (1 + rs))
    out = val.iloc[-1]
    if pd.isna(out):  # avg_loss == 0 -> pure up-move
        return 100.0 if avg_loss.iloc[-1] == 0 and avg_gain.iloc[-1] > 0 else float("nan")
    return float(out)


def sma(close: pd.Series, window: int) -> float:
    if len(close) < window:
        return float("nan")
    return float(close.tail(window).mean())


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 1:
        return float("nan")
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return float(tr.ewm(alpha=1 / period, min_periods=period).mean().iloc[-1])


def pct_return(close: pd.Series, days: int) -> float:
    if len(close) < days + 1:
        return float("nan")
    return float(close.iloc[-1] / close.iloc[-days - 1] - 1)


def beta_vs(close: pd.Series, benchmark: pd.Series, window: int = 252) -> float:
    """OLS beta of daily returns vs a benchmark close series."""
    a, b = close.align(benchmark, join="inner")
    ra, rb = a.pct_change().dropna(), b.pct_change().dropna()
    ra, rb = ra.align(rb, join="inner")
    ra, rb = ra.tail(window), rb.tail(window)
    if len(rb) < 60 or rb.var() == 0:
        return float("nan")
    return float(np.cov(ra, rb)[0, 1] / rb.var())


def swing_low(low: pd.Series, days: int) -> float:
    """Lowest low over the lookback — Sharran-style 'fear' entry reference."""
    if len(low) == 0:
        return float("nan")
    return float(low.tail(days).min())


def consolidation_low(low: pd.Series, days: int = 63, pct: float = 0.20) -> float:
    """A 're-tested' support: the pct-quantile of lows over the lookback.

    More robust than the raw minimum — represents where buyers repeatedly
    stepped in rather than a single capitulation wick.
    """
    if len(low) == 0:
        return float("nan")
    return float(low.tail(days).quantile(pct))


def compute_technicals(df: pd.DataFrame, spy_close: pd.Series | None) -> dict:
    """Full technical feature dict for one ticker's OHLCV frame."""
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    close = close.dropna()
    if len(close) < 60:
        return {}
    last = float(close.iloc[-1])
    hi52 = float(high.tail(252).max())
    lo52 = float(low.tail(252).min())
    sma50, sma200 = sma(close, 50), sma(close, 200)
    atrv = atr(high, low, close)
    avg_vol63 = float(vol.tail(63).mean())
    avg_vol21 = float(vol.tail(21).mean())

    return {
        "last": last,
        "ret_1m": pct_return(close, 21),
        "ret_3m": pct_return(close, 63),
        "ret_6m": pct_return(close, 126),
        "ret_12m": pct_return(close, 252),
        "high_52w": hi52,
        "low_52w": lo52,
        "pct_off_high": last / hi52 - 1 if hi52 else np.nan,
        "pct_above_low": last / lo52 - 1 if lo52 else np.nan,
        "sma50": sma50,
        "sma200": sma200,
        "dist_sma50": last / sma50 - 1 if sma50 else np.nan,
        "dist_sma200": last / sma200 - 1 if sma200 else np.nan,
        "trend_up": bool(sma50 > sma200) if sma50 and sma200 else False,
        "rsi14": rsi(close),
        "atr_pct": atrv / last if atrv and last else np.nan,
        "avg_dollar_vol": avg_vol63 * last,
        "vol_trend": avg_vol21 / avg_vol63 if avg_vol63 else np.nan,
        "support_3m": swing_low(low, 63),
        "consol_low_3m": consolidation_low(low, 63),
        "support_6m": swing_low(low, 126),
        "beta": beta_vs(close, spy_close) if spy_close is not None else np.nan,
    }
