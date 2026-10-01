"""Market data access via yfinance with a daily on-disk cache.

Three tiers of data, in order of cost:
  1. Bulk daily OHLCV for the whole universe (chunked yf.download)
  2. Fundamentals (Ticker.info) for the pre-filtered shortlist only
  3. Options chains, fetched lazily for the names the user inspects
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import re

import pandas as pd
import yfinance as yf
from curl_cffi import requests as cf_requests

log = logging.getLogger(__name__)

# Yahoo 429s plain-HTTP fingerprints; impersonate a real browser.
SESSION = cf_requests.Session(impersonate="chrome")

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
BENCH = {"SPY": "SPY", "TSX": "^GSPTSE"}
HIST_PERIOD = "1y"
CHUNK = 100


def normalize_symbol(s: str) -> str:
    """DFN TO -> DFN.TO; BBD.B.TO -> BBD-B.TO; BRK.B -> BRK-B (Yahoo style)."""
    s = re.sub(r"\s+TO$", ".TO", s.strip().upper())
    if s.endswith(".TO"):
        s = s[:-3].replace(".", "-") + ".TO"
    else:
        s = s.replace(".", "-")
    return s


def _fresh_session():
    """New impersonated session (new cookies/crumb) after a 401/429 burst."""
    return cf_requests.Session(impersonate="chrome")


def _today() -> str:
    return date.today().isoformat()


def _cache_path(kind: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{kind}_{_today()}.pkl"


def _load_cache(kind: str):
    p = _cache_path(kind)
    if p.exists():
        with p.open("rb") as f:
            return pickle.load(f)
    return None


def _save_cache(kind: str, obj) -> None:
    with _cache_path(kind).open("wb") as f:
        pickle.dump(obj, f)


def _download_chunk(tickers: list[str],
                    period: str = HIST_PERIOD) -> pd.DataFrame:
    return yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
        session=SESSION,
    )


def _bulk_download(syms: list[str], period: str = HIST_PERIOD,
                   min_rows: int = 60) -> dict[str, pd.DataFrame]:
    """Chunked yf.download with retry sweep. Returns {ticker: OHLCV}."""
    frames: dict[str, pd.DataFrame] = {}
    for i in range(0, len(syms), CHUNK):
        chunk = syms[i : i + CHUNK]
        try:
            raw = _download_chunk(chunk, period)
        except Exception as e:  # noqa: BLE001 - retry the chunk once
            log.warning("chunk download failed (%s); retrying once", e)
            time.sleep(5)
            try:
                raw = _download_chunk(chunk, period)
            except Exception as e2:  # noqa: BLE001
                log.warning("chunk retry failed: %s", e2)
                continue
        if raw.empty:
            continue
        for sym in chunk:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if sym not in raw.columns.get_level_values(0):
                        continue
                    df = raw[sym].dropna(how="all")
                else:  # single ticker came back flat
                    df = raw.dropna(how="all")
                if len(df) >= min_rows and \
                        df["Close"].dropna().size >= min_rows:
                    frames[sym] = df
            except Exception:  # noqa: BLE001 - skip malformed tickers
                continue
        time.sleep(2.5)  # stay under Yahoo's burst limit

    # sweep pass: retry the bulk-dropped names chunked, then individually
    missing = [s for s in syms if s not in frames]
    if missing:
        log.info("retrying %d dropped tickers", len(missing))
        for i in range(0, len(missing), 50):
            chunk = missing[i : i + 50]
            try:
                raw = _download_chunk(chunk, period)
                if not raw.empty:
                    for sym in chunk:
                        try:
                            if isinstance(raw.columns, pd.MultiIndex) \
                                    and sym in raw.columns.get_level_values(0):
                                df = raw[sym].dropna(how="all")
                            else:
                                continue
                            if len(df) >= min_rows:
                                frames[sym] = df
                        except Exception:  # noqa: BLE001
                            continue
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2.5)
    missing = [s for s in syms if s not in frames]
    for s in missing:
        df = download_one(s, period=period)
        if df is not None and len(df) >= min_rows:
            frames[s] = df
        time.sleep(0.3)
    return frames


def _latest_history_file() -> Path | None:
    """Most recent history cache older than today, for incremental updates."""
    files = sorted(CACHE_DIR.glob("history_*.pkl"))
    for p in reversed(files):
        if p.name != f"history_{_today()}.pkl":
            return p
    return None


def _merge_history(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([old, new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged.tail(400)  # ~18 months of daily bars is plenty for techs


def download_history(tickers: list[str],
                     force: bool = False) -> dict[str, pd.DataFrame]:
    """Return {ticker: OHLCV DataFrame}. Cached per calendar day.

    If yesterday's cache exists and covers this universe, only the last
    ~3 weeks are downloaded and merged - what makes a ~4k-ticker
    RRSP-wide universe feasible to refresh daily.
    """
    bench_syms = list(BENCH.values())
    all_syms = list(dict.fromkeys(bench_syms + tickers))

    if not force:
        cached = _load_cache("history")
        if cached is not None:
            missing = [s for s in all_syms if s not in cached]
            if not missing:
                return cached
            # Same-day cache exists but under-covers this universe (e.g.
            # switching to the wider RRSP universe) - top up the gap.
            log.info("history cache covers %d/%d tickers; fetching %d "
                     "missing", len(cached), len(all_syms), len(missing))
            frames = dict(cached)
            frames.update(_bulk_download(missing))
            _save_cache("history", frames)
            return frames

    prev: dict[str, pd.DataFrame] = {}
    prev_file = _latest_history_file()
    if prev_file is not None:
        try:
            with prev_file.open("rb") as f:
                prev = pickle.load(f)
        except Exception as e:  # noqa: BLE001
            log.warning("prior history cache unreadable: %s", e)
    coverage = len(set(tickers) & set(prev)) / max(len(tickers), 1)

    if prev and coverage >= 0.8:
        log.info("incremental refresh: %d tickers over cached history",
                 len(all_syms))
        fresh = _bulk_download(all_syms, period="20d", min_rows=1)
        frames = {}
        for sym in all_syms:
            old = prev.get(sym)
            new = fresh.get(sym)
            if old is None:
                continue  # new names need a full fetch below
            df = _merge_history(old, new) if new is not None else old
            if len(df) >= 60 and df["Close"].dropna().size >= 60:
                frames[sym] = df
        newbies = [s for s in all_syms if s not in prev]
        if newbies:
            log.info("full history for %d new tickers", len(newbies))
            frames.update(_bulk_download(newbies))
    else:
        frames = _bulk_download(all_syms)

    if frames:
        _save_cache("history", frames)
    return frames


def download_one(sym: str, session=None,
                 period: str = HIST_PERIOD) -> pd.DataFrame | None:
    """Single-ticker history; bypasses the daily cache."""
    try:
        df = yf.download(
            sym, period=period, auto_adjust=True,
            progress=False, session=session or SESSION,
        )
    except Exception:  # noqa: BLE001
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(how="all")
    return df if len(df) >= (60 if period == HIST_PERIOD else 1) else None


_FUND_FIELDS = [
    "shortName", "sector", "industry", "marketCap", "currency",
    "trailingPE", "forwardPE", "profitMargins", "grossMargins",
    "operatingMargins", "revenueGrowth", "earningsGrowth",
    "earningsQuarterlyGrowth", "dividendYield", "payoutRatio",
    "debtToEquity", "returnOnEquity", "freeCashflow",
    "targetMeanPrice", "recommendationKey", "numberOfAnalystOpinions",
    "currentPrice", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "beta",
    "totalDebt", "totalCash",
]


def _fetch_one_info(sym: str, session=None) -> tuple[str, dict]:
    try:
        info = yf.Ticker(sym, session=session or SESSION).info or {}
        return sym, {k: info.get(k) for k in _FUND_FIELDS}
    except Exception:  # noqa: BLE001 - treat as missing fundamentals
        return sym, {}


def fetch_fundamentals(
    tickers: list[str], force: bool = False, max_workers: int = 4
) -> dict[str, dict]:
    """Fundamentals for a shortlist; cached per calendar day."""
    cached = _load_cache("fundamentals") if not force else None
    cached = cached or {}
    missing = [t for t in tickers if not cached.get(t)]
    if not missing:
        return {t: cached[t] for t in tickers}

    for attempt in range(3):
        sess = SESSION if attempt == 0 else _fresh_session()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_fetch_one_info, t, sess): t for t in missing}
            for fut in as_completed(futs):
                sym, info = fut.result()
                if info:
                    cached[sym] = info
        missing = [t for t in tickers if not cached.get(t)]
        if not missing:
            break
        log.info("fundamentals retry %d: %d still missing", attempt + 1,
                 len(missing))
        time.sleep(8)

    if any(cached.get(t) for t in tickers):
        _save_cache("fundamentals", cached)
    return {t: cached.get(t, {}) for t in tickers}


def earnings_date(sym: str) -> str | None:
    """Next earnings date (YYYY-MM-DD) if Yahoo exposes it."""
    try:
        cal = yf.Ticker(sym, session=SESSION).calendar
        if cal is None:
            return None
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if isinstance(ed, (list, tuple)) and ed:
                return str(ed[0])[:10]
            if ed:
                return str(ed)[:10]
        elif hasattr(cal, "loc") and "Earnings Date" in cal.index:
            return str(cal.loc["Earnings Date"].iloc[0])[:10]
    except Exception:  # noqa: BLE001
        return None
    return None


def get_option_chain(sym: str, expiry: str):
    return yf.Ticker(sym, session=SESSION).option_chain(expiry)


def get_option_expiries(sym: str) -> tuple[str, ...]:
    try:
        return tuple(yf.Ticker(sym, session=SESSION).options)
    except Exception:  # noqa: BLE001
        return ()
