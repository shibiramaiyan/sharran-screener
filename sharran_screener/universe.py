"""Universe construction: US large/mid caps + S&P/TSX Composite (Canada).

US side uses the S&P 500 + S&P 400 (MidCap) constituent lists from
Wikipedia (~900 names, the practical Russell-1000 equivalent — iShares
gates IWB holdings downloads behind a JS app shell now). The TSX list comes
from Wikipedia's S&P/TSX Composite Index page, with an iShares XIC attempt
kept as a secondary path and a static mega-cap list as last resort.
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from yfinance import EquityQuery

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
UNIVERSE_CSV = CACHE_DIR / "universe.csv"
UNIVERSE_RRSP_CSV = CACHE_DIR / "universe_rrsp.csv"
UNIVERSE_WIDE_CSV = CACHE_DIR / "universe_wide.csv"
CACHE_TTL_SEC = 24 * 3600

XIC_URL = (
    "https://www.ishares.com/ca/products/239858/"
    "ishares-core-sp-tsx-capped-composite-index-etf/"
    "1467271812596.ajax?fileType=csv&fileName=XIC_holdings&dataType=fund"
)
TSX_WIKI_URL = "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index"
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP400_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

# Last-resort fallback: liquid large caps only.
FALLBACK_US = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "BRK-B",
    "JPM", "LLY", "V", "XOM", "UNH", "COST", "MA", "HD", "PG", "NFLX", "CRM",
    "AMD", "BAC", "KO", "PEP", "WMT", "ORCL", "MCD", "CSCO", "ABT", "LIN",
    "ADBE", "INTC", "DIS", "VZ", "T", "CVX", "QCOM", "TXN", "AMGN", "PM",
    "IBM", "GS", "MS", "CAT", "BA", "LOW", "SPGI", "GE", "ISRG", "SYK",
]

FALLBACK_CA = [
    "RY", "TD", "BNS", "BMO", "CM", "ENB", "TRP", "CNR", "CP", "SHOP",
    "BCE", "TRI", "BAM", "MFC", "SLF", "SU", "CNQ", "IMO", "WCN", "CSU",
    "ATD", "FTS", "EMA", "NTR", "MGA", "ABX", "WPM", "FNV", "L", "QBR-B",
]


def _yahoo_symbol(raw: str, market: str) -> str | None:
    """Normalise an index/ETF ticker into a Yahoo Finance symbol."""
    t = raw.strip().upper().replace(" ", "-").replace(".", "-")
    if not t or t == "-" or "CASH" in t:
        return None
    # Yahoo uses -B style classes: BRK-B, BBD-B.TO, QBR-B.TO
    if market == "CA":
        t = t + ".TO"
    return t


def _fetch_ishares_csv(url: str) -> pd.DataFrame:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    text = resp.text
    # iShares CSVs have ~9 preamble lines; find the real header row.
    lines = text.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith('"Ticker"')), None
    )
    if header_idx is None:
        raise ValueError("holdings header row not found")
    csv_text = "\n".join(lines[header_idx:])
    # drop trailing metadata rows after a blank line
    rows = []
    for ln in csv_text.splitlines():
        if not ln.strip():
            break
        rows.append(ln)
    df = pd.read_csv(io.StringIO("\n".join(rows)), dtype=str)
    return df


def _wiki_symbols(url: str, market: str) -> list[str]:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    for t in pd.read_html(io.StringIO(resp.text)):
        cols = [str(c).lower() for c in t.columns]
        hit = [i for i, c in enumerate(cols)
               if "symbol" in c or "ticker" in c]
        if hit:
            out = [
                s for s in (_yahoo_symbol(str(x), market)
                            for x in t[t.columns[hit[0]]].dropna())
                if s
            ]
            if len(out) >= 50:
                return out
    raise ValueError(f"no constituent table on {url}")


def fetch_us_universe() -> list[str]:
    """S&P 500 + S&P 400 as the US large/mid-cap universe."""
    tickers = _wiki_symbols(SP500_WIKI_URL, "US") + \
        _wiki_symbols(SP400_WIKI_URL, "US")
    return list(dict.fromkeys(tickers))


def _tsx_from_ishares() -> list[str]:
    df = _fetch_ishares_csv(XIC_URL)
    df.columns = [c.strip() for c in df.columns]
    tickers = []
    for raw in df["Ticker"].dropna():
        sym = _yahoo_symbol(raw, "CA")
        if sym:
            tickers.append(sym)
    if len(tickers) < 50:
        raise ValueError(f"XIC list too short: {len(tickers)}")
    return tickers


def _tsx_from_wikipedia() -> list[str]:
    return _wiki_symbols(TSX_WIKI_URL, "CA")


def fetch_tsx_composite() -> list[str]:
    try:
        return _tsx_from_ishares()
    except Exception as e:  # noqa: BLE001 - fall back to Wikipedia
        log.warning("XIC holdings fetch failed (%s); trying Wikipedia", e)
    try:
        return _tsx_from_wikipedia()
    except Exception as e:  # noqa: BLE001
        log.warning("Wikipedia TSX fetch failed (%s)", e)
    return [_yahoo_symbol(t, "CA") for t in FALLBACK_CA]


def load_universe(force_refresh: bool = False,
                  rrsp: bool = False,
                  india: bool = False) -> pd.DataFrame:
    """Return DataFrame[ticker, market, source] for the chosen universe.

    rrsp=True returns every common equity on CRA-designated exchanges
    (NYSE/NASDAQ/NYSE American + TSX/TSXV) that clears a minimal
    investability floor - i.e. everything RRSP-valid the model can
    reasonably score. india=True additionally sweeps the NSE (not
    RRSP-eligible - India is not a CRA designated exchange - but
    screenable for non-registered research).
    """
    csv = (UNIVERSE_WIDE_CSV if india else
           UNIVERSE_RRSP_CSV if rrsp else UNIVERSE_CSV)
    if not force_refresh and csv.exists():
        age = time.time() - csv.stat().st_mtime
        if age < CACHE_TTL_SEC:
            return pd.read_csv(csv)

    if rrsp or india:
        markets = ["US", "CA"] + (["IN"] if india else [])
        try:
            lists = {m: fetch_rrsp_market(m) for m in markets}
            if len(lists["US"]) < 1000 or len(lists["CA"]) < 100 or (
                    india and len(lists["IN"]) < 500):
                raise ValueError(
                    "screener too short: "
                    + str({m: len(v) for m, v in lists.items()}))
        except Exception as e:  # noqa: BLE001 - fall back to index universe
            log.warning("wide universe failed (%s); using index universe", e)
            df = load_universe(force_refresh)
            df["source"] = df["source"] + " (index fallback)"
            return df
        source = {"US": "NYSE/NASDAQ/AMEX", "CA": "TSX/TSXV", "IN": "NSE"}
        df = pd.DataFrame(
            [(t, m) for m, lst in lists.items() for t in lst],
            columns=["ticker", "market"],
        ).drop_duplicates("ticker")
    else:
        try:
            us = fetch_us_universe()
        except Exception as e:  # noqa: BLE001
            log.warning("US universe fetch failed (%s); using fallback list", e)
            us = FALLBACK_US
        ca = fetch_tsx_composite()
        source = {"US": "S&P 500+400", "CA": "TSX Composite"}
        df = pd.DataFrame(
            [(t, "US") for t in us] + [(t, "CA") for t in ca],
            columns=["ticker", "market"],
        ).drop_duplicates("ticker")
    df["source"] = df["market"].map(source)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    return df


# ---------------------------------------------------------------------------
# Wide universe: every common equity on CRA designated exchanges (plus NSE
# when india=True). Yahoo's equity screener is paginated 250/page and
# returns marketCap, avg volume and quoteType inline, so junk is dropped
# before the (expensive) history download.
# ---------------------------------------------------------------------------

_RRSP_EXCHANGES = {
    "US": ["NYQ", "NMS", "ASE"],
    "CA": ["TOR", "VAN"],
    "IN": ["NSI"],          # NSE; BSE adds mostly illiquid microcap dupes
}
_BAD_NAME = re.compile(
    r"(?i)(warrant|preferred|preference|autocallable|debenture|"
    r"subordinated|notes? due|structured|rate reset|etn\b|etf\b)")
# -P* preferreds, -WT/-WS warrants, -R rights, -U units (US only - on the TSX
# -UN is a REIT unit worth keeping and -U a USD-settled line).
_BAD_SYM_US = re.compile(r"-(P[A-Z]*|W[TS]?|R|U)$")
_BAD_SYM_CA = re.compile(r"-(P[A-Z]*|W[TS]?|R)\.TO$")
_BAD_SYM_IN = re.compile(r"-(W[TS]?|R)\.NS$")
# Investability floors, in each market's listing currency (~$50M mcap,
# ~$250K/day for US/CA; ~Rs.400cr / ~Rs.2cr for India).
_FLOORS = {
    "US": (50e6, 250e3),
    "CA": (50e6, 250e3),
    "IN": (4e9, 20e6),
}


def _rrsp_page(exchanges: list[str], offset: int) -> dict:
    for attempt in (0, 1):
        try:
            return yf.screen(
                EquityQuery("is-in", ["exchange", *exchanges]),
                size=250, offset=offset) or {}
        except Exception as e:  # noqa: BLE001
            if attempt:
                log.warning("screener page @%d failed: %s", offset, e)
                return {}
            time.sleep(5)
    return {}


def fetch_rrsp_market(market: str) -> list[str]:
    """All investable common equities/ADRs/REITs on one country's exchanges."""
    bad_sym = {"CA": _BAD_SYM_CA, "IN": _BAD_SYM_IN}.get(market, _BAD_SYM_US)
    min_mcap, min_dv = _FLOORS[market]
    out, offset, total = [], 0, 1
    while offset < total:
        resp = _rrsp_page(_RRSP_EXCHANGES[market], offset)
        quotes = resp.get("quotes") or []
        total = resp.get("total") or 0
        if not quotes:
            break
        for q in quotes:
            sym = q.get("symbol") or ""
            name = q.get("longName") or q.get("shortName") or ""
            if q.get("quoteType") != "EQUITY" or not sym:
                continue
            if _BAD_NAME.search(name) or bad_sym.search(sym):
                continue
            if (q.get("marketCap") or 0) < min_mcap:
                continue
            dv = (q.get("averageDailyVolume3Month") or 0) * \
                (q.get("regularMarketPrice") or 0)
            if dv < min_dv:
                continue
            out.append(sym)
        offset += len(quotes)
        time.sleep(0.4)
    return out

