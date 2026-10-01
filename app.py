"""Sharran Screener - daily stock ideas scored like Sharran Srivatsaa.

Run:  .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import pickle
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from sharran_screener import market_data as md
from sharran_screener import card, model, options, profile
from sharran_screener.indicators import compute_technicals
from sharran_screener.universe import load_universe

CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache"

st.set_page_config(page_title="Sharran Screener", layout="wide")

METHODOLOGY = """
### How this thinks like Sharran (sources: Business School podcast transcripts)

**The filter** — Investment X-Ray scorecard (Ep. 258 & 302). Every stock is
graded /25 on four buckets, exactly like his paper scorecard:
| Bucket | What earns points |
|---|---|
| Capital preservation | mega/large cap, profitable, low beta, low leverage, low daily vol |
| Tax efficiency | low dividend tax drag, buy-and-hold suitability (see account tip) |
| Cash flow | dividend yield, sustainable payout, FCF yield, covered-call income |
| Growth | revenue & earnings growth, analyst upside, 12m momentum |

**When to buy** — "Be greedy when others are fearful" (Ep. 333). He keeps
20% in *opportunistic cash* and deploys it when quality names drop 10-35%
below their 52-week high. The dip score rewards that discount — but only if
the X-Ray quality score holds, so you don't catch value traps.

**Best price** — he ladders into weakness instead of all-in market buys:
Zone 1 = recent support, Zone 2 = -20% from the 52w high, Zone 3 = -30%.

**Short term** — "ride the narrative, follow global growth" — relative
strength vs SPY, RSI 50-70, price above 50/200dma, rising volume.

**Long calls** — he runs options in cycles. Scanner picks slightly-ITM calls
(delta ~0.6), 45-200 days out, tight spread, real open interest.

**Business cards** — his #1 rule can't be coded ("only invest in what you
know and understand"), so every pick gets a one-page profile: what the
company does, how it earns, what funds its assets (equity vs debt), and 4
years of revenue/EPS/free-cash-flow history. Read it, then decide.

**Portfolio rules** (Ep. 333): 20% opportunistic cash · position ≤5% ·
theme ≤10% · hold 7-10 names · every position needs a second job
(pair trade shown per pick) · weekly check, monthly options review,
quarterly tax-loss harvest.
"""


def screen_cache_path() -> Path:
    return CACHE_DIR / f"screen_{date.today().isoformat()}.pkl"


@st.cache_data(show_spinner="Loading index constituents...", ttl=3600)
def get_universe(force: bool) -> pd.DataFrame:
    return load_universe(force_refresh=force)


def run_screen(force: bool, min_xray: float) -> tuple[list[model.Pick], dict]:
    cache_file = screen_cache_path()
    if not force and cache_file.exists():
        with cache_file.open("rb") as f:
            payload = pickle.load(f)
        picks = [p for p in payload["picks"] if p.xray >= min_xray]
        return picks, payload["meta"]

    uni = get_universe(force)
    tickers = uni["ticker"].tolist()
    market_map = dict(zip(uni["ticker"], uni["market"]))

    with st.status(f"Downloading 1y daily prices for {len(tickers)} tickers...",
                   expanded=True) as status:
        history = md.download_history(tickers, force=force)
        st.write(f"{len(history)} tickers with usable history")
        spy = history.get("SPY")
        spy_close = spy["Close"] if spy is not None else None
        spy_3m = float(spy_close.iloc[-1] / spy_close.iloc[-64] - 1) \
            if spy_close is not None and len(spy_close) > 64 else None

        st.write("Computing technicals...")
        techs = {
            s: compute_technicals(df, spy_close)
            for s, df in history.items()
            if s not in model.BENCH_VALUES
        }
        techs = {s: t for s, t in techs.items() if t}

        ranked = sorted(techs.items(),
                        key=lambda kv: model._prefilter_key(kv[1]),
                        reverse=True)
        shortlist = [s for s, t in ranked
                     if model._prefilter_key(t) >= 0][:model.SHORTLIST_N]
        st.write(f"{len(shortlist)} passed liquidity/price pre-filter; "
                 "fetching fundamentals...")

        funds = md.fetch_fundamentals(shortlist)
        picks = model.build_picks(
            {s: techs[s] for s in shortlist}, funds, market_map, spy_3m
        )
        picks = [p for p in picks if p.xray >= min_xray]
        status.update(label="Screen complete", state="complete")

    meta = {
        "universe_n": len(tickers),
        "with_history": len(history),
        "shortlist": len(shortlist),
        "ran_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    if picks:  # never cache an empty/poisoned run
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with cache_file.open("wb") as f:
            pickle.dump({"picks": picks, "meta": meta}, f)
    return picks, meta


def pick_row(p: model.Pick) -> dict:
    return {
        "Ticker": p.ticker,
        "Mkt": p.market,
        "Name": p.name,
        "Sector": p.sector,
        "Price": p.last,
        "X-Ray": p.xray,
        "CP": p.cp, "Tax": p.tax, "CF": p.cf, "Gr": p.gr,
        "Dip": p.dip_score,
        "Mom": p.short_score,
        "LT": p.long_score,
        "OffHigh%": round(p.tech.get("pct_off_high", 0) * 100, 1),
        "RSI": round(p.tech.get("rsi14", 0), 0),
        "Signal": p.entry_signal,
        "Zone1": p.entry_ladder.get("zone1_support"),
        "Zone2": p.entry_ladder.get("zone2_-20%"),
        "Zone3": p.entry_ladder.get("zone3_-30%"),
        "Pair": p.pair.get("ticker", ""),
    }


# ---------------------------------------------------------------------------
# Sidebar - his rules + controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Sharran Screener")
    st.caption("Daily ideas scored the way Sharran Srivatsaa grades investments.")

    portfolio = st.number_input("Portfolio value ($)", min_value=0.0,
                                value=100000.0, step=10000.0)
    mkt = st.radio("Market", ["Both", "US", "CA"], horizontal=True)
    min_xray = st.slider("Min X-Ray quality", 0, 100, 50)
    st.divider()
    st.markdown(
        "**His rules (Ep. 333)**\n"
        f"- Opportunistic cash: **${portfolio*0.20:,.0f}** (20%)\n"
        f"- Max per position: **${portfolio*0.05:,.0f}** (5%)\n"
        f"- Max per theme: **${portfolio*0.10:,.0f}** (10%)\n"
        "- Hold 7-10 positions total"
    )
    st.divider()
    force = st.checkbox("Ignore today's cache", value=False)
    run = st.button("Run daily screen", type="primary", width="stretch")
    st.divider()
    st.markdown(
        "**His cadence**\n"
        "- Fri: 5-min position check\n"
        "- Monthly: manage pairs & options (30-45d cycles)\n"
        "- Quarterly: tax-loss harvest losers"
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

cache_file = screen_cache_path()
picks, meta = [], {}
if cache_file.exists() and not run and not force:
    with cache_file.open("rb") as f:
        payload = pickle.load(f)
    picks, meta = payload["picks"], payload["meta"]
    picks = [p for p in picks if p.xray >= min_xray]
elif run:
    picks, meta = run_screen(force, min_xray)

if mkt != "Both":
    picks = [p for p in picks if p.market == mkt]

if not picks:
    st.title("Sharran Screener")
    if meta:
        st.warning("No picks under the current filters "
                   f"(market={mkt}, min X-Ray={min_xray}).")
    else:
        st.info("Press **Run daily screen** in the sidebar. First run "
                "downloads ~1,300 tickers of price history and takes a few "
                "minutes; it's cached for the rest of the day.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Universe", meta.get("universe_n", "-"))
c2.metric("With price data", meta.get("with_history", "-"))
c3.metric("Pass quality filter", len(picks))
c4.metric("Screen ran", meta.get("ran_at", "-"))

tab_lt, tab_st, tab_calls, tab_deep, tab_doc = st.tabs(
    ["Long-Term", "Short-Term", "Long Calls", "Business Cards", "Methodology"]
)

df_all = pd.DataFrame([pick_row(p) for p in picks])


def _lookup_pick(sym: str) -> model.Pick | None:
    p = next((x for x in picks if x.ticker == sym), None)
    if p is not None:
        return p
    df1 = md.download_one(sym)
    spy = md.download_one("SPY")
    if df1 is None:
        return None
    spy_close = spy["Close"] if spy is not None else None
    spy_3m = float(spy_close.iloc[-1] / spy_close.iloc[-64] - 1) \
        if spy_close is not None and len(spy_close) > 64 else None
    tech = compute_technicals(df1, spy_close)
    fund = md.fetch_fundamentals([sym]).get(sym, {})
    return model.build_picks({sym: tech}, {sym: fund}, {sym: "?"}, spy_3m)[0]


@st.dialog("Business Card", width="large")
def _card_dialog(sym: str) -> None:
    p = _lookup_pick(sym)
    if p is None:
        st.warning("No data for that ticker (use .TO suffix for TSX names).")
        return
    prof = profile.fetch_business_profile(sym)
    if prof is None:
        st.warning("Profile fetch failed (rate limit?) - try again shortly.")
        return
    card.render_business_card(p, prof, portfolio)


def _clickable_table(df: pd.DataFrame, key: str,
                     sym_col: str = "Ticker") -> None:
    """Table where clicking a row opens that stock's business card."""
    if st.session_state.pop(f"_open_{key}", False):
        st.session_state.pop(key, None)  # consumed: reset selection
    ev = st.dataframe(df, width="stretch", hide_index=True,
                      on_select="rerun", selection_mode="single-row",
                      key=key)
    if ev.selection.rows:
        st.session_state[f"_open_{key}"] = True
        _card_dialog(df.iloc[ev.selection.rows[0]][sym_col])


with tab_lt:
    st.subheader("Buy-and-hold candidates - quality on sale")
    lt = df_all.sort_values("LT", ascending=False).head(30)
    _clickable_table(lt, "lt_tbl")
    if not lt.empty:
        theme_counts = lt["Sector"].value_counts()
        over = theme_counts[theme_counts > 2]
        if not over.empty:
            st.warning("Theme concentration: " + ", ".join(
                f"{s} x{n}" for s, n in over.items()) +
                " — Sharran caps a theme at 10% (≈2 of 20 slots).")

with tab_st:
    st.subheader("Short-term momentum - ride the narrative")
    stt = df_all.sort_values("Mom", ascending=False).head(30)
    _clickable_table(
        stt[["Ticker", "Mkt", "Name", "Sector", "Price", "Mom", "X-Ray",
             "RSI", "OffHigh%", "Signal", "Pair"]], "st_tbl")

with tab_calls:
    st.subheader("Long-call chain scan")
    candidates = sorted(picks, key=lambda p: p.long_score, reverse=True)[:40]
    choices = {f"{p.ticker} — {p.name} (LT {p.long_score})": p
               for p in candidates}
    sel = st.multiselect("Scan contracts for:", list(choices)[:20],
                         default=list(choices)[:5])
    if st.button("Scan options chains"):
        rows = []
        prog = st.progress(0)
        for i, lbl in enumerate(sel):
            p = choices[lbl]
            exps = md.get_option_expiries(p.ticker)
            rows += options.scan_long_calls(
                p.ticker, p.tech.get("last", 0), exps,
                lambda e, t=p.ticker: md.get_option_chain(t, e))
            prog.progress((i + 1) / len(sel))
        prog.empty()
        if rows:
            cdf = pd.DataFrame(rows).sort_values("rank", ascending=False)
            _clickable_table(
                cdf[["ticker", "contract", "expiry", "dte", "strike", "mid",
                     "delta", "iv", "oi", "vol", "spread_pct", "breakeven",
                     "be_vs_spot", "rank"]],
                "calls_tbl", sym_col="ticker")
            st.caption("Delta ≈0.6 slightly-ITM, 45-200 DTE. Breakeven vs "
                       "spot shows how far the stock must move by expiry.")
        else:
            st.warning("No contracts passed the filters (liquidity/spread).")

with tab_deep:
    st.subheader("Business cards - understand before you buy")
    st.caption("His #1 rule: only invest in what you know and understand. "
               "Each card shows what the company does, how it earns, its "
               "balance sheet, and its earnings track record.")

    mode = st.radio("Show", ["Single pick", "Top-N report"],
                    horizontal=True)
    if mode == "Single pick":
        default = picks[0].ticker
        sym = st.selectbox(
            "Pick from the list",
            [p.ticker for p in sorted(picks, key=lambda p: p.long_score,
                                      reverse=True)],
            index=0)
        custom = st.text_input("…or any ticker (US or .TO)", value="")
        custom = md.normalize_symbol(custom)
        sym = custom or sym
        if sym:
            p = _lookup_pick(sym)
            if p is None:
                st.warning("No data for that ticker "
                           "(use .TO suffix for TSX names).")
            else:
                prof = profile.fetch_business_profile(sym)
                if prof is None:
                    st.warning("Profile fetch failed (rate limit?) - "
                               "try again in a minute.")
                else:
                    card.render_business_card(p, prof, portfolio)
                    if p.pair:
                        st.markdown(f"**Two-job pair:** {p.pair['note']} "
                                    f"(peer X-Ray {p.pair['xray']})")
                    if p.reasons:
                        with st.expander("Why it scored this way"):
                            st.markdown("- " + "\n- ".join(p.reasons))
                    ed = md.earnings_date(sym)
                    if ed:
                        st.caption(f"Next earnings: {ed} - size options "
                                   "trades around it")
    else:
        n = st.slider("Cards for top N long-term picks", 1, 20, 5)
        if st.button("Generate cards"):
            top_picks = sorted(picks, key=lambda p: p.long_score,
                               reverse=True)[:n]
            prog = st.progress(0)
            for i, p in enumerate(top_picks):
                prof = profile.fetch_business_profile(p.ticker)
                if prof:
                    card.render_business_card(p, prof, portfolio)
                else:
                    st.warning(f"{p.ticker}: profile unavailable "
                               "(rate limit - retry later)")
                prog.progress((i + 1) / len(top_picks))
            prog.empty()

with tab_doc:
    st.markdown(METHODOLOGY)
