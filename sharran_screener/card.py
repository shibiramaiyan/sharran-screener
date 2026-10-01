"""Business-card infographic renderer (Streamlit).

Sharran's rule #1 - "I only invest in stuff that I know and understand" -
can't be automated. This renders a one-page profile per pick so the user can
answer it: what they sell, who pays them, how strong the balance sheet is,
and whether earnings have grown for the last ~4 years.
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

from . import model


def _esc(s: str) -> str:
    """Escape $ so Streamlit doesn't read it as LaTeX math delimiters."""
    return s.replace("$", r"\$")


def _money(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    a = abs(v)
    if a >= 1e12:
        return f"${v/1e12:,.2f}T"
    if a >= 1e9:
        return f"${v/1e9:,.1f}B"
    if a >= 1e6:
        return f"${v/1e6:,.0f}M"
    return f"${v:,.0f}"


def _pct(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.0%}"


def _balance_bar(assets: float, equity: float, debt: float) -> str:
    """HTML stacked bar: what funds the asset base - equity vs debt vs other."""
    if not assets or np.isnan(assets) or assets <= 0:
        return ""
    eq = max(0.0, equity) / assets if not np.isnan(equity) else 0.0
    db = max(0.0, debt) / assets if not np.isnan(debt) else 0.0
    other = max(0.0, 1 - eq - db)
    return (
        '<div style="display:flex;height:26px;border-radius:6px;'
        'overflow:hidden;font-size:11px;font-weight:600;color:#fff">'
        f'<div style="width:{eq*100:.1f}%;background:#1b7f3b;display:flex;'
        f'align-items:center;justify-content:center">equity {eq:.0%}</div>'
        f'<div style="width:{db*100:.1f}%;background:#b03030;display:flex;'
        f'align-items:center;justify-content:center">debt {db:.0%}</div>'
        f'<div style="width:{other*100:.1f}%;background:#6b6b6b;display:flex;'
        f'align-items:center;justify-content:center">other {other:.0%}</div>'
        "</div>"
    )


def render_business_card(pick, prof: dict, portfolio_value: float = 0.0) -> None:
    """One infographic card. `pick` may be a model.Pick or None."""
    esc = html.escape
    name = prof.get("name") or (pick.ticker if pick else "?")
    ticker = pick.ticker if pick else ""
    sector = prof.get("sector") or (pick.sector if pick else "")
    industry = prof.get("industry") or (pick.industry if pick else "")

    st.markdown(f"## {esc(name)} · `{ticker}`")
    st.caption(" · ".join(x for x in [
        sector, industry, prof.get("hq") or "",
        f"{prof['employees']:,} employees" if prof.get("employees") else "",
        prof.get("website") or "",
    ] if x))

    if pick:
        c = st.columns(5)
        c[0].metric("X-Ray /100", pick.xray)
        c[1].metric("Cap. pres.", pick.cp)
        c[2].metric("Cash flow", pick.cf)
        c[3].metric("Growth", pick.gr)
        c[4].metric("Dip score", pick.dip_score)
        st.markdown(f"**Signal:** {pick.entry_signal}")
        if pick.entry_ladder:
            st.markdown(
                "**Entry ladder:** " + " · ".join(
                    f"{k} = {v}" for k, v in pick.entry_ladder.items()))
        if portfolio_value and pick.last:
            n = int(portfolio_value * 0.05 // pick.last)
            st.markdown(_esc(f"**Sizing:** max {n:,} shares "
                             f"(${portfolio_value*0.05:,.0f} = 5% rule)"))
        expl = model.explain_pick(pick)
        if expl:
            st.markdown("**Why this signal — in plain English**")
            st.markdown("\n".join(f"- {_esc(x)}" for x in expl))

    left, right = st.columns([3, 2])

    with left:
        st.markdown("#### What they do & how they earn")
        st.markdown(_esc(prof.get("summary") or "_No description available._"))

        st.markdown("#### Earnings track record")
        rev, ni = prof.get("revenue_yr") or {}, prof.get("net_income_yr") or {}
        if rev:
            df = pd.DataFrame(
                {"Revenue": {k: v / 1e9 for k, v in rev.items()},
                 "Net income": {k: v / 1e9 for k, v in ni.items()}})
            st.bar_chart(df)
            st.caption("Revenue vs net income, fiscal years ($B)")
        else:
            st.caption("No earnings history available.")
        eps = prof.get("eps_yr") or {}
        fcf = prof.get("fcf_yr") or {}
        if eps:
            st.markdown("**EPS (diluted):** " + " · ".join(
                f"{y}: {v:+.2f}" for y, v in eps.items()))
        if fcf:
            st.markdown(_esc("**Free cash flow:** " + " · ".join(
                f"{y}: {_money(v)}" for y, v in fcf.items())))
        if len(rev) >= 2:
            yrs = sorted(rev)
            cagr = (rev[yrs[-1]] / rev[yrs[0]]) ** (1 / (yrs[-1] - yrs[0])) - 1 \
                if rev[yrs[0]] > 0 else float("nan")
            if not np.isnan(cagr):
                st.markdown(f"**Revenue CAGR ({yrs[0]}-{yrs[-1]}):** {cagr:.1%}")

    with right:
        st.markdown("#### Balance sheet - what funds the assets")
        assets = prof.get("assets")
        if assets and not np.isnan(assets):
            st.markdown(
                _balance_bar(assets, prof.get("equity"), prof.get("debt")),
                unsafe_allow_html=True)
            st.caption(_esc(f"Total assets {_money(assets)} split by "
                            "funding source"))
            m1, m2 = st.columns(2)
            m1.metric("Total debt", _money(prof.get("debt")))
            m2.metric("Cash", _money(prof.get("cash")))
            m3, m4 = st.columns(2)
            m3.metric("Debt / assets", _pct(prof.get("debt_to_assets")))
            dte = prof.get("debt_to_equity")
            m4.metric("Debt / equity",
                      "n/a" if dte is None or np.isnan(dte) else f"{dte:.2f}x")
            st.metric("Current ratio",
                      "n/a" if np.isnan(prof.get("current_ratio") or np.nan)
                      else f"{prof['current_ratio']:.2f}")
        else:
            st.caption("No balance-sheet data.")

        st.markdown("#### Ask yourself (his checklist)")
        st.markdown(
            "- Can I explain what they sell in one sentence?\n"
            "- Who pays them, and is it recurring revenue?\n"
            "- Would I buy more if it dropped another 30%?\n"
            "- Does it have a second job? (pair / tax-loss swap)")

    st.divider()
