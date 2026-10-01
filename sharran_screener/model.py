"""The Sharran model: transcripts -> quantified scoring.

Sources (Business School podcast / sharran.com):

* Ep 258 & 302 - "Investment X-Ray" scorecard. Every investment is graded
  /25 on four buckets - capital preservation, tax efficiency, cash flow
  (yield) and growth - for a total /100.
* Ep 333 - portfolio rules: 20% opportunistic cash, no position > 5%,
  no theme > 10%, hold 7-10 positions, the "two-job rule" (every position
  should also enable a pair trade / tax-loss swap), weekly-monthly-quarterly
  cadence, buy when others are fearful ("2001: bought Apple at $5").
* Ep 316 - own quality companies for decades; the filter is a repeatable
  scorecard, not a hot tip.
* LinkedIn (Mar 2025) - "buy the dip: volatility makes great companies look
  cheap; follow global growth; ride the narrative".

Every scoring function returns (score, reasons) so the dashboard can show
exactly why a stock passed - the tool should be auditable like Sharran's
paper scorecard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BENCH_VALUES = {"SPY", "^GSPTSE"}

# ---------------------------------------------------------------------------
# X-Ray scorecard (Ep 258/302): four buckets x 25 pts
# ---------------------------------------------------------------------------

def _num(v, default=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def score_capital_preservation(tech: dict, fund: dict) -> tuple[float, list[str]]:
    """/25 - probability of getting principal back. Size, profitability,
    low beta, low leverage, low daily volatility."""
    pts, why = 0.0, []
    mc = _num(fund.get("marketCap"))
    pm = _num(fund.get("profitMargins"))
    beta = _num(tech.get("beta"))
    if np.isnan(beta):
        beta = _num(fund.get("beta"))
    de = _num(fund.get("debtToEquity"))
    atrp = _num(tech.get("atr_pct"))
    sector = (fund.get("sector") or "").lower()

    if mc >= 200e9:
        pts += 8; why.append("mega-cap ($200B+)")
    elif mc >= 50e9:
        pts += 7; why.append("large cap ($50B+)")
    elif mc >= 10e9:
        pts += 5; why.append("mid-large cap")
    elif mc >= 2e9:
        pts += 3; why.append("mid cap")
    elif mc > 0:
        pts += 1; why.append("small cap - capital at risk")

    if pm > 0.15:
        pts += 6; why.append(f"strong margins ({pm:.0%})")
    elif pm > 0.05:
        pts += 4; why.append(f"profitable ({pm:.0%} margin)")
    elif pm > 0:
        pts += 2; why.append("thin margins")
    else:
        why.append("unprofitable - fails quality gate")

    if not np.isnan(beta):
        if beta < 0.8:
            pts += 5; why.append(f"low beta ({beta:.2f})")
        elif beta < 1.2:
            pts += 4; why.append(f"market-like beta ({beta:.2f})")
        elif beta < 1.6:
            pts += 2; why.append(f"volatile beta ({beta:.2f})")
        else:
            pts += 1; why.append(f"high beta ({beta:.2f})")

    if "financial" in sector or np.isnan(de):
        pts += 3
    elif de < 50:
        pts += 4; why.append(f"low leverage (D/E {de:.0f})")
    elif de < 100:
        pts += 3
    elif de < 200:
        pts += 2; why.append(f"elevated leverage (D/E {de:.0f})")
    else:
        pts += 1; why.append(f"high leverage (D/E {de:.0f})")

    if not np.isnan(atrp):
        if atrp < 0.025:
            pts += 2; why.append("low daily volatility")
        elif atrp < 0.04:
            pts += 1

    return min(25.0, pts), why


def score_tax_efficiency(fund: dict) -> tuple[float, list[str]]:
    """/25 - Sharran: tax drag is the #1 wealth killer. For a stock this is
    mostly an account-location question, so score the *drag*: low/qualified
    dividends and buy-and-hold suitability score best."""
    dy = _num(fund.get("dividendYield")) or 0.0
    pts, why = 15.0, ["baseline: stock held long-term"]
    if dy == 0:
        pts += 7; why.append("no dividend -> no annual taxable income")
    elif dy < 0.01:
        pts += 6; why.append("minimal dividend tax drag")
    elif dy < 0.025:
        pts += 4; why.append("modest dividend (qualified -> lower drag)")
    else:
        pts += 1; why.append(
            f"high dividend {dy:.1%} -> yearly taxable income; "
            "consider holding in registered account"
        )
    pts += 3; why.append("no wash-sale / turnover drag if held (buy & hold)")
    return min(25.0, pts), why


def score_cash_flow(tech: dict, fund: dict) -> tuple[float, list[str]]:
    """/25 - yield: dividends + payout sustainability + FCF yield +
    covered-call income potential (he writes 30-45d covered calls)."""
    pts, why = 0.0, []
    dy = _num(fund.get("dividendYield")) or 0.0
    payout = _num(fund.get("payoutRatio"))
    mc = _num(fund.get("marketCap"))
    fcf = _num(fund.get("freeCashflow"))
    adv = _num(tech.get("avg_dollar_vol"))

    if dy >= 0.03:
        pts += 10; why.append(f"{dy:.1%} dividend yield")
    elif dy >= 0.015:
        pts += 7; why.append(f"{dy:.1%} dividend yield")
    elif dy > 0:
        pts += 4; why.append(f"small dividend ({dy:.1%})")
    else:
        why.append("no dividend yield")

    if dy > 0 and not np.isnan(payout):
        if 0 < payout < 0.6:
            pts += 5; why.append(f"sustainable payout ({payout:.0%})")
        elif payout < 1.0:
            pts += 3; why.append("high payout ratio - watch it")
        else:
            why.append("payout > 100% of earnings - unsustainable")

    if mc > 0 and not np.isnan(fcf) and fcf > 0:
        fcf_yield = fcf / mc
        if fcf_yield > 0.05:
            pts += 5; why.append(f"FCF yield {fcf_yield:.1%}")
        elif fcf_yield > 0.03:
            pts += 4; why.append(f"FCF yield {fcf_yield:.1%}")
        elif fcf_yield > 0.015:
            pts += 2

    if not np.isnan(adv) and adv > 50e6:
        pts += 5; why.append("liquid options market -> covered-call income")
    elif not np.isnan(adv) and adv > 10e6:
        pts += 3; why.append("options available for covered calls")

    return min(25.0, pts), why


def score_growth(tech: dict, fund: dict) -> tuple[float, list[str]]:
    """/25 - revenue/earnings growth, analyst upside, price momentum."""
    pts, why = 0.0, []
    rg = _num(fund.get("revenueGrowth"))
    eg = _num(fund.get("earningsGrowth"))
    if np.isnan(eg):
        eg = _num(fund.get("earningsQuarterlyGrowth"))
    tgt, px = _num(fund.get("targetMeanPrice")), _num(tech.get("last"))
    r12 = _num(tech.get("ret_12m"))

    if rg > 0.20:
        pts += 8; why.append(f"revenue +{rg:.0%}")
    elif rg > 0.10:
        pts += 6; why.append(f"revenue +{rg:.0%}")
    elif rg > 0.05:
        pts += 4
    elif rg > 0:
        pts += 2
    else:
        why.append("flat/declining revenue")

    if eg > 0.20:
        pts += 7; why.append(f"earnings +{eg:.0%}")
    elif eg > 0.10:
        pts += 5; why.append(f"earnings +{eg:.0%}")
    elif eg > 0:
        pts += 3

    if tgt > 0 and px > 0:
        upside = tgt / px - 1
        if upside > 0.25:
            pts += 5; why.append(f"analysts see +{upside:.0%}")
        elif upside > 0.10:
            pts += 3; why.append(f"analysts see +{upside:.0%}")
        elif upside > 0:
            pts += 1

    if r12 > 0.25:
        pts += 5; why.append(f"12m momentum +{r12:.0%}")
    elif r12 > 0:
        pts += 3
    else:
        pts += 1; why.append("negative 12m momentum")

    return min(25.0, pts), why


# ---------------------------------------------------------------------------
# Composite scores
# ---------------------------------------------------------------------------

@dataclass
class Pick:
    ticker: str
    market: str
    name: str
    sector: str
    industry: str
    last: float
    xray: float
    cp: float
    tax: float
    cf: float
    gr: float
    reasons: list[str] = field(default_factory=list)
    dip_score: float = 0.0
    short_score: float = 0.0
    long_score: float = 0.0
    entry_ladder: dict = field(default_factory=dict)
    entry_signal: str = ""
    pair: dict = field(default_factory=dict)
    tech: dict = field(default_factory=dict)
    fund: dict = field(default_factory=dict)


def xray_score(tech: dict, fund: dict) -> tuple[float, dict, list[str]]:
    """Return (total/100, {bucket: score}, combined reasons)."""
    cp, why_cp = score_capital_preservation(tech, fund)
    tx, why_tx = score_tax_efficiency(fund)
    cf, why_cf = score_cash_flow(tech, fund)
    gr, why_gr = score_growth(tech, fund)
    buckets = {"capital_preservation": cp, "tax_efficiency": tx,
               "cash_flow": cf, "growth": gr}
    reasons = why_cp + why_tx + why_cf + why_gr
    return cp + tx + cf + gr, buckets, reasons


def dip_opportunity(tech: dict, xray: float) -> tuple[float, list[str]]:
    """0-100: how 'on sale' a quality name is. Sharran deploys the 20%
    opportunistic cash when fear discounts good companies. Sweet spot is
    roughly -10% to -35% below the 52w high with quality intact."""
    pts, why = 0.0, []
    off = _num(tech.get("pct_off_high"))
    rsi = _num(tech.get("rsi14"))
    d200 = _num(tech.get("dist_sma200"))

    if np.isnan(off):
        return 0.0, ["no data"]
    if off <= -0.35:
        pts += 40; why.append(f"{off:.0%} below 52w high - deep discount")
    elif off <= -0.20:
        pts += 55; why.append(f"{off:.0%} below 52w high - big discount")
    elif off <= -0.10:
        pts += 45; why.append(f"{off:.0%} below 52w high - on sale")
    elif off <= -0.05:
        pts += 25; why.append(f"{off:.0%} below high - mild discount")
    else:
        pts += 10; why.append("near 52w high - full price")

    if not np.isnan(rsi):
        if rsi < 30:
            pts += 25; why.append(f"RSI {rsi:.0f} - capitulation zone")
        elif rsi < 40:
            pts += 18; why.append(f"RSI {rsi:.0f} - weak hands selling")
        elif rsi < 55:
            pts += 8
        else:
            why.append(f"RSI {rsi:.0f} - no fear discount")

    if not np.isnan(d200):
        if d200 < -0.15:
            pts += 10; why.append("well below 200dma - fear pricing")
        elif d200 < 0:
            pts += 5
        else:
            pts += 0

    if xray >= 60:
        pts += 10; why.append("quality intact - dip is opportunity, not value trap")
    elif xray < 45:
        pts -= 15; why.append("low quality - discount may be a value trap")

    return float(np.clip(pts, 0, 100)), why


def short_term_score(tech: dict, spy_3m: float | None) -> tuple[float, list[str]]:
    """0-100: 'ride the narrative / follow global growth' - momentum with
    confirmation, not chasing."""
    pts, why = 0.0, []
    r1, r3 = _num(tech.get("ret_1m")), _num(tech.get("ret_3m"))
    rsi = _num(tech.get("rsi14"))
    d50, d200 = _num(tech.get("dist_sma50")), _num(tech.get("dist_sma200"))
    vt = _num(tech.get("vol_trend"))
    off = _num(tech.get("pct_off_high"))

    if not np.isnan(r3):
        rel = r3 - (spy_3m or 0.0)
        if rel > 0.10:
            pts += 30; why.append(f"crushing SPY by {rel:.0%} (3m)")
        elif rel > 0.03:
            pts += 22; why.append(f"beating SPY by {rel:.0%} (3m)")
        elif rel > 0:
            pts += 14
        else:
            pts += 4; why.append("lagging the index (3m)")

    if not np.isnan(r1):
        if 0 < r1 < 0.15:
            pts += 15; why.append(f"steady 1m uptrend +{r1:.0%}")
        elif r1 >= 0.15:
            pts += 8; why.append("extended 1m move - chase risk")
        elif r1 > -0.05:
            pts += 6

    if not np.isnan(rsi):
        if 50 <= rsi <= 70:
            pts += 20; why.append(f"RSI {rsi:.0f} - strong, not overbought")
        elif 40 <= rsi < 50:
            pts += 10; why.append(f"RSI {rsi:.0f} - early momentum")
        elif rsi > 70:
            pts += 6; why.append(f"RSI {rsi:.0f} - overbought")
        else:
            why.append(f"RSI {rsi:.0f} - weak")

    if not np.isnan(d50) and not np.isnan(d200):
        if d50 > 0 and d200 > 0:
            pts += 20; why.append("above 50dma & 200dma - trend intact")
        elif d50 > 0:
            pts += 10; why.append("above 50dma")

    if not np.isnan(vt) and vt > 1.2:
        pts += 10; why.append("volume accelerating")
    if not np.isnan(off) and off > -0.03:
        pts += 5; why.append("pressing on 52w high")

    return float(np.clip(pts, 0, 100)), why


def entry_ladder(tech: dict) -> dict:
    """Sharran buys in ladders on weakness, never all at once. Zones:
      now   - market price if already discounted
      zone1 - retest of recent support / ~5% below market
      zone2 - ~20% below the 52w high (his 'crash discount' zone)
      zone3 - ~30% below the 52w high (2001-style fear pricing)
    """
    last = _num(tech.get("last"))
    hi52 = _num(tech.get("high_52w"))
    consol = _num(tech.get("consol_low_3m"))
    sup3 = _num(tech.get("support_3m"))
    if np.isnan(last) or np.isnan(hi52):
        return {}
    zone1 = max(x for x in [consol, sup3, last * 0.95] if not np.isnan(x))
    zone2 = hi52 * 0.80
    zone3 = hi52 * 0.70
    return {
        "market": round(last, 2),
        "zone1_support": round(min(zone1, last * 0.98), 2),
        "zone2_-20%": round(zone2, 2),
        "zone3_-30%": round(zone3, 2),
    }


def entry_signal(tech: dict, xray: float, ladder: dict) -> str:
    off = _num(tech.get("pct_off_high"))
    last = _num(tech.get("last"))
    z1 = ladder.get("zone1_support")
    if xray < 55:
        return "PASS - fails quality filter"
    if not np.isnan(off) and off <= -0.20:
        return "DEPLOY CASH - deep discount on quality"
    if not np.isnan(off) and off <= -0.10:
        return "BUY ZONE - on sale vs 52w high"
    if z1 and last <= z1 * 1.03:
        return "NEAR SUPPORT - limit orders at zone 1"
    return "WAIT - set limit orders at zones below"


def explain_pick(p: "Pick") -> list[str]:
    """Five plain-English bullets on why the signal fired."""
    b: list[str] = []
    buckets = {"capital preservation": p.cp, "tax efficiency": p.tax,
               "cash flow": p.cf, "growth": p.gr}
    best = max(buckets, key=buckets.get)
    worst = min(buckets, key=buckets.get)
    b.append(f"It scores {p.xray}/100 on his Investment X-Ray - strongest "
             f"on {best} ({buckets[best]:.0f}/25), weakest on {worst} "
             f"({buckets[worst]:.0f}/25).")

    off = _num(p.tech.get("pct_off_high"))
    if not np.isnan(off):
        b.append(f"Price is {abs(off):.0%} "
                 + ("below" if off < 0 else "above") +
                 f" its 52-week high - " +
                 ("a real 'greedy when others are fearful' discount."
                  if off <= -0.15 else
                  "a mild discount." if off <= -0.05 else
                  "near the top of its range - full price."))

    rsi = _num(p.tech.get("rsi14"))
    if not np.isnan(rsi):
        b.append(f"Sentiment: RSI {rsi:.0f} - " +
                 ("sold hard, fear is in the price." if rsi < 35 else
                  "calm-to-strong momentum." if rsi < 65 else
                  "hot/overbought - chasing risk."))

    lad = p.entry_ladder
    if lad:
        b.append(f"His style is laddered limit orders, not one market buy: "
                 f"zone 1 ~${lad.get('zone1_support')}, deeper at "
                 f"${lad.get('zone2_-20%')} and ${lad.get('zone3_-30%')}.")

    verdict = {
        "DEPLOY CASH": "Verdict: quality name at a deep discount - this is "
                       "what the 20% opportunistic cash pile is for.",
        "BUY ZONE": "Verdict: on sale vs its high - reasonable to start a "
                    "position, keep powder dry for deeper zones.",
        "NEAR SUPPORT": "Verdict: sitting near recent support - "
                        "limit orders at zone 1.",
        "WAIT": "Verdict: not cheap enough yet - rest limit orders at "
                "the zones and let the market come to you.",
        "PASS": "Verdict: fails his quality filter - skip regardless of "
                "price action.",
    }
    for k, v in verdict.items():
        if p.entry_signal.startswith(k):
            b.append(v)
            break
    return b[:5]


def position_sizing(portfolio_value: float, price: float) -> dict:
    """Ep 333 rules: 20% opportunistic cash, position <=5%, theme <=10%."""
    return {
        "opportunistic_cash": round(portfolio_value * 0.20, 2),
        "max_position_$": round(portfolio_value * 0.05, 2),
        "max_theme_$": round(portfolio_value * 0.10, 2),
        "max_shares": int(portfolio_value * 0.05 // price) if price else 0,
    }


# ---------------------------------------------------------------------------
# Screening orchestration
# ---------------------------------------------------------------------------

MIN_DOLLAR_VOL = 10e6      # must be liquid enough to size 5% positions
MIN_PRICE = 5.0            # no penny stocks - quality filter
SHORTLIST_N = 140          # fundamentals fetched only for the shortlist


def _prefilter_key(tech: dict) -> float:
    """Cheap ranking before paying for fundamentals: liquidity + a blend of
    dip-ness and momentum so both long-term and short-term buckets survive."""
    if (tech.get("avg_dollar_vol") or 0) < MIN_DOLLAR_VOL:
        return -1
    if (tech.get("last") or 0) < MIN_PRICE:
        return -1
    off = tech.get("pct_off_high") or 0
    r3 = tech.get("ret_3m") or -0.5
    return -min(off, 0) * 2 + r3  # bigger dip + better momentum ranks higher


def find_pair(ticker: str, industry: str, picks: list[Pick]) -> dict:
    """Two-job rule: every pick should pair against a same-industry peer.

    Returns the closest peer with its scores - Sharran pits e.g. HD vs LOW,
    KO vs PEP. If our pick dominates the peer, the pair thesis is
    'long pick / avoid-or-short peer'; it is also the tax-loss-swap
    candidate if the pick later drops."""
    same_ind = [
        p for p in picks
        if p.ticker != ticker and p.industry == industry
        and industry not in ("Unknown", "") and p.fund.get("marketCap")
    ]
    if not same_ind:
        return {}
    peer = min(same_ind, key=lambda p: p.xray)  # weakest same-industry peer
    return {
        "ticker": peer.ticker,
        "name": peer.name,
        "xray": round(peer.xray, 1),
        "note": (
            f"long {ticker} / avoid-or-pair {peer.ticker} ({industry})"
            if peer.xray else f"peer {peer.ticker}"
        ),
    }


def build_picks(
    techs: dict[str, dict],
    funds: dict[str, dict],
    market_map: dict[str, str],
    spy_3m: float | None,
) -> list[Pick]:
    picks: list[Pick] = []
    for sym, tech in techs.items():
        if sym in BENCH_VALUES:
            continue
        fund = funds.get(sym) or {}
        xr, buckets, reasons = xray_score(tech, fund)
        dip, dip_why = dip_opportunity(tech, xr)
        st, st_why = short_term_score(tech, spy_3m)
        ladder = entry_ladder(tech)
        signal = entry_signal(tech, xr, ladder)
        long_term = np.clip(0.6 * xr + 0.4 * dip, 0, 100)
        picks.append(Pick(
            ticker=sym,
            market=market_map.get(sym, "?"),
            name=fund.get("shortName") or sym,
            sector=fund.get("sector") or "Unknown",
            industry=fund.get("industry") or "Unknown",
            last=round(_num(tech.get("last")), 2),
            xray=round(xr, 1),
            cp=round(buckets["capital_preservation"], 1),
            tax=round(buckets["tax_efficiency"], 1),
            cf=round(buckets["cash_flow"], 1),
            gr=round(buckets["growth"], 1),
            reasons=reasons + dip_why,
            dip_score=round(dip, 1),
            short_score=round(st, 1),
            long_score=round(float(long_term), 1),
            entry_ladder=ladder,
            entry_signal=signal,
            tech=tech,
            fund=fund,
        ))
    # second pass: pair suggestions need the full pick list
    for p in picks:
        p.pair = find_pair(p.ticker, p.industry, picks)
    return picks
