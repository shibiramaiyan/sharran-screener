# Sharran Screener

A daily stock-analysis tool for US (Russell 1000) and Canadian (TSX
Composite) stocks, built to score investments the way **Sharran Srivatsaa**
describes in his Business School podcast interviews.

## Run it

```bash
/opt/homebrew/bin/python3.11 -m venv .venv     # once
.venv/bin/pip install -r requirements.txt      # once
.venv/bin/streamlit run app.py
```

Then press **Run daily screen** in the sidebar. The first run downloads
~1,300 tickers of daily price history (a few minutes); results are cached
in `data/cache/` for the rest of the day.

## What it does

| Tab | Sharran concept | Output |
|---|---|---|
| Long-Term | X-Ray scorecard (Ep. 258/302) + "buy the dip on quality" (Ep. 333) | Top 30 ranked by `0.6*X-Ray + 0.4*dip score`, entry ladder, pair trade |
| Short-Term | "Ride the narrative" momentum | Top 30 by relative strength, RSI 50-70, trend confirmation |
| Long Calls | His options-cycle approach, inverted for buying | Real chain scan: delta ~0.6, 45-200 DTE, tight spreads, real OI |
| Business Cards | Rule #1: "only invest in what you know and understand" | Per-company infographic: what they do, how they earn, assets-vs-debt breakdown, 4 years of revenue/EPS/FCF |
| Sidebar | Portfolio rules (Ep. 333) | 20% opportunistic cash, 5% position cap, 10% theme cap, cadence |

## Methodology sources

- **Ep. 333 – "My Exact System"**: 20% opportunistic cash; position ≤5%,
  theme ≤10%; 7-10 holdings; two-job/pair-trade rule (HD vs LOW); weekly /
  monthly / quarterly cadence; tax-loss harvesting; covered calls in
  30-45 day cycles inside tax-advantaged accounts.
- **Ep. 258 & 302 – "Investment X-Ray"**: score any investment /25 on
  capital preservation, tax efficiency, cash flow, growth → /100 total.
  He literally scores it on paper; this tool automates the scorecard.
- **Ep. 316**: quality companies held for decades; the filter matters more
  than the pick.
- **LinkedIn (Mar 2025)**: "buy the dip — volatility makes great companies
  look cheap"; follow global growth; ride the narrative.

### How each X-Ray bucket is quantified

- **Capital preservation /25**: market-cap tier, profit margin, beta,
  debt/equity, ATR% — probability you get principal back.
- **Tax efficiency /25**: dividend tax drag + buy-and-hold suitability.
  (The app also surfaces *where* to hold it — his real point in Ep. 333 is
  asset *location*: options trading in registered accounts, buy-and-hold
  in taxable.)
- **Cash flow /25**: dividend yield, payout sustainability, FCF yield,
  options liquidity for covered-call income.
- **Growth /25**: revenue growth, earnings growth, analyst-target upside,
  12-month price momentum.

### Entry price logic

He never buys all at once — he ladders into weakness with the 20% cash
reserve: Zone 1 = recent consolidation support, Zone 2 = -20% from the
52-week high, Zone 3 = -30%. The signal column says when to deploy vs
when to rest limit orders.

## Caveats

- Data is Yahoo Finance (free): delayed ~15min, occasionally rate-limited.
  Canadian names use `.TO` tickers in CAD.
- This is a screening aid, not financial advice; it mechanizes his public
  frameworks — judgment about *which* companies you understand (his #1
  rule: only invest in what you know) is still yours.
