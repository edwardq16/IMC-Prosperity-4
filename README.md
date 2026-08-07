# IMC Prosperity 4 — Algorithmic Trading Strategies
Organising Subcommittee

Algorithmic trading strategies written for **IMC Prosperity 4**, IMC Trading's global trading-simulation challenge for university students. The competition takes players through a deep-space-themed simulated market across five rounds, each with an algorithmic and a manual trading challenge, with the goal of maximising profit in the game's in-universe currency. Teams submit a Python `Trader` class each round, run against a simulated order book and scored on realised P&L.

**Result: overall rank #3,105 of 18,803 teams (#30,703 individual players); #2,498 in the algorithmic challenge.**

## Approach

- **Arbitrage** against a known or estimated fair value — buying below and selling above it whenever the order book offered a price outside that band.
- **Market making** — quoting both sides of the book around a fair price (sometimes adjusted for order book imbalance or current inventory), sized and skewed to manage position risk.
- **Implied volatility options trading** — for a chain of options on one product, back out each option's market-implied volatility, fit a smile across strikes, and trade options whose market-implied vol deviates from the fitted smile, while delta-hedging the resulting exposure in the underlying.

`datamodel.py` is the competition-provided data model — `TradingState`, `Order`, `OrderDepth`, and related classes — that every `Trader.run()` method receives and returns each round.

## Tech stack

Python, NumPy, SciPy, pandas, Matplotlib.
