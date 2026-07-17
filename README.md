# implied-probability-NVDA
To accompany the post on substack- https://inferenceletter.substack.com/p/what-the-options-market-is-actually?r=6gy5ci
 Implied Probability Distribution from Options Pricing

Extracts the market's implied probability distribution for a stock's future
price using options data, and compares it to the stock's own historical
realized volatility.

## What it does
- Pulls a live options chain via yfinance
- Finds the at-the-money implied volatility for a chosen expiry
- Builds the market-implied lognormal probability distribution (Black-Scholes)
- Compares it against trailing historical realized volatility
- Plots both distributions and prints concrete probability statements
  (e.g. "the market implies a 25% chance of a 10%+ move by expiry")

## Example
Applied to NVDA ahead of its Q2 FY2027 earnings report, this found the market
pricing 45.4% implied volatility versus 35.8% trailing realized volatility —
a meaningful gap, largely explained by the earnings event sitting inside the
expiry window used.

## Run it
pip install yfinance scipy matplotlib numpy
python implied_probability.py
