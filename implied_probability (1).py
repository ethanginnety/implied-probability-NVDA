"""
Implied Probability Distribution from Options Prices
------------------------------------------------------
What this does:
  1. Pulls a live options chain for a chosen ticker (via yfinance)
  2. Picks an expiry roughly N days out
  3. Finds the at-the-money implied volatility (averaging the nearest call & put)
  4. Uses the Black-Scholes assumption that terminal prices are lognormally
     distributed to build the market's IMPLIED probability distribution
     for where the stock will be at expiry
  5. Compares that to the stock's own historical realized volatility over
     a similar lookback window, plotting both distributions side by side
  6. Prints concrete probability statements, e.g. "the market implies a
     14% chance AAPL is above $220 by March expiry"

Run this locally (needs live internet access to Yahoo Finance).
Install deps first:  pip install yfinance scipy matplotlib numpy --break-system-packages
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from datetime import datetime

# ----------------------------
# CONFIG — change these
# ----------------------------
TICKER = "NVDA"          # any liquid, well-optioned name works
TARGET_DAYS_OUT = 40     # roughly how far out you want the expiry to be
RISK_FREE_RATE = 0.045   # approx current short-term rate; update as needed
THRESHOLDS_PCT = [-0.10, -0.05, 0.05, 0.10]  # +/-5% and +/-10% moves to report on


def pick_expiry(ticker_obj, target_days):
    """Pick the listed expiry closest to `target_days` calendar days out."""
    expiries = ticker_obj.options
    today = datetime.today()
    diffs = [(abs((datetime.strptime(e, "%Y-%m-%d") - today).days - target_days), e)
              for e in expiries]
    diffs.sort()
    best_expiry = diffs[0][1]
    days_out = (datetime.strptime(best_expiry, "%Y-%m-%d") - today).days
    return best_expiry, days_out


def get_atm_iv(ticker_obj, expiry, spot):
    """
    Find the strike closest to the current spot price and average the
    call + put implied volatility at that strike. Falls back to using
    whichever side has data if one side is missing/illiquid.
    """
    chain = ticker_obj.option_chain(expiry)
    calls, puts = chain.calls, chain.puts

    calls = calls.copy()
    puts = puts.copy()
    calls["dist"] = (calls["strike"] - spot).abs()
    puts["dist"] = (puts["strike"] - spot).abs()

    atm_call = calls.sort_values("dist").iloc[0]
    atm_put = puts.sort_values("dist").iloc[0]

    ivs = []
    if atm_call["impliedVolatility"] > 0:
        ivs.append(atm_call["impliedVolatility"])
    if atm_put["impliedVolatility"] > 0:
        ivs.append(atm_put["impliedVolatility"])

    if not ivs:
        raise ValueError("No usable implied volatility found near the money — "
                          "try a more liquid ticker or a different expiry.")

    return float(np.mean(ivs)), float(atm_call["strike"])


def build_lognormal(spot, iv, T, r):
    """
    Black-Scholes risk-neutral assumption: ln(S_T) ~ Normal(mu, sigma^2)
    with mu chosen so E[S_T] = forward price = S0 * exp(rT).
    Returns a scipy lognorm distribution object.
    """
    forward = spot * np.exp(r * T)
    mu = np.log(forward) - 0.5 * (iv ** 2) * T
    sigma = iv * np.sqrt(T)
    dist = lognorm(s=sigma, scale=np.exp(mu))
    return dist, forward


def historical_realized_vol(ticker_obj, lookback_days=252, horizon_days=30):
    """
    Pull historical daily closes, compute annualized realized volatility,
    then build a comparison lognormal distribution assuming the SAME
    horizon as the option (so it's an apples-to-apples comparison).
    """
    hist = ticker_obj.history(period=f"{lookback_days}d")
    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    daily_vol = log_returns.std()
    annualized_vol = daily_vol * np.sqrt(252)
    return float(annualized_vol)


def summarize_probabilities(dist, spot, thresholds_pct):
    """Print P(S_T above/below spot*(1+pct)) for each threshold requested."""
    print("\nImplied probabilities from OPTIONS pricing:")
    for pct in thresholds_pct:
        level = spot * (1 + pct)
        if pct >= 0:
            p = 1 - dist.cdf(level)
            print(f"  P(price > {level:,.2f}, i.e. {pct:+.0%} move)  = {p:.1%}")
        else:
            p = dist.cdf(level)
            print(f"  P(price < {level:,.2f}, i.e. {pct:+.0%} move)  = {p:.1%}")


def main():
    t = yf.Ticker(TICKER)
    spot = t.history(period="1d")["Close"].iloc[-1]

    expiry, days_out = pick_expiry(t, TARGET_DAYS_OUT)
    T = days_out / 365.0

    implied_vol, atm_strike = get_atm_iv(t, expiry, spot)
    print(f"Ticker: {TICKER}")
    print(f"Spot price: {spot:.2f}")
    print(f"Chosen expiry: {expiry}  ({days_out} days out)")
    print(f"ATM strike used: {atm_strike}")
    print(f"At-the-money implied volatility: {implied_vol:.1%}")

    implied_dist, forward = build_lognormal(spot, implied_vol, T, RISK_FREE_RATE)
    print(f"Forward price (risk-neutral mean): {forward:.2f}")

    summarize_probabilities(implied_dist, spot, THRESHOLDS_PCT)

    # --- Historical realized vol comparison over a similar horizon ---
    realized_vol = historical_realized_vol(t, lookback_days=252, horizon_days=days_out)
    realized_dist, _ = build_lognormal(spot, realized_vol, T, RISK_FREE_RATE)
    print(f"\nHistorical realized volatility (annualized, trailing 1yr): {realized_vol:.1%}")
    print("(vs. implied volatility priced in the options market above)")
    summarize_probabilities(realized_dist, spot, THRESHOLDS_PCT)

    # --- Plot both distributions ---
    price_range = np.linspace(spot * 0.6, spot * 1.6, 500)
    implied_pdf = implied_dist.pdf(price_range)
    realized_pdf = realized_dist.pdf(price_range)

    plt.figure(figsize=(9, 5.5))
    plt.plot(price_range, implied_pdf, label=f"Market-implied (IV={implied_vol:.1%})", linewidth=2)
    plt.plot(price_range, realized_pdf, label=f"Historical realized (vol={realized_vol:.1%})",
              linewidth=2, linestyle="--")
    plt.axvline(spot, color="grey", linestyle=":", label=f"Current price ({spot:.2f})")
    plt.title(f"{TICKER}: Implied vs. Historical Probability Distribution\n"
              f"at {expiry} expiry ({days_out} days out)")
    plt.xlabel("Price at expiry")
    plt.ylabel("Probability density")
    plt.legend()
    plt.tight_layout()
    out_path = f"{TICKER}_implied_probability.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved chart to {out_path}")


if __name__ == "__main__":
    main()
