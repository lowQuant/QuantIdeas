import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Download Historical Data
ticker = "LULU"
end_date = "2025-01-02"
start_date = "2021-01-02"  # Approximately four years prior

df = yf.download(ticker, start=start_date, end=end_date)
df.dropna(subset=["Close"], inplace=True)

# 2. Compute Log Returns
df["LogRet"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(subset=["LogRet"], inplace=True)

# ------------------------------------------------------------------------------
# Hodges–Tompkins Adjustment Factor
#    For overlapping returns, the variance is inflated. Hodges & Tompkins (2002)
#    provide a correction factor m for the *variance*, i.e. var_corrected = m * var_raw
#    which is:
#        m = 1 / ( 1 - (h / n) + (h^2 - 1) / (3n^2) )
#    where:
#        h = length of subseries (window size)
#        n = number of distinct subseries in the total T (n = T - h + 1)
#
#    Because this factor is for *variance*, we apply sqrt(m) to adjust volatility.
# ------------------------------------------------------------------------------

def hodges_tompkins_factor(h, total_obs):
    """
    Returns the factor by which the *variance* must be multiplied.
    For volatility (std), you should multiply by sqrt(this_factor).
    """
    n = total_obs - h + 1
    if n <= 0:
        return 1.0  # No adjustment if invalid
    denom = 1.0 - (h / n) + (h**2 - 1.0) / (3.0 * n**2)
    # If denom is very close to zero or negative, just return 1.0 to avoid blow-ups
    if denom <= 0:
        return 1.0
    return 1.0 / denom


# 3. Compute Rolling (Overlapping) Volatility for Each Window with Adjustment
windows = [20, 40, 60, 120, 240]  # ~1mo, 2mo, 3mo, 6mo, 1yr
total_obs = len(df["LogRet"])

vol_data = {}
for w in windows:
    ht_factor_var = hodges_tompkins_factor(w, total_obs)
    ht_factor_vol = np.sqrt(ht_factor_var)
    
    rolling_vol = (df["LogRet"].rolling(window=w).std().dropna()
        * np.sqrt(252)  # annualize
        * ht_factor_vol  # apply H–T correction
    )
    vol_data[w] = rolling_vol.values  # store as numpy array


# 4. Calculate the Summary Statistics for Each Window
stats = {}
for w in windows:
    arr = vol_data[w]
    if len(arr) > 0:
        stats[w] = {
            "Min":     np.min(arr),
            "Q1":      np.percentile(arr, 25),
            "Median":  np.median(arr),
            "Q3":      np.percentile(arr, 75),
            "Max":     np.max(arr),
        }
    else:
        # If no data, fill with NaN
        stats[w] = dict.fromkeys(["Min","Q1","Median","Q3","Max"], np.nan)

stats_df = pd.DataFrame(stats).T  # index=window-length, columns=stats

# ------------------------------------------------------------------------------
# 5. Plot the Volatility "Cone"
# ------------------------------------------------------------------------------
plt.figure(figsize=(8,5))

plt.plot(stats_df.index, stats_df["Max"],    color="darkgray",  label="Max")
plt.plot(stats_df.index, stats_df["Q3"],     color="gray",      label="75%ile")
plt.plot(stats_df.index, stats_df["Median"], color="black",     label="Median")
plt.plot(stats_df.index, stats_df["Q1"],     color="gray",      label="25%ile")
plt.plot(stats_df.index, stats_df["Min"],    color="darkgray",  label="Min")

plt.title(f"Volatility Cone for {ticker}")
plt.xlabel("Measurement Window (Trading Days)")
plt.ylabel("Annualized Volatility")
plt.ylim(0, 1) 
plt.legend()
plt.grid(True)
plt.show()
