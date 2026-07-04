"""
Pull the macro/financial series used as regime-classification signals
(NOT held as portfolio assets -- see checklist Section 3 note on why an
all-equity portfolio still uses these for regime labeling).

NOTE: This sandbox cannot reach the FRED API directly. Run this on your
own machine.

Get a free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html

Usage:
    pip install fredapi pandas
    python pull_fred_data.py YOUR_API_KEY
"""

import sys
import pandas as pd
from fredapi import Fred

SERIES = {
    "VIXCLS": "VIX (volatility)",
    "DGS10": "10-year Treasury yield",
    "DGS2": "2-year Treasury yield",
    "T10Y2Y": "10y-2y yield curve slope",
    "DTWEXBGS": "Trade-weighted dollar index",
    "CPIAUCSL": "CPI (inflation)",
    "BAMLH0A0HYM2": "High-yield credit spread",
}

START_DATE = "1995-01-01"
OUTPUT_PATH = "../data/fred_regime_signals.csv"


def pull_fred_series(api_key):
    fred = Fred(api_key=api_key)
    frames = {}

    print("Pulling FRED series...")
    for code, description in SERIES.items():
        try:
            series = fred.get_series(code, observation_start=START_DATE)
            frames[code] = series
            print(f"  {code} ({description}): {series.index.min().date()} to "
                  f"{series.index.max().date()}, {len(series.dropna())} obs")
        except Exception as e:
            print(f"  {code}: FAILED -- {e}")

    df = pd.DataFrame(frames)
    # Resample everything to monthly (some series are daily, CPI is already monthly)
    df_monthly = df.resample("MS").last()
    df_monthly.to_csv(OUTPUT_PATH)
    print(f"\nSaved monthly regime signals to {OUTPUT_PATH}")
    return df_monthly


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pull_fred_data.py YOUR_FRED_API_KEY")
        sys.exit(1)
    pull_fred_series(sys.argv[1])
