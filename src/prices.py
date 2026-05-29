import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os

# Your 20 Nifty IT stocks + the index
TICKERS = [
    'TCS.NS',
    'INFY.NS',
    'WIPRO.NS',
    'HCLTECH.NS',
    'TECHM.NS',
    'LTIMINDTREE.NS',
    'MPHASIS.NS',
    'PERSISTENT.NS',
    'COFORGE.NS',
    'KPITTECH.NS',
    'TATAELXSI.NS',
    'BSOFT.NS',
    'MASTEK.NS',
    'ZENSARTECH.NS',
    'SONATSOFTW.NS',
    'INTELLECT.NS',
    'HAPPSTMNDS.NS',
    'CYIENT.NS',
    'LTTS.NS',
    'OFSS.NS',
    '^CNXIT'
]

def fetch_and_save(start='2021-01-01'):
    print("Downloading price data...")
    raw = yf.download(TICKERS, start=start, auto_adjust=True, progress=True)
    prices = raw['Close']

    # Basic sanity checks
    print(f"\nShape: {prices.shape}")
    print(f"Date range: {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"\nMissing values per ticker:")
    print(prices.isnull().sum())

    # Save
    os.makedirs('data/raw', exist_ok=True)
    prices.to_csv('data/raw/prices.csv')
    print("\n✓ Saved to data/raw/prices.csv")
    return prices

def plot_cumulative(prices):
    # Normalise to 100 at start
    normalised = prices.drop(columns=['^CNXIT']).dropna(how='all')
    normalised = (normalised / normalised.iloc[0]) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in normalised.columns:
        ax.plot(normalised.index, normalised[col], label=col.replace('.NS',''), linewidth=1.2)

    ax.set_title('Nifty IT stocks — normalised to 100 at start', fontsize=13)
    ax.set_ylabel('Price (indexed)')
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/price_chart.png', dpi=150)
    print("✓ Chart saved to outputs/price_chart.png")
    plt.show()

if __name__ == '__main__':
    prices = fetch_and_save()
    plot_cumulative(prices)