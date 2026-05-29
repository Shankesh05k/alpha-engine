import pandas as pd
import numpy as np
import os
import time
from pytrends.request import TrendReq

# ── Search terms per company ───────────────────────────────────────────────
# Use the most common way Indians search for these stocks
SEARCH_TERMS = {
    'TCS.NS':        'TCS share price',
    'INFY.NS':       'Infosys share price',
    'WIPRO.NS':      'Wipro share price',
    'HCLTECH.NS':    'HCL Technologies share price',
    'TECHM.NS':      'Tech Mahindra share price',
    'LTIM.NS':       'LTIMindtree share price',
    'MPHASIS.NS':    'Mphasis share price',
    'PERSISTENT.NS': 'Persistent Systems share price',
    'COFORGE.NS':    'Coforge share price',
    'KPITTECH.NS':   'KPIT share price',
    'TATAELXSI.NS':  'Tata Elxsi share price',
    'BSOFT.NS':      'Birlasoft share price',
    'MASTEK.NS':     'Mastek share price',
    'ZENSARTECH.NS': 'Zensar share price',
    'SONATSOFTW.NS': 'Sonata Software share price',
    'INTELLECT.NS':  'Intellect Design share price',
    'HAPPSTMNDS.NS': 'Happiest Minds share price',
    'CYIENT.NS':     'Cyient share price',
    'LTTS.NS':       'L&T Technology share price',
    'OFSS.NS':       'Oracle Financial Services share price',
}

def fetch_trends(ticker, term, start_date='2021-01-01'):
    """
    Fetch weekly Google Trends data for one search term.
    Returns a DataFrame with columns: week, trends_raw
    geo='IN' restricts to India searches only.
    """
    pytrends = TrendReq(hl='en-IN', tz=330)  # tz=330 is IST

    try:
        pytrends.build_payload(
            [term],
            cat=0,
            timeframe=f'{start_date} {pd.Timestamp.today().date()}',
            geo='IN',
            gprop=''
        )
        df = pytrends.interest_over_time()

        if df.empty:
            print(f"  {ticker}: no data returned")
            return None

        df = df.reset_index()[['date', term]]
        df.columns = ['week', 'trends_raw']
        df['ticker'] = ticker
        df['week'] = pd.to_datetime(df['week'])
        return df

    except Exception as e:
        print(f"  {ticker}: error — {e}")
        return None


def add_trends_features(df):
    """
    From raw trends index (0-100), compute:
    - trends_zscore: normalised vs company's own baseline
    - trends_velocity: week over week change
    - trends_spike: is this week unusually high (zscore > 1.5)?
    """
    df = df.sort_values(['ticker', 'week']).copy()

    def normalise(grp):
        grp = grp.sort_values('week').copy()
        roll_mean = grp['trends_raw'].rolling(12, min_periods=4).mean()
        roll_std  = grp['trends_raw'].rolling(12, min_periods=4).std()
        grp['trends_zscore']   = (grp['trends_raw'] - roll_mean) / (roll_std + 1e-9)
        grp['trends_velocity'] = grp['trends_raw'].diff()
        grp['trends_spike']    = (grp['trends_zscore'] > 1.5).astype(int)
        return grp

    df = df.groupby('ticker', group_keys=False).apply(normalise)
    return df


def fetch_all(start_date='2021-01-01'):
    all_dfs = []

    for ticker, term in SEARCH_TERMS.items():
        print(f"Fetching trends for {ticker}: '{term}'...")
        df = fetch_trends(ticker, term, start_date)
        if df is not None:
            all_dfs.append(df)
            print(f"  ✓ {len(df)} weeks of data")
        # Polite delay — Google rate limits pytrends aggressively
        time.sleep(5)

    if not all_dfs:
        print("No data fetched.")
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = add_trends_features(combined)

    os.makedirs('data/processed', exist_ok=True)
    combined.to_csv('data/processed/google_trends.csv', index=False)
    print(f"\n✓ Saved to data/processed/google_trends.csv")
    print(f"  Shape: {combined.shape}")
    print(f"  Tickers: {combined['ticker'].nunique()}")
    print(f"  Date range: {combined['week'].min().date()} → {combined['week'].max().date()}")
    return combined


def plot_trends(df):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw trends for top 5 companies
    ax = axes[0]
    top5 = ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'PERSISTENT.NS', 'KPITTECH.NS']
    for ticker in top5:
        t = df[df['ticker'] == ticker].sort_values('week')
        ax.plot(t['week'], t['trends_raw'],
                label=ticker.replace('.NS', ''), linewidth=1)
    ax.set_title('Google Trends — raw search interest (India)')
    ax.set_ylabel('Search interest (0-100)')
    ax.legend(fontsize=8)

    # Z-score latest week
    ax = axes[1]
    latest = df[df['week'] == df['week'].max()].sort_values(
        'trends_zscore', ascending=True)
    colors = ['#1D9E75' if v > 0 else '#E24B4A'
              for v in latest['trends_zscore']]
    ax.barh(latest['ticker'].str.replace('.NS', ''),
            latest['trends_zscore'], color=colors)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(1.5, color='orange', linestyle=':', linewidth=1,
               label='Spike threshold')
    ax.set_title('Google Trends z-score — latest week')
    ax.set_xlabel('Z-score vs own baseline')
    ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/google_trends.png', dpi=150)
    print("✓ Saved to outputs/google_trends.png")
    plt.show()


if __name__ == '__main__':
    # Install pytrends if needed: pip install pytrends
    df = fetch_all(start_date='2021-01-01')
    if df is not None:
        plot_trends(df)