import pandas as pd
import numpy as np
import os
from scipy import stats

def merge_all_signals():
    # Load job posting features
    features = pd.read_csv('data/processed/features.csv', parse_dates=['week'])
    print(f"✓ Job features: {features.shape}")

    # Load Google Trends — align to nearest month
    trends = pd.read_csv('data/processed/google_trends.csv', parse_dates=['week'])
    trends['month'] = trends['week'].dt.to_period('M')
    trends_slim = trends[['ticker', 'month', 'trends_raw',
                           'trends_zscore', 'trends_velocity', 'trends_spike']]
    print(f"✓ Google Trends: {trends_slim.shape}")

    # Add month to features for merging
    features['month'] = features['week'].dt.to_period('M')

    # Merge on ticker + month
    merged = pd.merge(features, trends_slim, on=['ticker', 'month'], how='left')
    print(f"  Trends coverage after merge: {merged['trends_zscore'].notna().sum()} rows")

    # Load prices for forward returns
    prices = pd.read_csv('data/raw/prices.csv', index_col=0, parse_dates=True)
    stock_cols = [c for c in prices.columns if c != '^CNXIT']

    fwd7  = prices[stock_cols].pct_change(7,  fill_method=None).shift(-7)
    fwd14 = prices[stock_cols].pct_change(14, fill_method=None).shift(-14)
    idx7  = prices['^CNXIT'].pct_change(7,  fill_method=None).shift(-7)
    idx14 = prices['^CNXIT'].pct_change(14, fill_method=None).shift(-14)

    excess7  = fwd7.sub(idx7,  axis=0)
    excess14 = fwd14.sub(idx14, axis=0)

    def get_return(excess_df, ticker, week, tol=5):
        col = ticker if ticker in excess_df.columns else None
        if col is None:
            candidates = [c for c in excess_df.columns
                          if ticker.replace('.NS', '') in c]
            col = candidates[0] if candidates else None
        if col is None:
            return np.nan
        t0  = pd.Timestamp(week)
        idx = excess_df.index[
            (excess_df.index >= t0) &
            (excess_df.index <= t0 + pd.Timedelta(days=tol))
        ]
        if len(idx) == 0:
            return np.nan
        val = excess_df.loc[idx[0], col]
        return float(val) if not pd.isna(val) else np.nan

    merged['fwd_excess_7d']  = merged.apply(
        lambda r: get_return(excess7,  r['ticker'], r['week']), axis=1)
    merged['fwd_excess_14d'] = merged.apply(
        lambda r: get_return(excess14, r['ticker'], r['week']), axis=1)

    # Drop helper column
    merged = merged.drop(columns=['month'])

    print(f"\n✓ Final merged shape: {merged.shape}")
    print(f"  7d return coverage:   {merged['fwd_excess_7d'].notna().sum()} rows")
    print(f"  14d return coverage:  {merged['fwd_excess_14d'].notna().sum()} rows")
    print(f"  Trends coverage:      {merged['trends_zscore'].notna().sum()} rows")

    os.makedirs('data/processed', exist_ok=True)
    merged.to_csv('data/processed/merged_all.csv', index=False)
    print(f"\n✓ Saved to data/processed/merged_all.csv")
    return merged


def quick_ic_check(merged):
    print("\n=== IC check — all signals ===")
    print(f"{'Feature':<30} {'IC_7d':>8} {'IC_14d':>8}")
    print("-" * 50)

    feature_cols = [
        'volume_zscore',
        'skill_ai_ml_share',
        'skill_cloud_share',
        'skill_digital_share',
        'seniority_score',
        'skill_ai_ml_velocity',
        'trends_zscore',
        'trends_velocity',
        'trends_spike',
    ]
    feature_cols = [f for f in feature_cols if f in merged.columns]

    for col in feature_cols:
        row = []
        for target in ['fwd_excess_7d', 'fwd_excess_14d']:
            valid = merged[[col, target]].dropna()
            if len(valid) < 5:
                row.append('  N/A')
                continue
            ic, pval = stats.spearmanr(valid[col], valid[target])
            star = '*' if pval < 0.15 else ' '
            row.append(f'{ic:+.3f}{star}')
        print(f"{col:<30} {row[0]:>8} {row[1]:>8}")

    print("\n* = p < 0.15")


if __name__ == '__main__':
    merged = merge_all_signals()
    quick_ic_check(merged)