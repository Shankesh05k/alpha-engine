import requests
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta
from scipy import stats

# ── Company BSE codes ──────────────────────────────────────────────────────
BSE_CODES = {
    'TCS.NS':        '532540',
    'INFY.NS':       '500209',
    'WIPRO.NS':      '507685',
    'HCLTECH.NS':    '532281',
    'TECHM.NS':      '532755',
    'LTIMINDTREE.NS':'540005',
    'MPHASIS.NS':    '526299',
    'PERSISTENT.NS': '533179',
    'COFORGE.NS':    '532541',
    'KPITTECH.NS':   '542651',
    'TATAELXSI.NS':  '500408',
    'BSOFT.NS':      '532400',
    'MASTEK.NS':     '523704',
    'ZENSARTECH.NS': '504067',
    'SONATSOFTW.NS': '532221',
    'INTELLECT.NS':  '538835',
    'HAPPSTMNDS.NS': '543227',
    'CYIENT.NS':     '532175',
    'LTTS.NS':       '540115',
    'OFSS.NS':       '532466',
}

POSITIVE_KEYWORDS = [
    'order', 'contract', 'win', 'award', 'bagged', 'secured',
    'partnership', 'collaboration', 'agreement', 'deal', 'revenue',
    'growth', 'expansion', 'new client', 'multi-year', 'billion',
    'million', 'crore', 'selected', 'chosen', 'appointed',
    'large order', 'strategic', 'empanelled', 'empaneled',
    'tie-up', 'joint venture', 'jv', 'mou', 'memorandum',
    'preferred partner', 'exclusive', 'long term', 'long-term',
    'multi year', 'digital transformation', 'cloud migration'
]

NEGATIVE_KEYWORDS = [
    'loss', 'penalty', 'fine', 'litigation', 'lawsuit', 'dispute',
    'termination', 'cancelled', 'delay', 'downgrade', 'warning',
    'investigation', 'fraud', 'breach', 'layoff', 'restructur'
]

NEUTRAL_KEYWORDS = [
    'board meeting', 'dividend', 'agm', 'egm', 'annual report',
    'financial results', 'shareholding', 'record date'
]


def fetch_bse_announcements(security_code, from_date, to_date):
    url = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
    params = {
        'strCat':      '-1',
        'strPrevDate': from_date,
        'strScrip':    security_code,
        'strSearch':   'P',
        'strToDate':   to_date,
        'strType':     'C',
        'subcategory': '-1',
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer':    'https://www.bseindia.com/',
    }
    try:
        response = requests.get(url, params=params,
                                headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if 'Table' in data:
                return data['Table']
        return []
    except Exception as e:
        print(f"  Error: {e}")
        return []


def score_filing(headline, category=''):
    text = (headline + ' ' + category).lower()
    if any(k in text for k in NEUTRAL_KEYWORDS):
        return 0.0, 'neutral'
    pos_count = sum(1 for k in POSITIVE_KEYWORDS if k in text)
    neg_count = sum(1 for k in NEGATIVE_KEYWORDS if k in text)
    if pos_count == 0 and neg_count == 0:
        return 0.0, 'neutral'
    total = pos_count + neg_count
    score = (pos_count - neg_count) / total
    sentiment = ('positive' if score > 0
                 else 'negative' if score < 0
                 else 'neutral')
    return score, sentiment


def fetch_all_filings(days_back=365):
    to_date   = datetime.now().strftime('%Y%m%d')
    from_date = (datetime.now() -
                 timedelta(days=days_back)).strftime('%Y%m%d')
    all_filings = []

    for ticker, code in BSE_CODES.items():
        print(f"Fetching BSE filings for {ticker} (code: {code})...")
        filings = fetch_bse_announcements(code, from_date, to_date)

        if not filings:
            print(f"  No filings found")
            time.sleep(1)
            continue

        for f in filings:
            headline = (f.get('HEADLINE', '') or
                        f.get('NEWSSUB', '') or '')
            category = f.get('CATEGORYNAME', '') or ''
            date_str = (f.get('NEWS_DT', '') or
                        f.get('DISS_DT', '') or '')
            try:
                if date_str:
                    date = pd.to_datetime(date_str).date()
                else:
                    continue
            except Exception:
                continue

            score, sentiment = score_filing(headline, category)
            all_filings.append({
                'ticker':    ticker,
                'date':      date,
                'headline':  headline[:200],
                'category':  category,
                'score':     score,
                'sentiment': sentiment,
            })

        pos = sum(1 for f in all_filings
                  if f['ticker'] == ticker and f['score'] > 0)
        print(f"  ✓ {len(filings)} filings, {pos} positive")
        time.sleep(1)

    if not all_filings:
        print("No filings fetched.")
        return pd.DataFrame()

    df = pd.DataFrame(all_filings)
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.to_period('W').dt.start_time

    print(f"\n✓ Total filings: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} → "
          f"{df['date'].max().date()}")
    print(f"  Sentiment: {df['sentiment'].value_counts().to_dict()}")
    return df


def build_weekly_features(df):
    if df.empty:
        return pd.DataFrame()

    signal_df = df[df['sentiment'] != 'neutral'].copy()
    weekly = signal_df.groupby(['ticker', 'week']).agg(
        filing_count    = ('score', 'count'),
        sentiment_score = ('score', 'mean'),
        positive_count  = ('score', lambda x: (x > 0).sum()),
        negative_count  = ('score', lambda x: (x < 0).sum()),
    ).reset_index()

    weekly['positive_ratio'] = (
        weekly['positive_count'] /
        (weekly['filing_count'] + 1e-9)
    )
    weekly['big_news'] = (weekly['sentiment_score'] > 0.5).astype(int)

    def add_zscore(grp):
        grp = grp.sort_values('week').copy()
        roll_mean = grp['sentiment_score'].rolling(
            8, min_periods=2).mean()
        roll_std  = grp['sentiment_score'].rolling(
            8, min_periods=2).std()
        grp['filing_sentiment_zscore'] = (
            (grp['sentiment_score'] - roll_mean) /
            (roll_std + 1e-9)
        )
        return grp

    weekly = weekly.groupby(
        'ticker', group_keys=False
    ).apply(add_zscore)

    print(f"\n✓ Weekly filing features: {weekly.shape}")
    print(f"  Weeks: {weekly['week'].min().date()} → "
          f"{weekly['week'].max().date()}")
    print(f"\nSample — most positive weeks:")
    print(weekly.nlargest(5, 'sentiment_score')[[
        'ticker', 'week', 'filing_count',
        'sentiment_score', 'positive_count'
    ]].round(3).to_string(index=False))
    return weekly


def merge_filings(weekly_filings):
    merged = pd.read_csv(
        'data/processed/merged_all.csv', parse_dates=['week'])
    print(f"\n✓ Loaded merged_all.csv: {merged.shape}")

    # Drop old filing columns if they exist
    old_cols = ['filing_count', 'sentiment_score', 'positive_count',
                'negative_count', 'positive_ratio', 'big_news',
                'filing_sentiment_zscore']
    merged = merged.drop(
        columns=[c for c in old_cols if c in merged.columns],
        errors='ignore'
    )

    merged = pd.merge(merged, weekly_filings,
                      on=['ticker', 'week'], how='left')

    for col in old_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    print(f"  After merge: {merged.shape}")
    print(f"  Filing coverage: "
          f"{(merged['filing_count'] > 0).sum()} rows with filings")

    merged.to_csv('data/processed/merged_all.csv', index=False)
    print(f"✓ Saved updated merged_all.csv")
    return merged


def ic_check(merged):
    filing_features = [
        'sentiment_score', 'positive_ratio',
        'big_news', 'filing_sentiment_zscore', 'filing_count'
    ]
    filing_features = [f for f in filing_features
                       if f in merged.columns]
    valid = merged.dropna(subset=['fwd_excess_7d'])

    if len(valid) < 5:
        print("Not enough data for IC check.")
        return

    print(f"\n=== BSE Filing IC Analysis ===")
    print(f"{'Feature':<30} {'IC':>8} {'p-value':>10} {'N':>6}")
    print("-" * 58)

    for col in filing_features:
        v = valid[[col, 'fwd_excess_7d']].dropna()
        if len(v) < 5:
            print(f"{col:<30} {'N/A':>8} {'N/A':>10} {len(v):>6}")
            continue
        ic, pval = stats.spearmanr(v[col], v['fwd_excess_7d'])
        flag = ('*** KEEP' if abs(ic) > 0.03 and pval < 0.20
                else '    —')
        print(f"{col:<30} {ic:>+8.4f} {pval:>10.4f} "
              f"{len(v):>6}  {flag}")


def plot_filings(df, weekly):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    colors = {
        'positive': '#1D9E75',
        'negative': '#E24B4A',
        'neutral':  '#999'
    }
    sentiment_counts = df['sentiment'].value_counts()
    ax.bar(sentiment_counts.index, sentiment_counts.values,
           color=[colors.get(s, '#999')
                  for s in sentiment_counts.index])
    ax.set_title('BSE filing sentiment distribution')
    ax.set_ylabel('Number of filings')
    for i, (s, v) in enumerate(sentiment_counts.items()):
        ax.text(i, v + 5, str(v), ha='center')

    ax = axes[1]
    if not weekly.empty:
        recent = weekly[
            weekly['week'] >= weekly['week'].max() -
            pd.Timedelta(weeks=8)
        ]
        pivot = recent.pivot_table(
            index='ticker', columns='week',
            values='sentiment_score', aggfunc='mean'
        ).fillna(0)
        pivot.index = pivot.index.str.replace('.NS', '')
        im = ax.imshow(pivot.values, aspect='auto',
                       cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(
            [str(c.date()) for c in pivot.columns],
            rotation=45, ha='right', fontsize=7
        )
        ax.set_title('BSE filing sentiment heatmap\n(green = positive)')
        plt.colorbar(im, ax=ax, label='Sentiment score')

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/bse_filings.png', dpi=150)
    print("\n✓ Saved to outputs/bse_filings.png")
    plt.show()


if __name__ == '__main__':
    print("=== Fetching BSE Corporate Filings ===\n")
    df = fetch_all_filings(days_back=365)

    if df.empty:
        print("\n⚠️  No data fetched.")
        print("BSE API may be rate limiting. Try again in a few minutes.")
    else:
        os.makedirs('data/processed', exist_ok=True)
        df.to_csv('data/processed/bse_filings_raw.csv', index=False)
        print(f"\n✓ Saved raw filings")

        weekly = build_weekly_features(df)

        if not weekly.empty:
            weekly.to_csv(
                'data/processed/bse_filings_weekly.csv', index=False)
            print(f"✓ Saved weekly features")

            merged = merge_filings(weekly)
            ic_check(merged)
            plot_filings(df, weekly)

            print(f"\n=== Day 13 Complete ===")
            print(f"BSE filings added as third signal.")
            print(f"Run python src/model.py to retrain.")