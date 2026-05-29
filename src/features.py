import pandas as pd
import numpy as np
import sqlite3
import json
import re
import os

# ── Skill taxonomy ─────────────────────────────────────────────────────────
SKILL_CATS = {
    'ai_ml':    ['machine learning', 'deep learning', 'llm', 'genai', 'ai ',
                 'data science', 'nlp', 'computer vision', 'pytorch', 'tensorflow',
                 'generative ai', 'artificial intelligence'],
    'cloud':    ['aws', 'azure', 'gcp', 'cloud', 'devops', 'kubernetes',
                 'docker', 'terraform', 'microservices', 'devsecops'],
    'legacy':   ['sap', 'mainframe', 'cobol', 'oracle forms', 'as400',
                 'peoplesoft', 'siebel', 'informatica'],
    'digital':  ['react', 'angular', 'node', 'fullstack', 'full stack',
                 'mobile', 'flutter', 'react native', 'ios', 'android'],
    'data':     ['data engineer', 'data analyst', 'spark', 'hadoop', 'kafka',
                 'snowflake', 'databricks', 'sql', 'etl', 'pipeline'],
    'security': ['cybersecurity', 'security', 'soc', 'penetration', 'iam',
                 'zero trust', 'siem', 'devsecops'],
}

SENIOR_PATTERN = r'\b(senior|principal|lead|director|vp|head of|manager|architect|chief)\b'
JUNIOR_PATTERN  = r'\b(junior|fresher|associate|entry|graduate|trainee|intern|assistant)\b'


# ── Load jobs ──────────────────────────────────────────────────────────────
def load_jobs():
    conn = sqlite3.connect('data/jobs.db')
    df = pd.read_sql("SELECT * FROM jobs", conn)
    conn.close()

    df['skills_list'] = df['skills'].apply(lambda x: json.loads(x) if x else [])
    df['text'] = (
        df['title'].fillna('') + ' ' +
        df['skills_list'].apply(lambda s: ' '.join(s))
    ).str.lower()

    df['posted_date'] = pd.to_datetime(df['posted_date'], errors='coerce')
    df['scraped_at']  = pd.to_datetime(df['scraped_at'],  errors='coerce')
    df['week']        = df['posted_date'].dt.to_period('W').dt.start_time

    print(f"✓ Loaded {len(df)} jobs")
    print(f"  Date range: {df['posted_date'].min().date()} → {df['posted_date'].max().date()}")
    print(f"  Companies: {df['ticker'].nunique()}")
    return df


# ── Extract features ───────────────────────────────────────────────────────
def extract_features(df):
    records = []

    for ticker in df['ticker'].unique():
        company_df = df[df['ticker'] == ticker].copy()
        weeks = sorted(company_df['week'].dropna().unique())

        for week in weeks:
            this_week = company_df[company_df['week'] == week]
            if len(this_week) < 3:
                continue

            feat  = {'ticker': ticker, 'week': week}
            total = len(this_week) + 1e-9

            # Job count
            feat['job_count'] = len(this_week)

            # Skill category shares
            for cat, keywords in SKILL_CATS.items():
                pattern = '|'.join(re.escape(k) for k in keywords)
                count = this_week['text'].str.contains(pattern, na=False).sum()
                feat[f'skill_{cat}_share'] = count / total

            # Seniority score = senior% - junior% (bounded, no division by zero)
            senior_count = this_week['title'].str.lower().str.contains(
                SENIOR_PATTERN, regex=True, na=False).sum()
            junior_count = this_week['title'].str.lower().str.contains(
                JUNIOR_PATTERN, regex=True, na=False).sum()
            feat['senior_share']    = senior_count / total
            feat['junior_share']    = junior_count / total
            feat['seniority_score'] = (senior_count - junior_count) / total

            records.append(feat)

    features = pd.DataFrame(records)
    print(f"\n✓ Raw features shape: {features.shape}")
    return features


# ── Volume z-score ─────────────────────────────────────────────────────────
def add_volume_zscore(features, window=8):
    def zscore_company(grp):
        grp = grp.sort_values('week').copy()
        rolling_mean = grp['job_count'].rolling(window, min_periods=2).mean()
        rolling_std  = grp['job_count'].rolling(window, min_periods=2).std()
        grp['volume_zscore'] = (grp['job_count'] - rolling_mean) / (rolling_std + 1e-9)
        return grp

    features = features.groupby('ticker', group_keys=False).apply(zscore_company)
    return features


# ── Skill velocity ─────────────────────────────────────────────────────────
def add_skill_velocity(features):
    def velocity_company(grp):
        grp = grp.sort_values('week').copy()
        for cat in SKILL_CATS.keys():
            col = f'skill_{cat}_share'
            if col in grp.columns:
                grp[f'skill_{cat}_velocity'] = grp[col].diff()
        return grp

    features = features.groupby('ticker', group_keys=False).apply(velocity_company)
    return features


# ── Summary ────────────────────────────────────────────────────────────────
def summarise(features):
    print("\n=== Feature Summary ===")
    print(f"Shape: {features.shape}")
    print(f"Weeks: {features['week'].min().date()} → {features['week'].max().date()}")

    print(f"\nRows per ticker:")
    print(features.groupby('ticker').size().sort_values(ascending=False).to_string())

    print(f"\nMissing values:")
    nulls = features.isnull().sum()
    print(nulls[nulls > 0].to_string() if nulls.any() else "  None ✓")

    print(f"\nTop 5 by volume z-score this week:")
    last = features[features['week'] == features['week'].max()]
    cols = ['ticker', 'job_count', 'volume_zscore',
            'skill_ai_ml_share', 'skill_cloud_share', 'seniority_score']
    print(last[cols].sort_values('volume_zscore', ascending=False)
          .head(5).round(3).to_string(index=False))


# ── Plot ───────────────────────────────────────────────────────────────────
def plot_features(features):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Volume z-score over time
    ax = axes[0, 0]
    for ticker in features['ticker'].unique():
        t = features[features['ticker'] == ticker].sort_values('week')
        ax.plot(t['week'], t['volume_zscore'],
                label=ticker.replace('.NS', ''), alpha=0.7, linewidth=1)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title('Job posting volume z-score over time')
    ax.set_ylabel('Z-score vs own baseline')
    ax.legend(ncol=4, fontsize=7)

    # 2. AI/ML skill share latest week
    ax = axes[0, 1]
    latest = features[features['week'] == features['week'].max()].copy()
    latest = latest.sort_values('skill_ai_ml_share', ascending=True)
    ax.barh(
        latest['ticker'].str.replace('.NS', ''),
        latest['skill_ai_ml_share'] * 100,
        color='steelblue'
    )
    ax.set_title('AI/ML skill share — latest week (%)')
    ax.set_xlabel('% of job postings mentioning AI/ML')

    # 3. Seniority score latest week
    ax = axes[1, 0]
    latest = latest.sort_values('seniority_score', ascending=True)
    colors = ['#1D9E75' if v > 0 else '#E24B4A' for v in latest['seniority_score']]
    ax.barh(
        latest['ticker'].str.replace('.NS', ''),
        latest['seniority_score'],
        color=colors
    )
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title('Seniority score (senior% - junior%) — latest week')
    ax.set_xlabel('Positive = more senior hires')

    # 4. Weekly job count heatmap
    ax = axes[1, 1]
    pivot = features.pivot_table(
        index='ticker', columns='week', values='job_count', aggfunc='sum'
    )
    pivot.index = pivot.index.str.replace('.NS', '')
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [str(c.date()) for c in pivot.columns],
        rotation=45, ha='right', fontsize=7
    )
    ax.set_title('Weekly job count heatmap')
    plt.colorbar(im, ax=ax, label='Job count')

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/features.png', dpi=150)
    print("\n✓ Saved to outputs/features.png")
    plt.show()


# ── Save ───────────────────────────────────────────────────────────────────
def save(features):
    os.makedirs('data/processed', exist_ok=True)
    features.to_csv('data/processed/features.csv', index=False)
    print(f"✓ Saved to data/processed/features.csv")
    print(f"  Shape: {features.shape}")
    print(f"  Columns: {list(features.columns)}")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df       = load_jobs()
    features = extract_features(df)
    features = add_volume_zscore(features)
    features = add_skill_velocity(features)
    summarise(features)
    plot_features(features)
    save(features)