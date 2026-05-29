import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sqlite3
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="Alpha Engine — Nifty IT",
    page_icon="📈",
    layout="wide"
)

# ── Load data ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_merged():
    return pd.read_csv('data/processed/merged_all.csv', parse_dates=['week'])

@st.cache_data(ttl=3600)
def load_prices():
    return pd.read_csv('data/raw/prices.csv', index_col=0, parse_dates=True)

@st.cache_data(ttl=3600)
def load_backtest():
    path = 'data/processed/backtest_results.csv'
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=['week'])
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_jobs_summary():
    conn = sqlite3.connect('data/jobs.db')
    df = pd.read_sql('''
        SELECT ticker,
               COUNT(*)          as total_jobs,
               MIN(posted_date)  as earliest,
               MAX(scraped_at)   as last_scraped
        FROM jobs
        GROUP BY ticker
        ORDER BY total_jobs DESC
    ''', conn)
    conn.close()
    return df

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("Alpha Engine")
st.sidebar.caption("Nifty IT — Alternative Data Signal")

page = st.sidebar.radio(
    "Navigate",
    ["Live Signal", "Backtest", "Features", "Data Health"]
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
st.sidebar.caption("⚠️ Research only. Not financial advice.")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE SIGNAL
# ══════════════════════════════════════════════════════════════════════════
if page == "Live Signal":
    st.title("📡 Live Signal")
    st.caption("Model predictions for the current week based on job posting + trends data")

    merged = load_merged()
    latest_week = merged['week'].max()
    latest = merged[merged['week'] == latest_week].copy()

    st.markdown(f"**Signal week:** {latest_week.date()}")

    # Run model on latest data
    feat_cols = [
        'skill_ai_ml_velocity', 'skill_digital_share', 'skill_cloud_share',
        'volume_zscore', 'seniority_score', 'skill_security_share',
        'skill_cloud_velocity'
    ]
    feat_cols = [f for f in feat_cols if f in merged.columns]

    train = merged[merged['week'] < latest_week].dropna(
        subset=feat_cols + ['fwd_excess_7d'])
    train['label'] = (train['fwd_excess_7d'] > 0.01).astype(int)

    if len(train) < 10 or train['label'].nunique() < 2:
        st.warning("Not enough historical data yet. Keep scraping weekly.")
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[feat_cols].fillna(0))
        X_live  = scaler.transform(latest[feat_cols].fillna(0))

        model = GradientBoostingClassifier(
            n_estimators=50, max_depth=2,
            learning_rate=0.1, random_state=42
        )
        model.fit(X_train, train['label'])

        latest = latest.copy()
        latest['signal'] = model.predict_proba(X_live)[:, 1]
        latest = latest.sort_values('signal', ascending=False)
        latest['ticker_clean'] = latest['ticker'].str.replace('.NS', '')

        # Top 3 picks
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        for i, (_, row) in enumerate(latest.head(3).iterrows()):
            with cols[i]:
                signal_pct = f"{row['signal']:.0%}"
                delta = "🟢 Strong" if row['signal'] > 0.7 else "🟡 Moderate"
                st.metric(
                    label=f"#{i+1} {row['ticker_clean']}",
                    value=signal_pct,
                    delta=delta
                )

        st.markdown("---")

        # Full signal table
        st.subheader("All signals this week")
        display = latest[[
            'ticker_clean', 'signal', 'volume_zscore',
            'skill_cloud_velocity', 'skill_digital_share',
            'skill_ai_ml_velocity'
        ]].copy()
        display.columns = [
            'Ticker', 'Signal', 'Vol Z-score',
            'Cloud velocity', 'Digital share', 'AI/ML velocity'
        ]
        display = display.set_index('Ticker')

        # Color the signal column
        st.dataframe(
            display.style.background_gradient(
                subset=['Signal'], cmap='RdYlGn', vmin=0, vmax=1
            ).format({
                'Signal': '{:.1%}',
                'Vol Z-score': '{:.2f}',
                'Cloud velocity': '{:.3f}',
                'Digital share': '{:.1%}',
                'AI/ML velocity': '{:.3f}',
            }),
            use_container_width=True
        )

        # Signal bar chart
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['#1D9E75' if v > 0.5 else '#E24B4A'
                  for v in latest['signal']]
        ax.barh(latest['ticker_clean'][::-1],
                latest['signal'][::-1], color=colors[::-1])
        ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Signal strength (probability of outperformance)')
        ax.set_title(f'Signal rankings — week of {latest_week.date()}')
        ax.set_xlim(0, 1)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════
elif page == "Backtest":
    st.title("📊 Backtest Results")
    st.caption("Walk-forward out-of-sample predictions only — no lookahead bias")

    backtest = load_backtest()

    if backtest.empty:
        st.warning("No backtest results yet. Run `python src/model.py` first.")
    else:
        # Summary metrics
        correct = (backtest['pred_label'] == backtest['label']).mean()
        top_picks = backtest.groupby('week').apply(
            lambda g: g.nlargest(1, 'pred_prob')['fwd_excess_7d'].mean()
        )
        avg_picks = backtest.groupby('week')['fwd_excess_7d'].mean()
        edge = (top_picks - avg_picks).mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{correct:.1%}")
        col2.metric("Predictions", len(backtest))
        col3.metric("Weeks tested", backtest['week'].nunique())
        col4.metric("Mean edge vs avg", f"{edge:.2%}")

        st.markdown("---")

        # Decile analysis
        st.subheader("Hit rate by signal confidence")
        backtest['decile'] = pd.qcut(
            backtest['pred_prob'], 3,
            labels=['Low', 'Mid', 'High']
        )
        decile_stats = backtest.groupby('decile', observed=True).agg(
            hit_rate=('label', 'mean'),
            count=('label', 'count'),
            mean_return=('fwd_excess_7d', 'mean')
        ).round(3)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        colors = ['#E24B4A', '#EF9F27', '#1D9E75']
        ax.bar(decile_stats.index, decile_stats['hit_rate'],
               color=colors)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title('Hit rate by confidence decile')
        ax.set_ylabel('% actually outperformed')
        ax.set_ylim(0, 1)
        for i, (idx, row) in enumerate(decile_stats.iterrows()):
            ax.text(i, row['hit_rate'] + 0.02,
                    f"{row['hit_rate']:.0%}", ha='center')

        ax = axes[1]
        ax.bar(decile_stats.index, decile_stats['mean_return'] * 100,
               color=colors)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title('Mean 7d excess return by confidence decile')
        ax.set_ylabel('Mean excess return (%)')
        for i, (idx, row) in enumerate(decile_stats.iterrows()):
            ax.text(i, row['mean_return'] * 100 + 0.1,
                    f"{row['mean_return']*100:.1f}%", ha='center')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.subheader("All predictions")
        display = backtest[[
            'week', 'ticker', 'pred_prob', 'pred_label',
            'label', 'fwd_excess_7d'
        ]].copy()
        display.columns = [
            'Week', 'Ticker', 'Signal', 'Predicted',
            'Actual', '7d Excess Return'
        ]
        display['Correct'] = display['Predicted'] == display['Actual']
        display['Ticker'] = display['Ticker'].str.replace('.NS', '')
        st.dataframe(
            display.style.background_gradient(
                subset=['Signal'], cmap='RdYlGn', vmin=0, vmax=1
            ).format({
                'Signal': '{:.1%}',
                '7d Excess Return': '{:.2%}'
            }),
            use_container_width=True
        )

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — FEATURES
# ══════════════════════════════════════════════════════════════════════════
elif page == "Features":
    st.title("🔬 Feature Explorer")

    merged = load_merged()
    prices = load_prices()

    # Stock selector
    tickers = sorted(merged['ticker'].unique())
    selected = st.selectbox(
        "Select company",
        tickers,
        format_func=lambda x: x.replace('.NS', '')
    )

    company = merged[merged['ticker'] == selected].sort_values('week')

    if company.empty:
        st.warning("No feature data for this company yet.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Job posting volume z-score")
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.bar(company['week'], company['volume_zscore'],
                   color=['#1D9E75' if v > 0 else '#E24B4A'
                          for v in company['volume_zscore']])
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax.set_ylabel('Z-score vs own baseline')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Skill mix over time")
            skill_cols = [c for c in company.columns
                          if c.startswith('skill_') and 'share' in c
                          and 'velocity' not in c]
            if skill_cols:
                fig, ax = plt.subplots(figsize=(7, 3))
                bottom = np.zeros(len(company))
                colors_skill = ['#1D9E75', '#0C447C', '#EF9F27',
                                '#E24B4A', '#7F77DD', '#85B7EB']
                for j, col in enumerate(skill_cols):
                    vals = company[col].fillna(0).values
                    ax.bar(company['week'], vals, bottom=bottom,
                           label=col.replace('skill_', '').replace('_share', ''),
                           color=colors_skill[j % len(colors_skill)],
                           alpha=0.8)
                    bottom += vals
                ax.set_ylabel('Share of postings')
                ax.legend(fontsize=7, loc='upper right')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                plt.xticks(rotation=30)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # Price chart
        st.subheader("Stock price")
        ticker_col = selected if selected in prices.columns else None
        if ticker_col:
            price_data = prices[ticker_col].dropna().last('180D')
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(price_data.index, price_data.values,
                    color='steelblue', linewidth=1.2)
            ax.set_ylabel('Price (INR)')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Raw feature table
        st.subheader("Raw feature values")
        display_cols = ['week', 'job_count', 'volume_zscore'] + skill_cols
        display_cols = [c for c in display_cols if c in company.columns]
        st.dataframe(company[display_cols].round(3), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — DATA HEALTH
# ══════════════════════════════════════════════════════════════════════════
elif page == "Data Health":
    st.title("🏥 Data Health")
    st.caption("Monitor your data pipeline — scrape weekly to keep this green")

    jobs_summary = load_jobs_summary()
    merged = load_merged()

    # Overall stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total job records", f"{jobs_summary['total_jobs'].sum():,}")
    col2.metric("Companies tracked", len(jobs_summary))
    col3.metric("Feature weeks", merged['week'].nunique())
    col4.metric("Valid predictions", merged['fwd_excess_7d'].notna().sum())

    st.markdown("---")

    # Jobs per company
    st.subheader("Job records per company")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(
        jobs_summary['ticker'].str.replace('.NS', ''),
        jobs_summary['total_jobs'],
        color='steelblue'
    )
    ax.set_xlabel('Total job records scraped')
    ax.set_title('Scraped job records by company')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Coverage table
    st.subheader("Scrape coverage")
    display = jobs_summary.copy()
    display['ticker'] = display['ticker'].str.replace('.NS', '')
    display['last_scraped'] = pd.to_datetime(
        display['last_scraped']).dt.date
    display.columns = ['Ticker', 'Total Jobs', 'Earliest Post', 'Last Scraped']
    st.dataframe(display, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature data coverage")
    coverage = merged.groupby('ticker').agg(
        weeks=('week', 'nunique'),
        valid_7d=('fwd_excess_7d', lambda x: x.notna().sum()),
        latest_week=('week', 'max')
    ).reset_index()
    coverage['ticker'] = coverage['ticker'].str.replace('.NS', '')
    coverage['latest_week'] = pd.to_datetime(
        coverage['latest_week']).dt.date
    coverage.columns = ['Ticker', 'Feature Weeks',
                        'Valid Returns', 'Latest Week']
    st.dataframe(coverage, use_container_width=True)