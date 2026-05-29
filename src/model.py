import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats

# ── Config ─────────────────────────────────────────────────────────────────
# Use all features with |IC| > 0.03 — don't just use keepers
# With small data, more features help as long as tree depth is shallow
FEATURE_COLS = [
    'skill_ai_ml_velocity',
    'skill_digital_share',
    'skill_cloud_share',
    'volume_zscore',
    'seniority_score',
    'skill_security_share',
    'skill_cloud_velocity',
]

LABEL_THRESHOLD = 0.01   # 1% excess return = outperform
TARGET_COL      = 'fwd_excess_7d'
MIN_TRAIN_ROWS = 10  # lowered from 20 for early stage     # minimum rows needed to train


# ── Load data ──────────────────────────────────────────────────────────────
def load():
    df = pd.read_csv('data/processed/merged_all.csv', parse_dates=['week'])
    df = df.sort_values(['week', 'ticker']).reset_index(drop=True)

    # Keep only features that exist
    feat_cols = [f for f in FEATURE_COLS if f in df.columns]

    # Build binary label: 1 = outperformed by >1%, 0 = didn't
    df['label'] = (df[TARGET_COL] > LABEL_THRESHOLD).astype(int)
    df = df.dropna(subset=[TARGET_COL] + feat_cols)

    print(f"✓ Loaded {len(df)} rows with valid target + features")
    print(f"  Weeks: {sorted(df['week'].dt.date.unique())}")
    print(f"  Label balance: {df['label'].mean():.1%} positive")
    print(f"  Features: {feat_cols}")
    return df, feat_cols


# ── Walk-forward backtest ──────────────────────────────────────────────────
def walk_forward(df, feat_cols):
    """
    For each week (after minimum training period):
    - Train on all PAST weeks
    - Predict on current week
    - Record prediction vs actual

    This is the only valid way to backtest with time-series data.
    """
    weeks = sorted(df['week'].unique())
    all_results = []

    print(f"\n=== Walk-Forward Backtest ===")
    print(f"Total weeks: {len(weeks)}")
    print(f"Min training weeks: need {MIN_TRAIN_ROWS} rows\n")

    for i, test_week in enumerate(weeks):
        # Training data = everything BEFORE this week
        train = df[df['week'] < test_week].copy()
        test  = df[df['week'] == test_week].copy()

        if len(train) < MIN_TRAIN_ROWS:
            print(f"  Week {test_week.date()} — skipping (only {len(train)} train rows)")
            continue

        if len(test) == 0:
            continue

        X_train = train[feat_cols].fillna(0)
        y_train = train['label']
        X_test  = test[feat_cols].fillna(0)

        # Check we have both classes in training data
        if y_train.nunique() < 2:
            print(f"  Week {test_week.date()} — skipping (only one class in training)")
            continue

        # Scale features
        scaler  = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # Shallow GBM — prevents overfitting on small data
        model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=2,        # very shallow — critical for small datasets
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        model.fit(X_train_s, y_train)

        # Predict
        probs = model.predict_proba(X_test_s)[:, 1]
        preds = (probs > 0.5).astype(int)

        test = test.copy()
        test['pred_prob']  = probs
        test['pred_label'] = preds
        test['train_rows'] = len(train)

        all_results.append(test)
        print(f"  Week {test_week.date()} — trained on {len(train)} rows, "
              f"predicted {len(test)} stocks")

    if not all_results:
        print("\n⚠️  No predictions made — need more data.")
        print("    Keep running the scraper weekly.")
        return pd.DataFrame()

    results = pd.concat(all_results, ignore_index=True)
    print(f"\n✓ Total predictions: {len(results)}")
    return results


# ── Evaluate ───────────────────────────────────────────────────────────────
def evaluate(results):
    if results.empty:
        return

    print("\n=== Model Evaluation ===")

    # Classification metrics
    print("\nClassification report:")
    print(classification_report(
        results['label'], results['pred_label'],
        target_names=['Underperform', 'Outperform']
    ))

    # Hit rate — did top predicted stock actually outperform?
    print("Hit rate by confidence decile:")
    results['decile'] = pd.qcut(results['pred_prob'], 3,
                                 labels=['Low', 'Mid', 'High'])
    hit_by_decile = results.groupby('decile', observed=True).agg(
        actual_positive=('label', 'mean'),
        count=('label', 'count'),
        mean_return=(TARGET_COL, 'mean')
    ).round(3)
    print(hit_by_decile)

    # Weekly top pick performance
    print("\nWeekly top pick vs random:")
    weekly = []
    for week, grp in results.groupby('week'):
        if len(grp) < 2:
            continue
        top_pick    = grp.nlargest(1, 'pred_prob')[TARGET_COL].values[0]
        random_pick = grp[TARGET_COL].mean()
        weekly.append({
            'week': week,
            'top_pick_return': top_pick,
            'avg_return': random_pick,
            'edge': top_pick - random_pick
        })

    if weekly:
        wdf = pd.DataFrame(weekly)
        print(wdf.round(4).to_string(index=False))
        print(f"\nMean edge (top pick vs average): {wdf['edge'].mean():.4f}")
        print(f"Hit rate (top pick beat average): {(wdf['edge'] > 0).mean():.1%}")


# ── Feature importance ─────────────────────────────────────────────────────
def feature_importance(df, feat_cols):
    """Train on all data and show feature importance."""
    valid = df.dropna(subset=feat_cols + ['label'])
    if valid['label'].nunique() < 2:
        print("Can't compute importance — only one class.")
        return

    X = valid[feat_cols].fillna(0)
    y = valid['label']

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = GradientBoostingClassifier(
        n_estimators=50, max_depth=2,
        learning_rate=0.1, random_state=42
    )
    model.fit(X_s, y)

    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#1D9E75' if v > 0 else '#E24B4A' for v in imp.values]
    ax.barh(imp.index, imp.values, color=colors)
    ax.set_title('Feature importance — GBM trained on all data')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/feature_importance.png', dpi=150)
    print("\n✓ Saved to outputs/feature_importance.png")
    plt.show()
    return model


# ── Current week signal ────────────────────────────────────────────────────
def current_signal(df, feat_cols):
    """
    Train on all available data, predict this week's top picks.
    This is the 'live' output of your engine.
    """
    latest_week = df['week'].max()
    latest = df[df['week'] == latest_week].copy()
    train  = df[df['week'] < latest_week].copy()

    if len(train) < MIN_TRAIN_ROWS or train['label'].nunique() < 2:
        print("\n⚠️  Not enough data for live signal yet.")
        return

    X_train = train[feat_cols].fillna(0)
    y_train = train['label']
    X_live  = latest[feat_cols].fillna(0)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_live_s  = scaler.transform(X_live)

    model = GradientBoostingClassifier(
        n_estimators=50, max_depth=2,
        learning_rate=0.1, random_state=42
    )
    model.fit(X_train_s, y_train)

    latest = latest.copy()
    latest['signal'] = model.predict_proba(X_live_s)[:, 1]
    latest = latest.sort_values('signal', ascending=False)

    print(f"\n=== Live Signal — Week of {latest_week.date()} ===")
    print(f"{'Ticker':<20} {'Signal':>8} {'AI/ML vel':>12} {'Digital%':>10} {'Vol Z':>8}")
    print("-" * 62)

    display_cols = ['ticker', 'signal', 'skill_ai_ml_velocity',
                    'skill_digital_share', 'volume_zscore']
    display_cols = [c for c in display_cols if c in latest.columns]

    for _, row in latest.head(5).iterrows():
        ticker = row['ticker'].replace('.NS', '')
        signal = row.get('signal', 0)
        ai_vel = row.get('skill_ai_ml_velocity', 0)
        dig    = row.get('skill_digital_share', 0)
        vol_z  = row.get('volume_zscore', 0)
        print(f"{ticker:<20} {signal:>8.3f} {ai_vel:>12.3f} {dig:>10.3f} {vol_z:>8.3f}")

    print(f"\nTop pick: {latest.iloc[0]['ticker'].replace('.NS','')} "
          f"(signal={latest.iloc[0]['signal']:.3f})")
    print("\n⚠️  This is experimental research, not financial advice.")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df, feat_cols = load()

    print("\n" + "="*50)
    print("STEP 1: Walk-forward backtest")
    print("="*50)
    results = walk_forward(df, feat_cols)
    evaluate(results)

    print("\n" + "="*50)
    print("STEP 2: Feature importance")
    print("="*50)
    feature_importance(df, feat_cols)

    print("\n" + "="*50)
    print("STEP 3: Current week live signal")
    print("="*50)
    current_signal(df, feat_cols)

    # Save results
    if not results.empty:
        os.makedirs('data/processed', exist_ok=True)
        results.to_csv('data/processed/backtest_results.csv', index=False)
        print(f"\n✓ Saved backtest results to data/processed/backtest_results.csv")