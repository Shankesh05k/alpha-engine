import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
FEAT_COLS = [
    'skill_ai_ml_velocity', 'skill_digital_share', 'skill_cloud_share',
    'volume_zscore', 'seniority_score', 'skill_security_share',
    'skill_cloud_velocity'
]
TARGET        = 'fwd_excess_7d'
LABEL_THRESH  = 0.01
MIN_TRAIN     = 10
N_PERMUTATIONS = 500


# ── Walk-forward engine ────────────────────────────────────────────────────
def run_walkforward(df, feat_cols, shuffle_col=None):
    """
    Run walk-forward backtest.
    If shuffle_col is set, shuffle that column's values within each week
    before training — this breaks the signal while preserving structure.
    """
    weeks = sorted(df['week'].unique())
    all_preds = []

    for test_week in weeks:
        train = df[df['week'] < test_week].copy()
        test  = df[df['week'] == test_week].copy()

        if len(train) < MIN_TRAIN or train['label'].nunique() < 2:
            continue
        if len(test) == 0:
            continue

        # Shuffle signal if doing permutation
        if shuffle_col:
            for col in feat_cols:
                # Shuffle within each week to preserve cross-sectional structure
                for week in train['week'].unique():
                    mask = train['week'] == week
                    train.loc[mask, col] = (
                        train.loc[mask, col].sample(frac=1).values
                    )

        X_tr = train[feat_cols].fillna(0)
        y_tr = train['label']
        X_te = test[feat_cols].fillna(0)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = GradientBoostingClassifier(
            n_estimators=50, max_depth=2,
            learning_rate=0.1, random_state=42
        )
        model.fit(X_tr_s, y_tr)

        probs = model.predict_proba(X_te_s)[:, 1]
        test  = test.copy()
        test['pred_prob'] = probs
        all_preds.append(test)

    if not all_preds:
        return pd.DataFrame()
    return pd.concat(all_preds, ignore_index=True)


# ── Compute strategy metric ────────────────────────────────────────────────
def compute_sharpe(results):
    """
    Sharpe ratio of weekly excess returns from top-1 pick strategy.
    Higher = better. This is what we permute.
    """
    if results.empty:
        return np.nan

    weekly_returns = []
    for week, grp in results.groupby('week'):
        if len(grp) < 2:
            continue
        top_return = grp.nlargest(1, 'pred_prob')[TARGET].values[0]
        avg_return = grp[TARGET].mean()
        weekly_returns.append(top_return - avg_return)  # excess over random

    if len(weekly_returns) < 2:
        return np.nan

    sr = pd.Series(weekly_returns)
    return sr.mean() / (sr.std() + 1e-8) * np.sqrt(52)


def compute_hit_rate(results):
    """Hit rate of top-1 pick vs average."""
    if results.empty:
        return np.nan
    weekly = []
    for week, grp in results.groupby('week'):
        if len(grp) < 2:
            continue
        top = grp.nlargest(1, 'pred_prob')[TARGET].values[0]
        avg = grp[TARGET].mean()
        weekly.append(1 if top > avg else 0)
    return np.mean(weekly) if weekly else np.nan


def compute_top_decile_return(results):
    """Mean return of top-confidence picks."""
    if results.empty:
        return np.nan
    top_third = results.nlargest(max(1, len(results)//3), 'pred_prob')
    return top_third[TARGET].mean()


# ── Main permutation test ──────────────────────────────────────────────────
def permutation_test(df, feat_cols, n_perms=N_PERMUTATIONS):
    print(f"Running permutation test ({n_perms} iterations)...")
    print(f"Dataset: {len(df)} rows, {df['week'].nunique()} weeks\n")

    # Real signal performance
    real_results = run_walkforward(df, feat_cols)

    if real_results.empty:
        print("⚠️  No walk-forward predictions — need more data.")
        print(f"   Currently have {df['week'].nunique()} weeks.")
        print(f"   Need at least 2 weeks with {MIN_TRAIN}+ training rows each.")
        return None

    real_sharpe   = compute_sharpe(real_results)
    real_hitrate  = compute_hit_rate(real_results)
    real_top_ret  = compute_top_decile_return(real_results)

    print(f"Real strategy metrics:")
    print(f"  Sharpe ratio:          {real_sharpe:.3f}")
    print(f"  Hit rate vs avg:       {real_hitrate:.1%}")
    print(f"  Top decile mean return:{real_top_ret:.4f}")

    # Permuted null distribution
    null_sharpes  = []
    null_hitrates = []
    null_top_rets = []

    for i in tqdm(range(n_perms), desc='Permuting'):
        perm_df = df.copy()
        # Shuffle labels within each week — breaks signal, preserves structure
        for week in perm_df['week'].unique():
            mask = perm_df['week'] == week
            perm_df.loc[mask, 'label'] = (
                perm_df.loc[mask, 'label'].sample(frac=1).values
            )
        perm_results = run_walkforward(perm_df, feat_cols)
        if not perm_results.empty:
            null_sharpes.append(compute_sharpe(perm_results))
            null_hitrates.append(compute_hit_rate(perm_results))
            null_top_rets.append(compute_top_decile_return(perm_results))

    null_sharpes  = [x for x in null_sharpes  if not np.isnan(x)]
    null_hitrates = [x for x in null_hitrates if not np.isnan(x)]
    null_top_rets = [x for x in null_top_rets if not np.isnan(x)]

    # P-values
    p_sharpe  = (np.array(null_sharpes)  >= real_sharpe).mean()  if null_sharpes  else np.nan
    p_hitrate = (np.array(null_hitrates) >= real_hitrate).mean() if null_hitrates else np.nan
    p_topret  = (np.array(null_top_rets) >= real_top_ret).mean() if null_top_rets else np.nan

    print(f"\n=== Permutation Test Results ===")
    print(f"{'Metric':<25} {'Real':>8} {'Null mean':>10} {'p-value':>10} {'Significant?':>14}")
    print("-" * 72)
    print(f"{'Sharpe ratio':<25} {real_sharpe:>8.3f} "
          f"{np.mean(null_sharpes) if null_sharpes else np.nan:>10.3f} "
          f"{p_sharpe:>10.3f}  "
          f"{'YES ✓' if p_sharpe < 0.1 else 'not yet'}")
    print(f"{'Hit rate':<25} {real_hitrate:>8.1%} "
          f"{np.mean(null_hitrates) if null_hitrates else np.nan:>10.1%} "
          f"{p_hitrate:>10.3f}  "
          f"{'YES ✓' if p_hitrate < 0.1 else 'not yet'}")
    print(f"{'Top decile return':<25} {real_top_ret:>8.4f} "
          f"{np.mean(null_top_rets) if null_top_rets else np.nan:>10.4f} "
          f"{p_topret:>10.3f}  "
          f"{'YES ✓' if p_topret < 0.1 else 'not yet'}")

    print(f"\n{'='*50}")
    print(f"HONEST ASSESSMENT")
    print(f"{'='*50}")
    print(f"Predictions made: {len(real_results)}")
    print(f"Weeks tested:     {real_results['week'].nunique()}")
    print(f"\nWith {real_results['week'].nunique()} test week(s), p-values are")
    print(f"unstable. You need 10+ test weeks for reliable significance.")
    print(f"\nWhat the numbers mean right now:")
    if not np.isnan(real_sharpe) and real_sharpe > 0:
        print(f"  ✓ Positive Sharpe ({real_sharpe:.2f}) — strategy beats random")
    if not np.isnan(real_hitrate) and real_hitrate > 0.5:
        print(f"  ✓ Hit rate above 50% ({real_hitrate:.0%}) — top pick beats avg more often than not")
    print(f"\nKeep scraping weekly. Rerun this after 8+ weeks of predictions.")

    return {
        'real_sharpe':   real_sharpe,
        'real_hitrate':  real_hitrate,
        'null_sharpes':  null_sharpes,
        'null_hitrates': null_hitrates,
        'p_sharpe':      p_sharpe,
        'p_hitrate':     p_hitrate,
    }


# ── Plot ───────────────────────────────────────────────────────────────────
def plot_permutation(results_dict):
    if results_dict is None:
        return

    null_sharpes  = results_dict['null_sharpes']
    null_hitrates = results_dict['null_hitrates']
    real_sharpe   = results_dict['real_sharpe']
    real_hitrate  = results_dict['real_hitrate']
    p_sharpe      = results_dict['p_sharpe']
    p_hitrate     = results_dict['p_hitrate']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sharpe permutation
    ax = axes[0]
    if null_sharpes:
        ax.hist(null_sharpes, bins=40, alpha=0.7,
                color='steelblue', label='Shuffled signal')
        ax.axvline(real_sharpe, color='red', linewidth=2,
                   label=f'Real Sharpe ({real_sharpe:.2f})')
        ax.axvline(np.percentile(null_sharpes, 90),
                   color='orange', linestyle='--', linewidth=1,
                   label='90th percentile null')
    ax.set_title(f'Permutation test — Sharpe ratio\np-value = {p_sharpe:.3f}')
    ax.set_xlabel('Sharpe ratio')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)

    # Hit rate permutation
    ax = axes[1]
    if null_hitrates:
        ax.hist(null_hitrates, bins=20, alpha=0.7,
                color='steelblue', label='Shuffled signal')
        ax.axvline(real_hitrate, color='red', linewidth=2,
                   label=f'Real hit rate ({real_hitrate:.0%})')
        ax.axvline(np.percentile(null_hitrates, 90),
                   color='orange', linestyle='--', linewidth=1,
                   label='90th percentile null')
    ax.set_title(f'Permutation test — Hit rate\np-value = {p_hitrate:.3f}')
    ax.set_xlabel('Hit rate')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)

    plt.suptitle('Permutation tests — are results better than random?',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/permutation_test.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved to outputs/permutation_test.png")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Load data
    df = pd.read_csv('data/processed/merged_all.csv', parse_dates=['week'])
    df = df.sort_values(['week', 'ticker']).reset_index(drop=True)

    feat_cols = [f for f in FEAT_COLS if f in df.columns]
    df['label'] = (df[TARGET] > LABEL_THRESH).astype(int)
    df = df.dropna(subset=[TARGET] + feat_cols)

    print(f"✓ Dataset: {len(df)} rows, {df['week'].nunique()} weeks")
    print(f"  Label balance: {df['label'].mean():.1%} positive\n")

    results = permutation_test(df, feat_cols, n_perms=N_PERMUTATIONS)
    plot_permutation(results)