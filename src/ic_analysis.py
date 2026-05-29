import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy import stats

# ── Plot ───────────────────────────────────────────────────────────────────
def plot_ic(results_df, merged, horizon):
    if results_df.empty:
        print("Nothing to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # IC bar chart
    ax = axes[0]
    rs = results_df.sort_values('ic')
    colors = ['#1D9E75' if v > 0 else '#E24B4A' for v in rs['ic']]
    ax.barh(rs['feature'], rs['ic'], color=colors)
    ax.axvline(0,     color='gray',  linestyle='--', linewidth=0.8)
    ax.axvline(0.03,  color='green', linestyle=':',  linewidth=1)
    ax.axvline(-0.03, color='green', linestyle=':',  linewidth=1)
    ax.set_title(f'IC per feature vs {horizon}d excess return')
    ax.set_xlabel('IC (Spearman)')
    for i, (_, row) in enumerate(rs.iterrows()):
        ax.text(row['ic'] + 0.002 if row['ic'] >= 0 else row['ic'] - 0.002,
                i, f"p={row['pval']:.2f}", va='center', fontsize=8,
                ha='left' if row['ic'] >= 0 else 'right',
                color='green' if row['is_signal'] else 'gray')

    # Scatter: best feature vs return
    ax = axes[1]
    best = results_df.loc[results_df['ic'].abs().idxmax(), 'feature']
    valid = merged[[best, 'fwd_excess']].dropna()

    ax.scatter(valid[best], valid['fwd_excess'] * 100,
               alpha=0.5, s=25, color='steelblue')

    if len(valid) > 2:
        z = np.polyfit(valid[best], valid['fwd_excess'] * 100, 1)
        x_r = np.linspace(valid[best].min(), valid[best].max(), 100)
        ax.plot(x_r, np.poly1d(z)(x_r), color='red', linewidth=1.5)

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel(best)
    ax.set_ylabel(f'{horizon}d excess return (%)')
    ax.set_title(f'Best feature: {best}')

    ic_val = results_df[results_df['feature'] == best]['ic'].values[0]
    ax.text(0.05, 0.95, f'IC = {ic_val:.4f}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/ic_analysis.png', dpi=150)
    print("✓ Saved to outputs/ic_analysis.png")
    plt.show()


# ── Interpret ──────────────────────────────────────────────────────────────
def interpret(results_df, merged, horizon):
    n_weeks = merged['week'].nunique()
    print(f"\n=== Honest Assessment ===")
    print(f"Observations: {len(merged)} | Weeks: {n_weeks} | Horizon: {horizon}d")
    print(f"\nWith {n_weeks} weeks of data results are directional only.")
    print(f"You need 20+ weeks for statistically meaningful IC.")
    print(f"Keep running the scraper weekly — rerun this after 8 weeks.")

    if results_df.empty:
        return

    print(f"\nBest feature by |IC|:")
    best = results_df.loc[results_df['ic'].abs().idxmax()]
    direction = ("more hiring → outperformance ✓"
                 if best['ic'] > 0
                 else "more hiring → underperformance ✗ (unexpected)")
    print(f"  {best['feature']}: IC={best['ic']:+.4f}, p={best['pval']:.3f}")
    print(f"  Direction: {direction}")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # Load the fully merged dataset including Google Trends
    merged = pd.read_csv('data/processed/merged_all.csv', parse_dates=['week'])
    merged = merged.rename(columns={'fwd_excess_7d': 'fwd_excess'})
    merged = merged.dropna(subset=['fwd_excess'])
    horizon = 7

    print(f"✓ Loaded merged_all.csv: {merged.shape}")
    print(f"  Valid rows with 7d return: {len(merged)}")

    # All features including trends
    feature_cols = [
        'volume_zscore',
        'skill_ai_ml_share',
        'skill_cloud_share',
        'skill_legacy_share',
        'skill_digital_share',
        'skill_data_share',
        'skill_security_share',
        'seniority_score',
        'skill_ai_ml_velocity',
        'skill_cloud_velocity',
        'trends_zscore',
        'trends_velocity',
    ]
    feature_cols = [f for f in feature_cols if f in merged.columns]

    results = []
    print(f"\n=== Final IC Analysis — all signals ===")
    print(f"{'Feature':<30} {'IC':>8} {'p-value':>10} {'N':>6}  Signal?")
    print("-" * 68)

    for col in feature_cols:
        valid = merged[[col, 'fwd_excess']].dropna()
        if len(valid) < 5:
            print(f"{col:<30} {'N/A':>8} {'N/A':>10} {len(valid):>6}")
            continue
        ic, pval = stats.spearmanr(valid[col], valid['fwd_excess'])
        is_signal = abs(ic) > 0.03 and pval < 0.20
        flag = '*** KEEP' if is_signal else '    —'
        print(f"{col:<30} {ic:>+8.4f} {pval:>10.4f} {len(valid):>6}  {flag}")
        results.append({
            'feature':   col,
            'ic':        ic,
            'pval':      pval,
            'n':         len(valid),
            'is_signal': is_signal
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        keepers = results_df[results_df['is_signal']]['feature'].tolist()
        print(f"\n✓ Features for model: {keepers}")

        # Save keeper list for Day 8
        os.makedirs('data/processed', exist_ok=True)
        with open('data/processed/keeper_features.json', 'w') as f:
            json.dump(keepers, f)
        print(f"✓ Saved keeper list to data/processed/keeper_features.json")

        plot_ic(results_df, merged, horizon)
        interpret(results_df, merged, horizon)
    else:
        print("No features computed — check data.")