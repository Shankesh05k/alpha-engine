import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_prices():
    prices = pd.read_csv('data/raw/prices.csv', index_col=0, parse_dates=True)
    return prices

def compute_forward_returns(prices, forward_days=30):
    """
    For each stock on each date, what was the return over the next N days?
    shift(-forward_days) means "look forward" — critical to avoid lookahead bias.
    """
    # Drop the index from stock columns
    stocks = prices.drop(columns=['^CNXIT'], errors='ignore')
    index  = prices['^CNXIT'] if '^CNXIT' in prices.columns else None

    # Forward return for each stock
    fwd_returns = stocks.pct_change(forward_days).shift(-forward_days)

    # Excess return = stock return - index return
    if index is not None:
        index_fwd = index.pct_change(forward_days).shift(-forward_days)
        excess_returns = fwd_returns.sub(index_fwd, axis=0)
    else:
        excess_returns = fwd_returns

    return fwd_returns, excess_returns

def summarise(excess_returns):
    """
    Print basic stats — helps you understand the distribution
    of returns you're trying to predict.
    """
    flat = excess_returns.stack().dropna()
    print("=== Excess Return Distribution (30d) ===")
    print(f"Total observations : {len(flat)}")
    print(f"Mean excess return : {flat.mean():.2%}")
    print(f"Std deviation      : {flat.std():.2%}")
    print(f"% positive (beat index) : {(flat > 0).mean():.1%}")
    print(f"% strongly positive (>2%): {(flat > 0.02).mean():.1%}")
    print(f"% strongly negative (<-2%): {(flat < -0.02).mean():.1%}")

def plot_return_distribution(excess_returns):
    flat = excess_returns.stack().dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of all excess returns
    axes[0].hist(flat, bins=80, color='steelblue', alpha=0.7, edgecolor='none')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[0].axvline(0.02, color='green', linestyle='--', linewidth=1, label='>2% threshold')
    axes[0].axvline(-0.02, color='orange', linestyle='--', linewidth=1, label='<-2% threshold')
    axes[0].set_title('Distribution of 30-day excess returns')
    axes[0].set_xlabel('Excess return vs Nifty IT')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Per-stock mean excess return
    per_stock = excess_returns.mean().sort_values()
    colors = ['#E24B4A' if v < 0 else '#1D9E75' for v in per_stock]
    axes[1].barh(
        [t.replace('.NS','') for t in per_stock.index],
        per_stock.values * 100,
        color=colors
    )
    axes[1].axvline(0, color='gray', linewidth=0.8)
    axes[1].set_title('Mean 30-day excess return per stock (%)')
    axes[1].set_xlabel('Mean excess return (%)')

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/return_distribution.png', dpi=150)
    print("\n✓ Chart saved to outputs/return_distribution.png")
    plt.show()

def save(fwd_returns, excess_returns):
    os.makedirs('data/processed', exist_ok=True)
    fwd_returns.to_csv('data/processed/forward_returns_30d.csv')
    excess_returns.to_csv('data/processed/excess_returns_30d.csv')
    print("✓ Saved forward_returns_30d.csv")
    print("✓ Saved excess_returns_30d.csv")

if __name__ == '__main__':
    prices = load_prices()
    print(f"Loaded prices: {prices.shape[0]} days, {prices.shape[1]} tickers")

    fwd_returns, excess_returns = compute_forward_returns(prices, forward_days=30)
    summarise(excess_returns)
    plot_return_distribution(excess_returns)
    save(fwd_returns, excess_returns)