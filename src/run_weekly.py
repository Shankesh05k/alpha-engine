import subprocess
import datetime

print(f"=== Weekly pipeline — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")

steps = [
    ("Scraping Naukri job postings",  "python src/scraper.py"),
    ("Computing job features",         "python src/features.py"),
    ("Fetching BSE filings",           "python src/bse_filings.py"),
    ("Merging all signals",            "python src/merge_signals.py"),
    ("Running IC analysis",            "python src/ic_analysis.py"),
    ("Training model + backtest",      "python src/model.py"),
    ("Running permutation test",       "python src/permutation_test.py"),
]

print("Steps this run:")
for name, _ in steps:
    print(f"  • {name}")
print()

failed = False
for name, cmd in steps:
    print(f"\n{'='*50}")
    print(f"▶  {name}")
    print(f"{'='*50}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n⚠️  '{name}' failed — stopping pipeline.")
        failed = True
        break

print(f"\n{'='*50}")
if failed:
    print("Pipeline finished with errors. Check output above.")
else:
    print("✓ Pipeline complete.")
    print("  Run: streamlit run dashboard.py")
print(f"  Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*50}")