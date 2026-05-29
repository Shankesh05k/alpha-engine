import subprocess
import datetime

print(f"=== Weekly pipeline run — {datetime.datetime.now()} ===\n")

steps = [
    ("Scraping job postings",   "python src/scraper.py"),
    ("Computing features",       "python src/features.py"),
    ("Merging all signals",      "python src/merge_signals.py"),
    ("Running IC analysis",      "python src/ic_analysis.py"),
    ("Running model",            "python src/model.py"),
]

for name, cmd in steps:
    print(f"\n{'='*50}")
    print(f"Running: {name}")
    print(f"{'='*50}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"⚠️  {name} failed — check error above")
        break

print(f"\n=== Done — {datetime.datetime.now()} ===")