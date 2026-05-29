import os

old = 'LTIM.NS'
new = 'LTIMINDTREE.NS'

files = [
    'src/prices.py',
    'src/scraper.py',
    'src/features.py',
    'src/merge_signals.py',
    'src/ic_analysis.py',
    'src/returns.py',
]

for f in files:
    if not os.path.exists(f):
        print(f"  skipping {f} — not found")
        continue
    content = open(f, encoding='utf-8').read()
    if old in content:
        content = content.replace(old, new)
        open(f, 'w', encoding='utf-8').write(content)
        print(f"✓ Fixed {f}")
    else:
        print(f"  {f} — no change needed")

print("\nDone. Now run: python src/prices.py")