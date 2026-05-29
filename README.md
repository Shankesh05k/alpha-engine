# Alpha Engine — Alternative Data Signals for Indian IT Stocks

Predicts short-term stock outperformance using job postings, Google Trends, 
and skill mix analysis. Built from scratch in 2 weeks.

---

## The Idea

IT companies hire before contracts start — weeks before revenue shows up in 
earnings. Tracking *what* and *how fast* they're hiring on Naukri, combined 
with retail search interest, gives an early signal on which stocks will 
outperform the Nifty IT index.

---

## Signals

| Signal | Source | IC (7d) | Direction |
|---|---|---|---|
| Cloud hiring velocity | Naukri.com | +0.08 | More cloud = outperform |
| Digital skill share | Naukri.com | +0.17 | More digital = outperform |
| AI/ML hiring velocity | Naukri.com | -0.37 | Rapid AI hiring = underperform |
| Search interest spike | Google Trends | +0.25 | Search spike = outperform |

The most interesting finding: **AI/ML hiring is a negative signal**. 
Companies suddenly hiring lots of AI engineers are signalling cost pressure, 
not revenue growth.

---

## Results (3 weeks of data)

| Metric | Value | Baseline |
|---|---|---|
| Accuracy | 60% | 44% |
| Top pick hit rate | 100% | 63.6% |
| Top decile return | 3.20% | 2.67% |

Not statistically significant yet — need 10+ test weeks. 
Results are directionally correct. Accumulating evidence weekly.

## How It Works
Naukri scraper → Feature engineering → IC filter → GBM model → Live signal
Google Trends  ↗   ↗ Walk-forward backtest

---

## Stack

Python · Playwright · scikit-learn · pytrends · yfinance · Streamlit · SQLite

---

## Setup

```bash
pip install pandas numpy matplotlib yfinance scikit-learn scipy playwright pytrends streamlit tqdm
playwright install chromium

python src/prices.py
python src/scraper.py
python src/features.py
python src/merge_signals.py
python src/model.py

streamlit run dashboard.py
```

**Weekly update:**
```bash
python run_weekly.py
```

---

## Universe

20 Nifty IT stocks — TCS, Infosys, Wipro, HCL Tech, Tech Mahindra, 
LTIMindtree, Mphasis, Persistent, Coforge, KPIT, Tata Elxsi, Birlasoft, 
Mastek, Zensar, Sonata, Intellect Design, Happiest Minds, Cyient, LTTS, OFSS

---

## Limitations

- Only 3 weeks of job posting history — backtest strengthens with more data
- Naukri doesn't capture LinkedIn or referral hiring
- No transaction costs in backtest
- Research project only — not financial advice

---

## Roadmap

- [x] Naukri scraper + feature pipeline
- [x] Google Trends integration
- [x] Walk-forward backtest + permutation testing
- [x] Streamlit dashboard
- [ ] BSE corporate filings (third signal)
- [ ] 20+ weeks of data for significant results

---

*Bangalore · May 2026*
---

