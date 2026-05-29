import asyncio
import json
import os
import sqlite3
import re
import pandas as pd
from datetime import datetime, timedelta
from playwright.async_api import async_playwright

# ── Company map ────────────────────────────────────────────────────────────
COMPANIES = {
    'TCS.NS':        'tata-consultancy-services',
    'INFY.NS':       'infosys',
    'WIPRO.NS':      'wipro',
    'HCLTECH.NS':    'hcl-technologies',
    'TECHM.NS':      'tech-mahindra',
    'LTIMINDTREE.NS':       'ltimindtree',
    'MPHASIS.NS':    'mphasis',
    'PERSISTENT.NS': 'persistent-systems',
    'COFORGE.NS':    'coforge',
    'KPITTECH.NS':   'kpit-technologies',
    'TATAELXSI.NS':  'tata-elxsi',
    'BSOFT.NS':      'birlasoft',
    'MASTEK.NS':     'mastek',
    'ZENSARTECH.NS': 'zensar-technologies',
    'SONATSOFTW.NS': 'sonata-software',
    'INTELLECT.NS':  'intellect-design-arena',
    'HAPPSTMNDS.NS': 'happiest-minds',
    'CYIENT.NS':     'cyient',
    'LTTS.NS':       'l-and-t-technology-services',
    'OFSS.NS':       'oracle-financial-services-software',
}

# ── Database ───────────────────────────────────────────────────────────────
def init_db(db_path='data/jobs.db'):
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker       TEXT NOT NULL,
            company      TEXT NOT NULL,
            title        TEXT,
            skills       TEXT,
            experience   TEXT,
            location     TEXT,
            posted_raw   TEXT,
            posted_date  TEXT,
            scraped_at   TEXT NOT NULL,
            UNIQUE(ticker, title, posted_raw)
        )
    ''')
    conn.commit()
    print("✓ Database ready")
    return conn

# ── Date parser ────────────────────────────────────────────────────────────
def parse_posted_date(raw, scraped_at):
    scraped = pd.Timestamp(scraped_at)
    raw = str(raw).lower().strip()

    if any(x in raw for x in ['just now', 'few hours', 'today', 'hour']):
        return scraped.date().isoformat()

    m = re.search(r'(\d+)\+?\s*day', raw)
    if m:
        return (scraped - timedelta(days=int(m.group(1)))).date().isoformat()

    m = re.search(r'(\d+)\+?\s*week', raw)
    if m:
        return (scraped - timedelta(weeks=int(m.group(1)))).date().isoformat()

    m = re.search(r'(\d+)\s*month', raw)
    if m:
        return (scraped - timedelta(days=int(m.group(1)) * 30)).date().isoformat()

    return scraped.date().isoformat()

# ── Scraper ────────────────────────────────────────────────────────────────
async def scrape_company(page, ticker, company_slug, max_pages=3):
    results = []
    scraped_at = datetime.now().isoformat()

    for pg in range(1, max_pages + 1):
        url = f"https://www.naukri.com/{company_slug}-jobs?pageNo={pg}"
        print(f"  Fetching page {pg}: {url}")

        try:
            await page.goto(url, wait_until='domcontentloaded', timeout=45000)

            try:
                await page.wait_for_selector('div.cust-job-tuple', timeout=15000)
            except Exception:
                print(f"  Cards never appeared on page {pg} — stopping")
                break

            cards = await page.query_selector_all('div.cust-job-tuple')
            print(f"  Found {len(cards)} cards")

            if not cards:
                break

            page_count = 0
            for card in cards:
                try:
                    # Title
                    title_el = await card.query_selector('a.title')
                    title = (await title_el.inner_text()).strip() if title_el else 'N/A'
                    if title == 'N/A':
                        continue

                    # Date
                    date_el = await card.query_selector('span.job-post-day')
                    posted_raw = (await date_el.inner_text()).strip() if date_el else 'N/A'

                    # Skills
                    tag_els = await card.query_selector_all('li.tag-li')
                    skills = []
                    for t in tag_els:
                        s = await t.inner_text()
                        if s and s.strip():
                            skills.append(s.strip())

                    # Experience
                    exp_el = await card.query_selector('span.expwdth')
                    experience = (await exp_el.inner_text()).strip() if exp_el else 'N/A'

                    # Location
                    loc_el = await card.query_selector('span.locWdth')
                    location = (await loc_el.inner_text()).strip() if loc_el else 'N/A'

                    results.append({
                        'ticker':      ticker,
                        'company':     company_slug,
                        'title':       title,
                        'skills':      json.dumps(skills),
                        'experience':  experience,
                        'location':    location,
                        'posted_raw':  posted_raw,
                        'posted_date': parse_posted_date(posted_raw, scraped_at),
                        'scraped_at':  scraped_at,
                    })
                    page_count += 1

                except Exception:
                    continue

            print(f"  Page {pg}: {page_count} jobs extracted")

            if page_count == 0:
                break

            await asyncio.sleep(2)

        except Exception as e:
            print(f"  Error on page {pg}: {e}")
            break

    return results

# ── DB insert ──────────────────────────────────────────────────────────────
def insert_jobs(conn, jobs):
    inserted = 0
    for job in jobs:
        try:
            conn.execute('''
                INSERT OR IGNORE INTO jobs
                (ticker, company, title, skills, experience,
                 location, posted_raw, posted_date, scraped_at)
                VALUES (?,?,?,?,?,?,?,?,?)
            ''', (
                job['ticker'], job['company'], job['title'],
                job['skills'], job['experience'], job['location'],
                job['posted_raw'], job['posted_date'], job['scraped_at']
            ))
            inserted += 1
        except Exception as e:
            print(f"  DB error: {e}")
    conn.commit()
    return inserted

# ── DB summary ─────────────────────────────────────────────────────────────
def check_db():
    conn = sqlite3.connect('data/jobs.db')
    df = pd.read_sql('''
        SELECT ticker,
               COUNT(*)         as total_jobs,
               MIN(posted_date) as earliest_post,
               MAX(scraped_at)  as last_scraped
        FROM jobs
        GROUP BY ticker
        ORDER BY total_jobs DESC
    ''', conn)
    conn.close()
    print("\n=== Jobs DB Summary ===")
    print("No data yet." if df.empty else df.to_string(index=False))
    return df

# ── Runner ─────────────────────────────────────────────────────────────────
async def run_scraper(companies=None, max_pages=3):
    if companies is None:
        companies = COMPANIES

    conn = init_db()
    total = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1280, 'height': 800}
        )
        page = await context.new_page()

        for ticker, slug in companies.items():
            print(f"\nScraping {ticker} ({slug})...")
            jobs = await scrape_company(page, ticker, slug, max_pages=max_pages)
            inserted = insert_jobs(conn, jobs)
            total += inserted
            print(f"  ✓ {len(jobs)} scraped, {inserted} new inserted")
            await asyncio.sleep(3)

        await browser.close()

    conn.close()
    print(f"\n✓ Total new records: {total}")

# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== FULL RUN — all 20 companies, 3 pages each ===")
    asyncio.run(run_scraper(companies=COMPANIES, max_pages=3))
    check_db()