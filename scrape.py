from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright  # Add this import
from bs4 import BeautifulSoup
import httpx
import json
import asyncio
from datetime import datetime
from typing import Dict, List
import time

def scrape_all_topics_with_playwright(main_url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(main_url)
        page.wait_for_timeout(3000)
        html = page.content()
        browser.close()
    soup = BeautifulSoup(html, "html.parser")
    topic_links = [a['href'] for a in soup.find_all('a', href=True)]
    return topic_links

def scrape_topics_with_playwright(main_url, topic_hrefs):
    data = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(main_url)
        page.wait_for_timeout(500)
        for href in topic_hrefs:
            page.evaluate(f"window.location.hash = '{href}'")
            page.wait_for_timeout(1200)
            html = page.content()
            soup = BeautifulSoup(html, "html.parser")
            main_content = soup.find("section", class_="content") or soup.find("article", class_="markdown-section")
            if main_content:
                data[href] = main_content.get_text(separator="\n", strip=True)
            else:
                data[href] = ""
        browser.close()
    return data

async def scrape_discourse_topics(category_id: int, start_date: str, end_date: str) -> List[Dict]:
    """Fetch topics from Discourse within date range using Playwright."""
    
    DISCOURSE_URL = "https://discourse.onlinedegree.iitm.ac.in"
    all_topics = []
    MAX_PAGES = 10  # Limit to first 10 pages
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=False,
                slow_mo=100
            )
            context = await browser.new_context()
            page = await context.new_page()
            await page.set_viewport_size({"width": 1280, "height": 720})
            
            print("Navigating to login page...")
            await page.goto(f"{DISCOURSE_URL}/login")
            print("Please log in manually in the browser window...")
            await page.wait_for_selector('.current-user', timeout=300000)
            await context.storage_state(path="auth.json")
            
            # Fetch first 10 pages of topics
            for page_number in range(MAX_PAGES):
                print(f"Fetching page {page_number + 1} of {MAX_PAGES}...")
                url = f"{DISCOURSE_URL}/c/courses/tds-kb/34/l/latest.json?page={page_number}"
                await page.goto(url)
                await page.wait_for_load_state('networkidle')
                
                data = await page.evaluate("""() => {
                    const pre = document.querySelector('pre');
                    return pre ? JSON.parse(pre.textContent) : null;
                }""")
                
                if not data or 'topic_list' not in data or not data['topic_list'].get('topics'):
                    print(f"No more topics found on page {page_number + 1}")
                    break
                
                page_topics = data['topic_list'].get('topics', [])
                all_topics.extend(page_topics)
                print(f"Found {len(page_topics)} topics on page {page_number + 1}")
                
                # Short pause between pages
                await page.wait_for_timeout(1000)
            
            print(f"Found total of {len(all_topics)} topics before filtering")
            
            # Filter topics by date
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            
            filtered_topics = [
                topic for topic in all_topics
                if start <= datetime.strptime(topic["created_at"][:10], "%Y-%m-%d") <= end
            ]
            
            print(f"Filtered to {len(filtered_topics)} topics within date range")
            await browser.close()
            return filtered_topics
                
    except Exception as e:
        print(f"Error fetching topics: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return []

async def scrape_all_discourse_content(start_date: str = "2025-01-01", 
                                     end_date: str = "2025-04-14") -> Dict[str, Dict]:
    """Scrape Discourse content within date range."""
    discourse_data = {}
    
    # Example: Fetch topics from category 5 (adjust as needed)
    topics = await scrape_discourse_topics(5, start_date, end_date)
    
    for topic in topics:
        topic_data = await scrape_discourse_topic(topic["id"])
        created_at = datetime.strptime(topic["created_at"][:10], "%Y-%m-%d")
        
        # Only include topics within date range
        if created_at >= datetime.strptime(start_date, "%Y-%m-%d") and \
           created_at <= datetime.strptime(end_date, "%Y-%m-%d"):
            
            discourse_data[f"discourse/{topic['id']}"] = {
                "title": topic["title"],
                "url": f"https://discourse.onlinedegree.iitm.ac.in/t/{topic['slug']}/{topic['id']}",
                "content": "\n".join([
                    BeautifulSoup(post.get("cooked", ""), "html.parser").get_text(separator=" ", strip=True)
                    for post in topic_data.get("post_stream", {}).get("posts", [])
                ]),
                "created_at": topic["created_at"]
            }
    
    return discourse_data

async def scrape_discourse_topic(topic_id: int) -> Dict:
    """Fetch a single topic and its content."""
    DISCOURSE_URL = "https://discourse.onlinedegree.iitm.ac.in"
    
    try:
        # Re-use the same browser context that was authenticated earlier
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            # Load the stored authentication state
            context = await browser.new_context(storage_state="auth.json")
            page = await context.new_page()
            
            # Navigate to the topic JSON endpoint
            url = f"{DISCOURSE_URL}/t/{topic_id}.json"
            await page.goto(url)
            await page.wait_for_load_state('networkidle')
            
            # Get the JSON content
            data = await page.evaluate("""() => {
                const pre = document.querySelector('pre');
                return pre ? JSON.parse(pre.textContent) : null;
            }""")
            
            await browser.close()
            return data if data else {}
            
    except Exception as e:
        print(f"Error fetching topic {topic_id}: {str(e)}")
        return {}

if __name__ == "__main__":
    print("Starting scraping process...")
    
    # Scrape course content
    main_url = "https://tds.s-anand.net/#/README"
    print("Scraping course content...")
    topic_hrefs = [href for href in scrape_all_topics_with_playwright(main_url) if href.startswith("#/")]
    print(f"Found {len(topic_hrefs)} course topics")
    scraped_topic_data = scrape_topics_with_playwright(main_url, topic_hrefs)
    print(f"Scraped {len(scraped_topic_data)} course topics")
    
    # Scrape Discourse content
    print("\nScraping Discourse content...")
    try:
        discourse_data = asyncio.run(scrape_all_discourse_content(
            start_date="2025-01-01",
            end_date="2025-04-14"
        ))
        print(f"Scraped {len(discourse_data)} Discourse topics")
    except Exception as e:
        print(f"Error scraping Discourse: {str(e)}")
        discourse_data = {}
    
    # Combine both data sources
    combined_data = {**scraped_topic_data, **discourse_data}
    print(f"\nTotal items in combined data: {len(combined_data)}")
    print(f"Course topics: {len(scraped_topic_data)}")
    print(f"Discourse topics: {len(discourse_data)}")
    
    # Save combined data
    output_file = "scraped_data.json"
    print(f"\nSaving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    print("Done!")