"""
One Jump Scraper

Main scraping script.

This scraping script IS multithreaded so
that it runs a bit faster.

scrape_webpage is the main endpoint that this
scraper will call.

In general, there are two types of scrapes:

    One Jump Scrape
    Zero Jump Scrape

The pipeline is as follows.

Given a list of hyperlinks:

    one_jump_links
    zero_jump_links

The scraper will first collect all links:

    links := zero_jump_links + one_jump_links + 
                get_one_jump_links(one_jump_links)

Where get_one_jump_links will return all links in the
webpage for link in one_jump_links.

It will deduplicate by passing this into a set.

Finally, it will call the scrape_webpage function
for each link in this set, and store it in the
specified folder path.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set
import time

from constants import *


def scrape_webpage(link: str, folder_path: str) -> tuple[bool, str]:
    """
    Scrapes link and stores ONLY TEXT to folder_path with a descriptive name.
    Ensures name is unique before writing to it, so that you don't overwrite.
    
    Args:
        link: URL to scrape
        folder_path: Directory path to save the scraped content
    
    Returns:
        tuple[bool, str]: (Success status, error message if failed)
    """
    try:
        # Create folder if it doesn't exist
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        
        # Fetch the webpage
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(link, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse HTML and extract text only
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Generate a descriptive filename from the URL
        parsed_url = urlparse(link)
        domain = parsed_url.netloc.replace('www.', '')
        path = parsed_url.path.strip('/').replace('/', '_')
        
        # Create base filename
        if path:
            base_name = f"{domain}_{path}"
        else:
            base_name = domain
        
        # Clean the filename (remove invalid characters)
        base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
        base_name = base_name[:200]  # Limit length
        
        # Ensure unique filename (use .txt extension now)
        file_path = Path(folder_path) / f"{base_name}.txt"
        counter = 1
        while file_path.exists():
            file_path = Path(folder_path) / f"{base_name}_{counter}.txt"
            counter += 1
        
        # Write text content to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"✓ Scraped: {link}")
        return True, ""
        
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Failed to scrape {link}: {error_msg}")
        return False, error_msg


def one_jump_collector(link: str) -> List[str]:
    """
    Returns all outgoing hyperlinks/links from link.
    
    Args:
        link: URL to collect links from
    
    Returns:
        list[str]: List of absolute URLs found on the page
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(link, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        links = []
        
        # Find all anchor tags with href attribute
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            # Convert relative URLs to absolute URLs
            absolute_url = urljoin(link, href)
            
            # Only include http/https links
            if absolute_url.startswith(('http://', 'https://')):
                links.append(absolute_url)
        
        print(f"✓ Collected {len(links)} links from: {link}")
        return links
        
    except Exception as e:
        print(f"✗ Failed to collect links from {link}: {str(e)}")
        return []


def scrape(zero_jump_links: List[str], one_jump_links: List[str], 
           scrape_info_path: str, folder_path: str, max_workers: int = 10):
    """
    Collects all links from zero_jump_links and all links from one_jump_links 
    and one_jump_collector(link) for link in one_jump_links.
    
    Passes it into a set to deduplicate, then sorts the links in alphabetical order.
    These links are then written to the file scrape_info_path.
    
    Finally, scrape_webpage(link, folder_path) is called for each link in this set.
    
    Creates a log file tracking successful and failed scrapes.
    
    Args:
        zero_jump_links: List of links to scrape directly
        one_jump_links: List of links to collect outgoing links from
        scrape_info_path: Path to write the final list of links
        folder_path: Directory to save scraped webpages
        max_workers: Number of threads for parallel processing (default: 10)
    """
    print(f"\n{'='*60}")
    print(f"Starting scraping process...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    print(f"Collecting links from one_jump_links...")
    all_links: Set[str] = set(zero_jump_links)
    all_links.update(one_jump_links)
    
    collected_links = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_link = {executor.submit(one_jump_collector, link): link 
                         for link in one_jump_links}
        
        for future in as_completed(future_to_link):
            try:
                links = future.result()
                collected_links.extend(links)
            except Exception as e:
                print(f"✗ Error processing link: {str(e)}")
    
    all_links.update(collected_links)
    
    print(f"\nDeduplicating and sorting {len(all_links)} links...")
    sorted_links = sorted(list(all_links))
    
    print(f"\nWriting links to {scrape_info_path}...")
    Path(scrape_info_path).parent.mkdir(parents=True, exist_ok=True)
    with open(scrape_info_path, 'w', encoding='utf-8') as f:
        for link in sorted_links:
            f.write(f"{link}\n")
    print(f"✓ Wrote {len(sorted_links)} links to {scrape_info_path}")
    
    print(f"\nScraping {len(sorted_links)} webpages...")
    successful_scrapes = []
    failed_scrapes = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_link = {executor.submit(scrape_webpage, link, folder_path): link 
                         for link in sorted_links}
        
        for future in as_completed(future_to_link):
            link = future_to_link[future]
            try:
                success, error_msg = future.result()
                if success:
                    successful_scrapes.append(link)
                else:
                    failed_scrapes.append((link, error_msg))
            except Exception as e:
                failed_scrapes.append((link, str(e)))
                print(f"✗ Error: {str(e)}")
    
    # Step 5: Write scrape results log
    log_path = Path(scrape_info_path).parent / "scrape_results.txt"
    print(f"\nStep 5: Writing scrape results to {log_path}...")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("SCRAPE RESULTS LOG\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total links attempted: {len(sorted_links)}\n")
        f.write(f"Successfully scraped: {len(successful_scrapes)}\n")
        f.write(f"Failed: {len(failed_scrapes)}\n\n")
        
        f.write("="*60 + "\n")
        f.write("SUCCESSFUL SCRAPES\n")
        f.write("="*60 + "\n")
        for link in sorted(successful_scrapes):
            f.write(f"✓ {link}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("FAILED SCRAPES\n")
        f.write("="*60 + "\n")
        for link, error in sorted(failed_scrapes):
            f.write(f"✗ {link}\n")
            f.write(f"  Error: {error}\n\n")
    
    print(f"✓ Wrote scrape results to {log_path}")
    
    # Summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Scraping complete!")
    print(f"{'='*60}")
    print(f"Total links processed: {len(sorted_links)}")
    print(f"Successfully scraped: {len(successful_scrapes)}")
    print(f"Failed: {len(failed_scrapes)}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Average time per page: {elapsed_time/len(sorted_links):.2f} seconds")
    print(f"Results saved to: {log_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    zero_jump = []
    zero_jump_seeds = open(f"{SEEDS_ROOT}/zero_jump_seeds.txt", "r")
    zero_jump.extend(zero_jump_seeds.readlines())
    zero_jump_seeds.close()
    zero_jump_seeds = open(f"{SEEDS_ROOT}/manual_seeds.txt", "r")
    zero_jump.extend(zero_jump_seeds.readlines())

    one_jump = []
    one_jump_seeds = open(f"{SEEDS_ROOT}/one_jump_seeds.txt", "r")
    one_jump.extend(one_jump_seeds.readlines())
    one_jump_seeds.close()

    scrape(
        zero_jump_links=zero_jump,
        one_jump_links=one_jump,
        scrape_info_path=f"{DATA_ROOT}/scrape_output_oct10/links.txt",
        folder_path=f"{DATA_ROOT}/scrape_output_oct10/pages",
        max_workers=8
    )