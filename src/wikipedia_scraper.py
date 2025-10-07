import os
import re
import requests
from bs4 import BeautifulSoup
from typing import List
from urllib.parse import urljoin, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed

from constants import RAW_DATA_ROOT, WIKIPEDIA_REQUEST_HEADER, WIKI_BASE

"""
Author: Chris Su
Date: Oct. 4, 2025
Program Name: Multithreaded Wikipedia Scraper

Graph based wikipedia scraper which scrapes a wikipedia
webpage and the neighboring outgoing links from that wikipedia
webpage.

Multithreaded for faster processing.
"""

def find_neighbors(wikipedia_link: str) -> List[str]:
    response = requests.get(wikipedia_link, headers=WIKIPEDIA_REQUEST_HEADER)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    neighbors = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.startswith("/wiki/") and not any(
            href.startswith(prefix) for prefix in [
                "/wiki/Special:", "/wiki/Help:", "/wiki/Category:", "/wiki/File:", "/wiki/Portal:"
            ]
        ):
            full_url = urljoin(WIKI_BASE, href)
            neighbors.append(full_url)

    return list(set(neighbors))


def process_link(link: str, folder_path: str):
    try:
        response = requests.get(link, headers=WIKIPEDIA_REQUEST_HEADER)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {link}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.find("h1", {"id": "firstHeading"})
    if not title:
        return None
    title_text = title.text.strip()
    safe_title = re.sub(r"[^a-zA-Z0-9_\-]", "_", title_text)

    content_div = soup.find("div", {"id": "mw-content-text"})
    if not content_div:
        return None
    paragraphs = content_div.find_all("p")
    text_content = "\n".join(p.get_text() for p in paragraphs if p.get_text().strip())

    file_path = os.path.join(folder_path, f"{safe_title}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    print(f"Saved: {file_path}")
    return file_path


def get_wikipedia_info(wikipedia_links: List[str], folder_path: str, max_workers: int = 10):
    os.makedirs(folder_path, exist_ok=True)
    seen = set()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for link in wikipedia_links:
            if link in seen:
                continue
            seen.add(link)
            futures.append(executor.submit(process_link, link, folder_path))

        for future in as_completed(futures):
            future.result()

start = "https://en.wikipedia.org/wiki/Pittsburgh"
neighbors = find_neighbors(start)
neighbors.append(start)
print(f"Found {len(neighbors)} neighbors")

get_wikipedia_info(neighbors, f"{RAW_DATA_ROOT}/wikipedia/Pittsburgh-one-jump/")

with open(f"{RAW_DATA_ROOT}/wikipedia/Pittsburgh-one-jump/___info___.txt", "w") as f:
    f.write(
        f"""
Data collected with node and one neighbor from node.

Node: {start}
Number of neighbors: {len(neighbors)}
""")
