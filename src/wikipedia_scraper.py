import os
import re
import requests
from bs4 import BeautifulSoup
from typing import List
from urllib.parse import urljoin, unquote

from constants import RAW_DATA_ROOT, WIKIPEDIA_REQUEST_HEADER, WIKI_BASE

def find_neighbors(wikipedia_link: str) -> List[str]:
    """
    find_neighbors

    param: 
        wikipedia_link (str): the wikipedia link to find neighbors from

    returns:
        List[str]: the (deduplicated) neighboring wikipedia links
    """

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


def get_wikipedia_info(wikipedia_links: List[str], folder_path: str):
    os.makedirs(folder_path, exist_ok=True)

    seen = set()
    for link in wikipedia_links:
        if link in seen:
            continue
        seen.add(link)

        try:
            response = requests.get(link, headers=WIKIPEDIA_REQUEST_HEADER)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to fetch {link}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.find("h1", {"id": "firstHeading"})
        if not title:
            continue
        title_text = title.text.strip()
        safe_title = re.sub(r"[^a-zA-Z0-9_\-]", "_", title_text)  # safe filename

        content_div = soup.find("div", {"id": "mw-content-text"})
        if not content_div:
            continue
        paragraphs = content_div.find_all("p")
        text_content = "\n".join(p.get_text() for p in paragraphs if p.get_text().strip())

        file_path = os.path.join(folder_path, f"{safe_title}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_content)

        print(f"Saved: {file_path}")

start = "https://en.wikipedia.org/wiki/Carnegie_Mellon_University"
neighbors = find_neighbors(start)
neighbors.append(start)
print(f"Found {len(neighbors)} neighbors")
    
get_wikipedia_info(neighbors, f"{RAW_DATA_ROOT}/wikipedia/cmu-one-jump/")

f = open(f"{RAW_DATA_ROOT}/wikipedia/cmu-one-jump/___info___.txt", "w")
f.write(
f"""
Data collected with node and one neighbor from node.

Node: {start}
Number of neighbors: {len(neighbors)}
""")