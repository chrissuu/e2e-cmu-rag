import os

def safe_get(key: str):
    val_opt = os.getenv(key)

    if val_opt:
        return val_opt
    else:
        raise ValueError(f"Env variable {key} was not found.")

REPO_ROOT_PATH = safe_get("E2E_CMU_RAG")
DATA_ROOT = f"{REPO_ROOT_PATH}/data"
RAW_DATA_ROOT = f"{REPO_ROOT_PATH}/data/raw"
CLEANED_DATA_ROOT = f"{REPO_ROOT_PATH}/data/cleaned"
ANNOTATION_DATA_ROOT = f"{REPO_ROOT_PATH}/data/to-annotate/"

WIKIPEDIA_REQUEST_HEADER = {
    "User-Agent": "11711 ANLP HW 2 (contact: chrissu@andrew.cmu.edu)"
}
WIKI_BASE = "https://en.wikipedia.org"

SEEDS_ROOT = f"{REPO_ROOT_PATH}/src/seeds"