import json
import re

from constants import *
# ===== CONFIGURATION - CHANGE THESE PATHS =====
INPUT_FILE = f"{REPO_ROOT_PATH}/chrissu/system_outputs/system_output_1.json"
OUTPUT_FILE = f"{REPO_ROOT_PATH}/chrissu/system_outputs/system_output_2.json"
# ==============================================

def clean_answer(text):
    """
    Extract the core answer by removing explanations and formatting markers.
    """
    if not text or text.strip() == '""':
        return ""
    text = text.strip()
    if text == "Not found.":
        return "Not found."
    separators = [
        r'\n\nExplanation:',
        r'\nExplanation:',
        r'={3,}\s*END ANSWER\s*={3,}',
        r'={3,}\s*DOCUMENTS\s*={3,}'
    ]
    
    for separator in separators:
        parts = re.split(separator, text, maxsplit=1, flags=re.IGNORECASE)
        text = parts[0]
    
    text = re.sub(r'={3,}.*?={3,}', '', text, flags=re.DOTALL)
    text = text.strip()
    
    return text

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

cleaned_data = {}
for key, value in data.items():
    cleaned_data[key] = clean_answer(value)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, indent=2, ensure_ascii=False)