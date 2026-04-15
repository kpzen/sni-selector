import requests
from bs4 import BeautifulSoup
import json
import re
import os
import sys

# 1. Read URLs from the text file
url_file = "urls.txt"

# Quick safety check to make sure the file exists
if not os.path.exists(url_file):
    print(f"Error: Could not find '{url_file}'. Please make sure it is in the same folder as this script.")
    sys.exit(1)

with open(url_file, "r", encoding="utf-8") as f:
    # Read each line, strip away newline characters/whitespace, and ignore empty lines
    URLS = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(URLS)} URLs from {url_file}. Starting scraper...")

def get_list_under_header(header_element):
    """
    Extracts text from all <li> elements following a specific <h2> header,
    until it hits the next section (like an <hr> or another <h2>).
    """
    items = []
    if not header_element:
        return items
    
    # Loop through all sibling tags that come after this header
    for sibling in header_element.find_next_siblings():
        # Stop looking if we hit a new section divider or header
        if sibling.name in ['h2', 'hr']:
            break
        
        # If it's an unordered list, extract all list items
        if sibling.name == 'ul':
            for li in sibling.find_all('li'):
                # separator=" " prevents inline tags (like <a>) from merging with text
                text = li.get_text(separator=" ", strip=True)
                if text:
                    # Clean up any accidental double spaces created by the separator
                    text = re.sub(r'\s+', ' ', text)
                    items.append(text)
                    
    return items

database = []

for url in URLS:
    try:
        # We set a timeout to prevent the script from hanging on bad connections
        res = requests.get(url, timeout=10)
        
        # Explicitly set the encoding to UTF-8 to handle Swedish characters natively
        res.encoding = 'utf-8' 
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 1. Parse SNI Code and Title from the <h1> tag
        h1_element = soup.find('h1')
        if not h1_element:
            print(f"Skipping {url}: No <h1> tag found.")
            continue
            
        h1_text = h1_element.get_text(separator=" ", strip=True)
        # Splits "29.101 - Tillverkning..." into ["29.101", "Tillverkning..."]
        parts = h1_text.split(" - ", 1) 
        sni_code = parts[0].strip() if len(parts) > 0 else ""
        title = parts[1].strip() if len(parts) > 1 else ""

        # 2. Locate the specific <h2> headers
        h2_covers = soup.find('h2', string=lambda text: text and "Omfattar" == text.strip())
        h2_also_covers = soup.find('h2', string=lambda text: text and "Omfattar även" in text.strip())
        h2_not_covers = soup.find('h2', string=lambda text: text and "Omfattar inte" in text.strip())
        h2_examples = soup.find('h2', string=lambda text: text and "Exempel på vad som ingår" in text.strip())

        # 3. Build the JSON Object for this URL
        sni_entry = {
            "sni": sni_code,
            "title": title,
            "url": url,
            "covers": get_list_under_header(h2_covers),
            "also_covers": get_list_under_header(h2_also_covers),
            "does_not_cover": get_list_under_header(h2_not_covers),
            "examples": get_list_under_header(h2_examples)
        }
        
        database.append(sni_entry)
        print(f"Successfully scraped: {sni_code} - {title}")
        
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")

# Save the raw data to JSON, preserving Swedish characters
with open("raw_data.json", "w", encoding="utf-8") as f:
    json.dump(database, f, indent=2, ensure_ascii=False)
    
print(f"\nDone! Saved {len(database)} entries to raw_data.json")