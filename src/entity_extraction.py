import re
from typing import List, Dict
from datetime import datetime

# Example complaint keywords (expand as needed)
COMPLAINT_KEYWORDS = [
    'broken', 'late', 'error', 'not working', 'failed', 'missing', 'damaged', 'defective', 'delay', 'crash', 'issue', 'problem', 'lost', 'refund', 'replace', 'cancel', 'incorrect', 'unavailable', 'slow', 'disconnect', 'freeze'
]

def extract_products(text: str, product_list: List[str]) -> List[str]:
    found = []
    for product in product_list:
        if product.lower() in text.lower():
            found.append(product)
    return found

def extract_dates(text: str) -> List[str]:
    # Simple regex for dates (expand as needed)
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',
        r'\b\d{4}-\d{1,2}-\d{1,2}\b',
        r'\b\w+ \d{1,2}, \d{4}\b'
    ]
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)
    return dates

def extract_complaint_keywords(text: str) -> List[str]:
    found = []
    for kw in COMPLAINT_KEYWORDS:
        if kw in text.lower():
            found.append(kw)
    return found

def extract_entities(text: str, product_list: List[str]) -> Dict:
    return {
        'products': extract_products(text, product_list),
        'dates': extract_dates(text),
        'complaint_keywords': extract_complaint_keywords(text)
    }
