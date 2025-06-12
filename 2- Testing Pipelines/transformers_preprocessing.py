import pandas as pd
import re
from tqdm import tqdm
import string

def norm_cost(text):
    norm_text = re.sub('[$€£]\d+', '#COST', text)
    return norm_text

def find_monetary_values(text):
    """Find monetary values and replace with MONETARY_VALUE_STAMP"""
    # Pattern for currency symbols followed by numbers or numbers followed by currency codes
    pattern = r'[$€£¥₹]\s*\d+(?:[.,]\d+)*|\d+(?:[.,]\d+)*\s*(?:USD|EUR|GBP|JPY|INR|dollars|euros|pounds)'
    return re.sub(pattern, " MONETARY_VALUE_STAMP ", text, flags=re.IGNORECASE)

def find_dates(text):
    """Find dates and replace with DATE_STAMP"""
    # Various date formats: MM/DD/YYYY, Month DD YYYY, DD-MM-YYYY, etc.
    pattern = r'\b(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4})\b'
    return re.sub(pattern, " DATE_STAMP ", text, flags=re.IGNORECASE)

def find_time_references(text):
    """Find time references and replace with TIME_STAMP"""
    # Various time formats: 12:30, 12:30 PM, 12h30, etc.
    pattern = r'\b(?:\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?|\d{1,2}[Hh]\d{2}(?:[Mm])?)\b'
    return re.sub(pattern, " TIME_STAMP ", text, flags=re.IGNORECASE)


def clean(text_list):

    updates = []

    for j in tqdm(text_list):

        text = j

        #LOWERCASE TEXT
        text = text.lower()

        #REMOVE NUMERICAL DATA and PUNCTUATION
        #text = re.sub("[^a-zA-Z]"," ", text )
        text = find_monetary_values(text)
        text = find_dates(text)
        text = norm_cost(text)
        text = find_time_references(text)
        
        text = re.sub(r'@\w+', '', text)

        # Remove non-letter characters and normalize whitespace
        
        text = re.sub(r'\b\d+(\.\d+)?%', 'PERCENTAGE', text)
        text = re.sub(r'\b\d+\b', '', text)

        text = re.sub(r'\s+', ' ', text).strip()

        text = re.sub(f"[{re.escape(string.punctuation)}]", '', text).strip()

        text = re.sub(r'\$[A-Za-z_]\w*', ' ',text).strip()

        text = re.sub('https?://\S+|www\.\S+', '',text).strip()

        text = re.sub(r' - ', ' ',text).strip()

        text = re.sub(r':', ' ',text).strip()

        text = re.sub(r'#\w+', '',text).strip()


        updates.append(text)

    return updates