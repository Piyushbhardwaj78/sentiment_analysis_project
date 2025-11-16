
import re

def clean_text(text):
    # basic cleaning: lowercase, remove urls, non-alphanum, extra spaces
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
