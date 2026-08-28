"""
SmartSpend AI - Text Preprocessing Module
Provides text cleaning and Thai-English tokenization for expense categorization.
"""

import re
from typing import List
from pythainlp.tokenize import word_tokenize

def clean_text(text: str) -> str:
    """
    Cleans raw text by:
    - Converting to string and trimming whitespace
    - Lowercasing English letters
    - Removing special symbols while preserving Thai characters, English words, numbers, and spaces
    """
    if text is None:
        return ""
    
    text = str(text).strip()
    text = text.lower()
    
    # Replace common separators/symbols with space
    text = re.sub(r"[/\\_\-+:@#$&*!?.,;()\[\]{}'\"]", " ", text)
    
    # Remove any remaining unwanted non-alphanumeric (except Thai range \u0E00-\u0E7F)
    text = re.sub(r"[^\w\s\u0E00-\u0E7F]", " ", text)
    
    # Normalize multiple whitespace characters to single space
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def tokenize_text(text: str, engine: str = "newmm") -> List[str]:
    """
    Tokenizes mixed Thai and English text into word tokens using PyThaiNLP.
    Filters out empty tokens and standalone whitespace.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    
    tokens = word_tokenize(cleaned, engine=engine, keep_whitespace=False)
    # Remove any stray spaces or empty strings
    filtered_tokens = [t.strip() for t in tokens if t.strip()]
    return filtered_tokens

def prepare_text_feature(merchant: str, memo: str) -> str:
    """
    Combines merchant and memo into a single tokenized space-separated string.
    Suitable for input into TfidfVectorizer.
    """
    raw_combined = f"{merchant or ''} {memo or ''}".strip()
    tokens = tokenize_text(raw_combined)
    return " ".join(tokens)
