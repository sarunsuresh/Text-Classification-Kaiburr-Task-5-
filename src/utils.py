import re
import string
from collections import Counter
from typing import Optional, List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd


# --- Ensure NLTK resources are available ---
# --- Ensure NLTK resources are available ---
def _ensure_nltk():
    try:
        stopwords.words("english")
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/wordnet")
        nltk.data.find("tokenizers/punkt_tab/english")  # Add this check
    except LookupError:
        nltk.download("stopwords")
        nltk.download("punkt")
        nltk.download("punkt_tab")  # Add this download
        nltk.download("wordnet")
        nltk.download("omw-1.4")


_ensure_nltk()

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
LABEL_MAP = {
    0: "Credit reporting, repair, or other",
    1: "Debt collection",
    2: "Consumer Loan",
    3: "Mortgage",
}


# --- Basic text cleaning ---
def simple_clean(text: Optional[str]) -> str:
    """Lowercase, remove URLs, emails, punctuation, digits; collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Tokenization and Lemmatization ---
def tokenize_and_lemmatize(text: str, remove_stopwords: bool = True) -> str:
    """Tokenize, remove stopwords (optional), and lemmatize tokens."""
    if not text:
        return ""
    tokens = word_tokenize(text)
    out = []
    for t in tokens:
        if remove_stopwords and t in STOPWORDS:
            continue
        out.append(LEMMATIZER.lemmatize(t))
    return " ".join(out)


# --- Combined preprocessing pipeline ---
def preprocess_text(text: str) -> str:
    """Clean, tokenize, and lemmatize text."""
    return tokenize_and_lemmatize(simple_clean(text))


# --- Detect text column in DataFrame ---
def find_text_column(df: pd.DataFrame) -> str:
    """Try to detect a complaint text column in a dataset."""
    candidates = [
        'consumer_complaint_narrative',
        'complaint_what_happened',
        'complaint_text',
        'complaint',
        'issue',
        'description',
        'text',
        'consumer_complaint',
    ]
    for c in candidates:
        if c in df.columns:
            return c

    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if not obj_cols:
        raise ValueError('No text-like columns found in DataFrame')

    lengths = {c: df[c].astype(str).map(len).mean() for c in obj_cols}
    return max(lengths, key=lengths.get)


# --- Map product to label ---
def map_product_to_label(product: str) -> Optional[int]:
    """Convert product name to numeric label."""
    if not isinstance(product, str):
        return None
    p = product.lower()
    if 'credit' in p or 'reporting' in p or 'repair' in p:
        return 0
    if 'debt' in p or 'collection' in p:
        return 1
    if 'consumer' in p and 'loan' in p:
        return 2
    if 'mortgage' in p or 'home' in p:
        return 3
    if 'loan' in p and ('personal' in p or 'installment' in p):
        return 2
    return None


# --- Find top N words ---
def top_n_words(texts: List[str], n: int = 25):
    """Return top N most frequent words in the given list of texts."""
    c = Counter()
    for t in texts:
        if not isinstance(t, str):
            continue
        for w in t.split():
            c[w] += 1
    return c.most_common(n)
