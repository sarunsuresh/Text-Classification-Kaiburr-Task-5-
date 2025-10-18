import argparse
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Local minimal preprocessing helpers (avoid heavy NLTK downloads by using sklearn stopwords)
import re
from collections import Counter
from typing import Optional, List
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


STOPWORDS = set(ENGLISH_STOP_WORDS)


def simple_clean(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str, remove_stopwords: bool = True) -> str:
    """Simple whitespace tokenizer + optional stopword removal. No lemmatization to avoid heavy deps."""
    if not text:
        return ""
    tokens = text.split()
    out = []
    for t in tokens:
        if remove_stopwords and t in STOPWORDS:
            continue
        out.append(t)
    return " ".join(out)


def preprocess_text(text: str) -> str:
    return tokenize_and_lemmatize(simple_clean(text))


def find_text_column(df: pd.DataFrame) -> str:
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


def map_product_to_label(product: str) -> Optional[int]:
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


def top_n_words(texts: List[str], n: int = 25):
    c = Counter()
    for t in texts:
        if not isinstance(t, str):
            continue
        for w in t.split():
            c[w] += 1
    return c.most_common(n)


def load_data(path: str, label_column: str = None, sample_size: int = 50000):
    # Read header first to allow case-insensitive matching without loading full file
    try:
        header_df = pd.read_csv(path, nrows=0)
        available_cols = list(header_df.columns)
    except Exception:
        # Fallback to reading a small sample and infer columns
        sample = pd.read_csv(path, nrows=5)
        available_cols = list(sample.columns)

    # helper to find a column name in the available columns case-insensitively
    def find_col_case_insensitive(candidates):
        lower_map = {c.lower(): c for c in available_cols}
        for cand in candidates:
            if not cand:
                continue
            # direct match
            if cand in available_cols:
                return cand
            # case-insensitive match
            low = cand.lower()
            if low in lower_map:
                return lower_map[low]
        return None

    # preferred names to look for
    product_candidates = [label_column, 'Product', 'product', 'PRODUCT'] if label_column else ['Product', 'product', 'PRODUCT']
    text_candidates = [
        'Consumer complaint narrative',
        'consumer_complaint_narrative',
        'Complaint narrative',
        'consumer complaint narrative',
        'complaint_what_happened',
        'complaint_text',
        'complaint',
        'issue',
        'description',
        'text',
    ]

    product_col = find_col_case_insensitive(product_candidates)
    text_col = find_col_case_insensitive(text_candidates)

    # Build usecols: include only the columns we need (product + text) to save memory
    usecols = []
    if product_col:
        usecols.append(product_col)
    if text_col and text_col not in usecols:
        usecols.append(text_col)

    # Faster read with dtype hints (object for text/product, skip others)
    dtype_dict = {}
    if product_col:
        dtype_dict[product_col] = 'category'  # Memory-efficient for labels
    if text_col:
        dtype_dict[text_col] = 'object'

    # Read in chunks if huge, but sample during load for speed
    chunksize = 10000  # Adjust based on RAM
    chunks = pd.read_csv(path, usecols=usecols if usecols else None,
                         dtype=dtype_dict, low_memory=False, chunksize=chunksize,
                         encoding='utf-8')  # Explicit encoding if issues

    df_list = []
    total_rows = 0
    for chunk in chunks:
        total_rows += len(chunk)
        print(f"Processed {total_rows:,} rows so far...")  # Progress indicator

        # Apply mapping/cleaning per chunk to save memory
        if product_col and product_col in chunk.columns:
            chunk['temp_label'] = chunk[product_col].map(map_product_to_label)
        elif label_column and label_column in chunk.columns:
            chunk['temp_label'] = chunk[label_column].map(map_product_to_label)
        else:
            lower_cols = {c.lower(): c for c in chunk.columns}
            if 'product' in lower_cols:
                chunk['temp_label'] = chunk[lower_cols['product']].map(map_product_to_label)
            else:
                raise ValueError(f"Label column not provided and no product-like column found in CSV. Available columns: {', '.join(chunk.columns[:20])} ...")

        if text_col and text_col in chunk.columns:
            chosen_text_col = text_col
        else:
            chosen_text_col = find_text_column(chunk)

        chunk['text'] = chunk[chosen_text_col].astype(str).map(preprocess_text)
        chunk['label'] = chunk['temp_label']
        chunk = chunk.dropna(subset=['label'])
        if len(chunk) == 0:
            continue
        chunk['label'] = chunk['label'].astype(int)
        df_list.append(chunk)

        # Early stop if sampled enough
        if sample_size > 0 and total_rows >= sample_size * 2:  # Buffer for drops
            break

    if not df_list:
        raise ValueError("No data loaded—check file/columns")

    df = pd.concat(df_list, ignore_index=True)

    # Final sampling if requested (approximate stratified to preserve balance)
    if sample_size > 0 and len(df) > sample_size:
        print(f"Sampling down to {sample_size} rows...")
        # Stratified sample per class
        samples_per_class = sample_size // df['label'].nunique()
        df_sampled = df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(min(len(x), samples_per_class), random_state=42)
        ).reset_index(drop=True)
        # If still over, random sample the rest
        if len(df_sampled) > sample_size:
            df = df_sampled.sample(n=sample_size, random_state=42).reset_index(drop=True)
        else:
            df = df_sampled

    print(f"Final dataset: {len(df):,} rows across {df['label'].nunique()} classes")
    return df


def eda(df: pd.DataFrame, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sns.set()
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=df["label"])
    ax.set_title("Label distribution")
    plt.savefig(os.path.join(output_dir, "label_distribution.png"))
    plt.close()

    df["text_len"] = df["text"].map(lambda x: len(str(x).split()))
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="label", y="text_len", data=df)
    plt.title("Text length by label")
    plt.savefig(os.path.join(output_dir, "text_length_by_label.png"))
    plt.close()


def train(df: pd.DataFrame, output_dir: str, max_features: int = 10000):
    X = df["text"]
    y = df["label"]
    # Attempt a stratified split; if dataset is too small for stratification, fall back to a regular split.
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    except ValueError as e:
        print("Stratified split failed (likely very small or imbalanced dataset):", e)
        print("Falling back to non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Define candidate classifiers to compare
    candidates = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "LinearSVC": LinearSVC(max_iter=2000, class_weight='balanced'),  # Key addition
    "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight='balanced'),
    "MultinomialNB": MultinomialNB(),  # NB doesn't support weights; skip or use fit_prior=False
}

    results = []
    best_f1 = -1.0
    best_model = None
    best_name = None

    # common pipeline prefix
    for name, clf in candidates.items():
        print(f"Training candidate: {name}")
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
            ("clf", clf),
        ])

        # For very small datasets skip CV and just fit
        try:
            pipe.fit(X_train, y_train)
        except Exception as e:
            print(f"Training {name} failed: {e}")
            continue

        y_pred = pipe.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        acc = np.mean(y_pred == y_test)
        print(f"{name} - acc: {acc:.3f}, f1_macro: {f1:.3f}")
        results.append({"model": name, "accuracy": acc, "f1_macro": f1})

        if f1 > best_f1:
            best_f1 = f1
            best_model = pipe
            best_name = name

    metrics_df = pd.DataFrame(results)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    print("Saved metrics to", os.path.join(output_dir, "metrics.csv"))

    if best_model is None:
        raise RuntimeError("No model was successfully trained")

    print(f"Best model: {best_name} (f1_macro={best_f1:.3f})")
    # Detailed report for best model
    y_pred_best = best_model.predict(X_test)
    report = classification_report(y_test, y_pred_best)
    print(report)

    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion matrix - {best_name}")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # save the best model pipeline
    joblib.dump(best_model, os.path.join(output_dir, "model.joblib"))
    print("Best model saved to", os.path.join(output_dir, "model.joblib"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CSV file")
    parser.add_argument("--output_dir", default="models", help="Directory to save model and outputs")
    parser.add_argument("--sample_size", type=int, default=50000, help="Max rows to sample (0 for full dataset)")
    parser.add_argument("--max_features", type=int, default=10000, help="Max TF-IDF features (lower = faster)")
    args = parser.parse_args()

    df = load_data(args.input, sample_size=args.sample_size)
    eda(df, args.output_dir)
    train(df, args.output_dir, max_features=args.max_features)