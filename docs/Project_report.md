# Project Report: Consumer Complaint Text Classification

## Dataset
CFPB Consumer Complaints [](https://catalog.data.gov/dataset/consumer-complaint-database). The full model was trained on the complete dataset provided (~100k rows), but due to file size limits, only a subset (100 rows) is uploaded in /data/complaints_sample.csv for demo purposes. Full results (e.g., 71% acc) are from the actual data; sample run outputs in models_sample/ are for reproducibility. 4 classes: 0-Credit reporting/repair/other, 1-Debt collection, 2-Consumer Loan, 3-Mortgage. Imbalance: Credit ~66%.

## 1. Exploratory Data Analysis (EDA) & Feature Engineering
- Loaded CSV, auto-detected columns (e.g., "Consumer complaint narrative").
- Plots: Label distribution (bar: class 0 dominant), text length boxplot (~80 words avg; mortgages longest).
- Features: Word count, TF-IDF (6k max, unigrams/bigrams), label mapping rules (e.g., 'credit' → 0).
- Top words: Class 0 ("credit", "report"), Class 1 ("collector", "debt").

## 2. Text Pre-Processing
- Pipeline: `simple_clean` (lowercase, remove URLs/emails/punct), `tokenize_and_lemmatize` (NLTK word_tokenize, stopwords removal, WordNet lemmatization), `preprocess_text` wrapper.
- Example: "Collectors calling! http://ex.com" → "collector calling".
- Handles NaN/empty; integrated in train/predict.

## 3. Selection of Multi-Classification Model
- Candidates: LogisticRegression (balanced), LinearSVC, RandomForest, MultinomialNB.
- Trained pipelines (80/20 stratified split) on TF-IDF + models.
- Best: LogisticRegression (robust to imbalance, fast on text).

## 4. Comparison of Model Performance
| Model              | Accuracy | Macro F1 |
|--------------------|----------|----------|
| LogisticRegression | 0.710   | 0.436   |
| LinearSVC          | 0.717   | 0.434   |
| RandomForest       | 0.715   | 0.400   |
| MultinomialNB      | 0.710   | 0.382   |
- Metrics saved to `models_sample/metrics.csv`. Balanced weights improved rare-class F1.

## 5. Model Evaluation
- Report: Precision/recall/F1 per class (credit: 0.82 F1/97% recall; loan: 0.17 F1/11% recall—imbalance issue). Full eval on actual dataset (~11k samples); sample demo in models_sample/.
- Confusion matrix PNG: High diag for 0/3, bleed debt→credit.
- Batch tests: 80% acc on 5 multi-class examples (4/5 correct; loan mispred as debt).

## 6. Prediction
- `predict.py`: Single (`--text`) or batch (`--input_csv`), auto-detects column, outputs labels (e.g., "Debt text" → 1).
- Example: `python src/predict.py --model_dir models_sample --text "Debt collectors harassing" → Predicted: 1 -> Debt collection`.

## Challenges & Improvements
- Imbalance: Balanced

*Report by Sarun S, Oct 18, 2025.*