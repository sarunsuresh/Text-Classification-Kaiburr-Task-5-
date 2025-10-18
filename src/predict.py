import argparse
import joblib
import os
from pathlib import Path
import pandas as pd

from utils import simple_clean, tokenize_and_lemmatize, preprocess_text, find_text_column, LABEL_MAP# Add LABEL_MAP to utils.py if missing (see below)

# If LABEL_MAP not in utils, add this to utils.py:
# LABEL_MAP = {0: "Credit reporting, repair, or other", 1: "Debt collection", 2: "Consumer Loan", 3: "Mortgage"}

def load_model(model_dir: str):
    model_path = os.path.join(model_dir, "model.joblib")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)

def predict_text(model, text: str):
    processed = preprocess_text(text)  # Uses your full pipeline (clean + tokenize + lemmatize)
    pred = model.predict([processed])[0]
    return int(pred), LABEL_MAP.get(pred, "Unknown")

def main(args):
    model = load_model(args.model_dir)

    if args.text:
        label_idx, label_name = predict_text(model, args.text)
        print(f"Predicted: {label_idx} -> {label_name}")
    elif args.input_csv:
        df = pd.read_csv(args.input_csv)
        text_col = find_text_column(df)  # Uses your utils function—auto-detects!
        print(f"Using text column: {text_col}")
        
        def apply_pred(t):
            if pd.isna(t):
                return -1  # Or "Unknown"
            idx, _ = predict_text(model, str(t))
            return idx
        
        df['prediction_label'] = df[text_col].apply(apply_pred)
        df['prediction_name'] = df['prediction_label'].map(LABEL_MAP).fillna("Unknown")
        
        out_path = Path(args.model_dir) / 'batch_predictions.csv'
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} predictions to {out_path}")
        print(df[['prediction_label', 'prediction_name']].value_counts().sort_index())  # Quick summary
    else:
        print("Provide --text or --input_csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict complaint categories using trained model")
    parser.add_argument("--model_dir", default="../models_radeon", help="Directory containing model.joblib")
    parser.add_argument("--text", help="Single text to predict")
    parser.add_argument("--input_csv", help="CSV file with texts to predict (batch)")
    args = parser.parse_args()
    main(args)