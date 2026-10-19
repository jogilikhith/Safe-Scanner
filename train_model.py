import os
import io
import zipfile
import urllib.request
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Set base directory to wherever this script is saved in VS Code
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "spam.csv")
MODEL_PATH = os.path.join(BASE_DIR, "phishing_model.pkl")

UCI_DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"


def load_or_download_dataset() -> pd.DataFrame:
    """Loads local spam.csv if present, or auto-downloads the 5,572-row UCI dataset."""
    if os.path.exists(CSV_PATH):
        print(f"[1/5] Loading local dataset from: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH, encoding="latin-1")
        # Handle Kaggle's v1/v2 column naming
        if "v1" in df.columns and "v2" in df.columns:
            df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
        return df

    print("[1/5] 'spam.csv' not found locally. Auto-downloading 5,572-row dataset from UCI...")
    response = urllib.request.urlopen(UCI_DATASET_URL)
    with zipfile.ZipFile(io.BytesIO(response.read())) as z:
        with z.open("SMSSpamCollection") as raw_file:
            df = pd.read_csv(raw_file, sep="\t", header=None, names=["label", "text"], encoding="utf-8")

    # Save locally as spam.csv so future VS Code runs are instant
    df.to_csv(CSV_PATH, index=False, encoding="latin-1")
    print(f"      Saved dataset locally to: {CSV_PATH}")
    return df


def get_modern_phishing_samples() -> pd.DataFrame:
    """Augments classic SMS spam with modern UPI, KYC, and link-based phishing samples."""
    modern_data = {
        "label": ["spam"] * 10 + ["ham"] * 5,
        "text": [
            "URGENT: Your bank account is suspended. Click http://bit.ly/verify-kyc to update PAN and KYC immediately.",
            "Congratulations! You won a Rs 50,000 cash prize on UPI. Claim now at http://free-reward-claim.com",
            "Dear customer, your electricity bill is unpaid. Power will be disconnected tonight at 9 PM. Call 9876543210.",
            "Alert: Unauthorized login attempt detected. Verify your netbanking password at http://secure-login-update.net",
            "Your parcel delivery failed due to incomplete address. Update address here: http://track-shipment-info.org",
            "Income Tax Refund of Rs 14,500 is approved. Click link to verify bank account number and IFSC urgently.",
            "Part-time job offer! Earn Rs 5000 daily from home by liking YouTube videos. Join Telegram http://t.me/easycash",
            "FINAL NOTICE: Your SIM card will be deactivated within 24 hours. Submit Aadhaar OTP immediately.",
            "Your credit card reward points worth Rs 4,800 expire today. Redeem into cash at http://cc-rewards-portal.in",
            "Netflix subscription payment declined. Update your debit card CVV and PIN at http://billing-update-account.com",
            "Rs 250.00 debited from your account via UPI to Swiggy. Ref No 6291048291.",
            "Your Amazon order #402-1122334 has been shipped and is scheduled for delivery by Tuesday.",
            "Hi team, please find attached the Python scripts and Jupyter Notebook report for tomorrow's review.",
            "Class schedule update: Tomorrow's Java programming lecture is moved to Room 302 at 11 AM.",
            "Your monthly broadband invoice for October has been generated. View it inside the official app."
        ]
    }
    return pd.DataFrame(modern_data)


def clean_dataset_with_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Applies Pandas data cleaning: null removal, deduplication, and length filtering."""
    print("[2/5] Running Pandas Data Cleaning & Outlier Removal...")
    initial_rows = len(df)

    # Combine classic dataset with modern phishing examples
    df = pd.concat([df, get_modern_phishing_samples()], ignore_index=True)

    # 1. Drop missing/null rows
    df.dropna(subset=["label", "text"], inplace=True)

    # 2. Clean text and remove exact duplicate messages
    df["text"] = df["text"].astype(str).str.strip()
    df.drop_duplicates(subset=["text"], keep="first", inplace=True)

    # 3. Map labels ('ham' -> 0, 'spam' -> 1)
    df["label_Clean"] = df["label"].astype(str).str.lower().str.strip().map({"ham": 0, "spam": 1, "0": 0, "1": 1})
    df.dropna(subset=["label_Clean"], inplace=True)
    df["label_num"] = df["label_Clean"].astype(int)

    # 4. Outlier Filtering based on message length
    df["char_length"] = df["text"].str.len()
    upper_cap = df["char_length"].quantile(0.995)
    df = df[(df["char_length"] >= 4) & (df["char_length"] <= upper_cap)].copy()

    # 5. Create normalized lowercase feature column
    df["clean_text"] = df["text"].str.lower()

    print(f"      Raw rows: {initial_rows} | Cleaned & Deduplicated rows: {len(df)}")
    print(f"      Class breakdown -> Safe (0): {(df['label_num'] == 0).sum()} | Phishing/Spam (1): {(df['label_num'] == 1).sum()}")
    return df


def train_and_evaluate(df: pd.DataFrame):
    """Splits dataset, trains Scikit-learn pipeline, prints evaluation metrics, and saves model."""
    print("[3/5] Splitting data (80% Train / 20% Test, Stratified)...")
    X = df["clean_text"]
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("[4/5] Training TF-IDF + Random Forest Pipeline...")
    model_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3500, ngram_range=(1, 2), stop_words="english")),
        ("classifier", RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1))
    ])

    model_pipeline.fit(X_train, y_train)

    # Evaluate on unseen 20% test set
    y_pred = model_pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred) * 100
    rec = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 55)
    print("      MODEL EVALUATION METRICS (20% UNSEEN TEST SET)")
    print("=" * 55)
    print(f"  Accuracy  : {acc:.2f}%  (Overall correct predictions)")
    print(f"  Precision : {prec:.2f}%  (When it flags Spam, how often it's right)")
    print(f"  Recall    : {rec:.2f}%  (Percentage of real Spam caught)")
    print(f"  F1-Score  : {f1:.2f}%  (Balance between Precision and Recall)")
    print("-" * 55)
    print("  Confusion Matrix:")
    print(f"    True Safe (TN)     : {cm[0][0]:4d} | False Alarms (FP) : {cm[0][1]:4d}")
    print(f"    Missed Scams (FN)  : {cm[1][0]:4d} | Caught Scams (TP) : {cm[1][1]:4d}")
    print("=" * 55 + "\n")

    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Safe (0)", "Phishing/Spam (1)"]))

    # Save trained pipeline for FastAPI backend
    print(f"[5/5] Saving production model to: {MODEL_PATH}")
    joblib.dump(model_pipeline, MODEL_PATH)
    print("Done! Model successfully updated and saved.")


if __name__ == "__main__":
    raw_df = load_or_download_dataset()
    clean_df = clean_dataset_with_pandas(raw_df)
    train_and_evaluate(clean_df)