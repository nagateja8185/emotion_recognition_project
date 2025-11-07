import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# --- Absolute Paths (update if needed) ---
TRAIN_FILE = r"C:\emotion_detection_project\data\text\train.txt"
VAL_FILE   = r"C:\emotion_detection_project\data\text\val.txt"
TEST_FILE  = r"C:\emotion_detection_project\data\text\test.txt"
OUT_DIR    = r"C:\emotion_detection_project\models\text_emotion"

print("📄 TRAIN FILE:", TRAIN_FILE)
print("📄 VAL FILE  :", VAL_FILE)
print("📄 TEST FILE :", TEST_FILE)

def load_text_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Missing file: {path}")

    # Read using semicolon delimiter
    df = pd.read_csv(path, sep=";", header=None, names=["text", "emotion"], engine="python")

    # Clean up
    df = df.dropna(subset=["text", "emotion"])
    df["text"] = df["text"].astype(str).str.strip()
    df["emotion"] = df["emotion"].astype(str).str.strip()
    df = df[df["text"] != ""]
    return df

def main():
    train_df = load_text_file(TRAIN_FILE)
    val_df = load_text_file(VAL_FILE)
    test_df = load_text_file(TEST_FILE)

    X_train, y_train = train_df["text"], train_df["emotion"]
    X_val, y_val = val_df["text"], val_df["emotion"]
    X_test, y_test = test_df["text"], test_df["emotion"]

    print(f"✅ Loaded {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples.")
    print("Detected classes:", sorted(set(y_train)))

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=300))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    print("\n📊 Validation Report:\n", classification_report(y_val, preds))

    preds_test = pipeline.predict(X_test)
    print("\n🧪 Test Report:\n", classification_report(y_test, preds_test))

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(pipeline, os.path.join(OUT_DIR, "pipeline.joblib"))
    print("✅ Text emotion model saved to", OUT_DIR)

if __name__ == "__main__":
    main()
