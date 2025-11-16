
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_reviews.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    return pd.read_csv(path)

def train_and_save():
    df = load_data()
    X = df['review'].astype(str)
    y = df['sentiment'].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification report:\n", classification_report(y_test, preds))
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "sentiment_pipeline.joblib"))
    print("Saved model to", os.path.join(MODEL_DIR, "sentiment_pipeline.joblib"))

if __name__ == '__main__':
    train_and_save()
