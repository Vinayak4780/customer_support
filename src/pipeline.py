import os
import numpy as np
from preprocessing import preprocess_text
from models import load_model
from entity_extraction import extract_entities
import pandas as pd
import joblib

# Load the fitted TF-IDF vectorizer
try:
    tfidf_vectorizer = joblib.load('src/tfidf_vectorizer.pkl')
except Exception as e:
    raise RuntimeError('TF-IDF vectorizer could not be loaded. Make sure to run main.py first.') from e

# Load product list from the original data (for entity extraction)
def get_product_list(data_path):
    df = pd.read_excel(data_path)
    return df['product'].dropna().unique().tolist()

# Integration function
def predict_ticket(ticket_text, product_list, model_issue_path, model_urgency_path):
    # Preprocess
    text_clean = preprocess_text(ticket_text)
    # Feature engineering
    features = {
        'ticket_text_clean': text_clean,
        'ticket_length': len(text_clean.split()),
        'sentiment': 0.0
    }
    # Sentiment
    from textblob import TextBlob
    features['sentiment'] = TextBlob(text_clean).sentiment.polarity
    # TF-IDF
    X_tfidf = tfidf_vectorizer.transform([text_clean])
    # Combine features (TF-IDF + others)
    X = np.hstack([
        X_tfidf.toarray(),
        np.array([[features['ticket_length'], features['sentiment']]])
    ])
    # Load models
    model_issue = load_model(model_issue_path)
    model_urgency = load_model(model_urgency_path)
    # Predict
    pred_issue = model_issue.predict(X)[0]
    pred_urgency = model_urgency.predict(X)[0]
    # Entity extraction
    entities = extract_entities(ticket_text, product_list)
    return {
        'predicted_issue_type': pred_issue,
        'predicted_urgency_level': pred_urgency,
        'entities': entities
    }
