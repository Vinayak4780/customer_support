import os
import pandas as pd
from src.preprocessing import load_and_preprocess_data
from src.features import add_ticket_length, add_sentiment_score, fit_tfidf, tfidf_vectorizer
import joblib
from src.models import train_classifier, evaluate_classifier, save_model
from src.entity_extraction import extract_entities
from sklearn.model_selection import train_test_split
import numpy as np

# Paths
DATA_PATH = 'ai_dev_assignment_tickets_complex_1000.xls'
MODEL_ISSUE_PATH = 'src/issue_type_model.pkl'
MODEL_URGENCY_PATH = 'src/urgency_level_model.pkl'

# 1. Load and preprocess data
df = load_and_preprocess_data(DATA_PATH)

# 2. Feature engineering
df = add_ticket_length(df)
df = add_sentiment_score(df)
X_tfidf = fit_tfidf(df)
# Save the fitted TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, 'src/tfidf_vectorizer.pkl')
X = np.hstack([
    X_tfidf.toarray(),
    df[['ticket_length', 'sentiment']].values
])

# 3. Prepare targets
y_issue = df['issue_type']
y_urgency = df['urgency_level']

# 4. Train/test split
X_train, X_test, y_issue_train, y_issue_test, y_urgency_train, y_urgency_test = train_test_split(
    X, y_issue, y_urgency, test_size=0.2, random_state=42, stratify=y_issue
)

# 5. Train models
model_issue = train_classifier(X_train, y_issue_train, model_type='logreg')
model_urgency = train_classifier(X_train, y_urgency_train, model_type='logreg')

# 6. Evaluate
report_issue = evaluate_classifier(model_issue, X_test, y_issue_test)
report_urgency = evaluate_classifier(model_urgency, X_test, y_urgency_test)

# 7. Save models
save_model(model_issue, MODEL_ISSUE_PATH)
save_model(model_urgency, MODEL_URGENCY_PATH)

# 8. Example: Entity extraction for first ticket
product_list = df['product'].dropna().unique().tolist()
example_text = df.iloc[0]['ticket_text']
entities = extract_entities(example_text, product_list)

# 9. Print reports and example
print('Issue Type Classification Report:')
print(report_issue)
print('\nUrgency Level Classification Report:')
print(report_urgency)
print('\nExample Entity Extraction:')
print(entities)
