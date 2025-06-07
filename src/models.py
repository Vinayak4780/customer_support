import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Model training and prediction for multi-class classification
def train_classifier(X, y, model_type='logreg'):
    if model_type == 'logreg':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError('Unsupported model_type')
    model.fit(X, y)
    return model

def evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, output_dict=True)

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

# Example usage:
# model = train_classifier(X_train, y_train, model_type='logreg')
# report = evaluate_classifier(model, X_test, y_test)
# save_model(model, 'model.pkl')
# model = load_model('model.pkl')
