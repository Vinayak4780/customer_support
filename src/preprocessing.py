import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Text preprocessing functions
def normalize_text(text):
    if pd.isnull(text):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def tokenize(text):
    return text.split()

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_text(text):
    text = normalize_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return ' '.join(tokens)

def load_and_preprocess_data(filepath):
    df = pd.read_excel(filepath)
    # Handle missing data
    df = df.dropna(subset=['ticket_text', 'issue_type', 'urgency_level'])
    df['ticket_text'] = df['ticket_text'].fillna('')
    df['ticket_text_clean'] = df['ticket_text'].apply(preprocess_text)
    return df
