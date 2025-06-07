import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

# TF-IDF Vectorizer (fit on training data, transform on all)
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

def add_ticket_length(df, text_col='ticket_text_clean'):
    df['ticket_length'] = df[text_col].apply(lambda x: len(x.split()))
    return df

def add_sentiment_score(df, text_col='ticket_text_clean'):
    df['sentiment'] = df[text_col].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df

def fit_tfidf(df, text_col='ticket_text_clean'):
    return tfidf_vectorizer.fit_transform(df[text_col])

def transform_tfidf(df, text_col='ticket_text_clean'):
    return tfidf_vectorizer.transform(df[text_col])

# Example usage:
# df = add_ticket_length(df)
# df = add_sentiment_score(df)
# X_tfidf = fit_tfidf(df)
