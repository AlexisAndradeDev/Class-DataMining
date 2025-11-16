# -*- coding: utf-8 -*-
"""
Read docs.md for an explanation of the practice.

P5: Linear models and correlation

Martín Alexis Martínez Andrade
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

df = pd.read_csv("../Dataset/modified_clean_data.csv")

def perform_linear_regression(df: pd.DataFrame, x_col: str, y_col: str) -> None:
    X = df[[x_col]]
    y = df[y_col]
    # discard nulls
    mask = X.notnull().all(axis=1) & y.notnull()

    X, y = X[mask], y[mask]

    lm = LinearRegression().fit(X, y)
    print(f"{x_col} vs. {y_col}")
    print(f'R^2: {lm.score(X, y):.4f}, Coefficient: {lm.coef_[0]:.5f}')
    plt.scatter(X, y, alpha=0.2)
    plt.plot(X, lm.predict(X), color='red')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

print(df.dtypes)

# correlation matrix in numerical columns (HelpfulnessNumerator, Denominator, Score, Timestamp, HelpfulnessScore and ReviewLength)
corr = df[['HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Timestamp', 'HelpfulnessScore', 'ReviewLength']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# Parse to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Group by product, get first and last review date, and count reviews
agg = df.groupby('ProductId').agg(
    first_review=('Date', 'min'),
    last_review=('Date', 'max'),
    n_reviews=('ProductId', 'count')
)

# Compute elapsed days from first to last review
agg['days_elapsed'] = (agg['last_review'] - agg['first_review']).dt.days

X = agg[['days_elapsed']]
y = agg['n_reviews']

lm = LinearRegression().fit(X, y)
print(f'Amount of reviews of a product vs Amount of days passed since its first review')
print(f'R^2: {lm.score(X, y):.4f}')
print(f'Coefficient: {lm.coef_[0]:.4f} reviews/day')

# Plot
plt.scatter(X, y, alpha=0.2, label='Products')
plt.plot(X, lm.predict(X), color='red', linewidth=2, label='Linear fit')
plt.xlabel('Days Since First Review')
plt.ylabel('Number of Reviews')
plt.legend()
plt.show()

perform_linear_regression(df, 'HelpfulnessDenominator', 'HelpfulnessScore')

perform_linear_regression(df, 'ReviewLength', 'HelpfulnessScore')

# keep only reviews with N or more votes
n_votes_threshold = 10
df_filtered = df[df['HelpfulnessDenominator'] >= n_votes_threshold]
print(f"Reviews kept (reviews with {n_votes_threshold} or more votes): {df_filtered.shape[0]}")

df_sampled = df_filtered.copy()

df_sampled['Sentiment'] = df_sampled['Text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# use linear model func
perform_linear_regression(df_sampled, 'Sentiment', 'HelpfulnessScore')

analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    vs = analyzer.polarity_scores(str(text))
    return vs['compound']  # [-1 ... 1], best as continuous variable

df_sampled = df_filtered.copy().dropna(subset=['Text'])
df_sampled['Sentiment'] = df_sampled['Text'].apply(get_vader_sentiment)

perform_linear_regression(df_sampled, 'Sentiment', 'HelpfulnessScore')

def run_bert_sentiment_analysis():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    from tqdm import tqdm

    # Sample (for execution speed)
    # keep random_state=73 to have the same results as the sample available for
    # download
    df_sampled = df_filtered.sample(n=2000, random_state=73)
    print(f"Reviews sampled: {df_sampled.shape[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    def bert_sentiment_score(text):
        # handle empty text
        if not isinstance(text, str) or not text.strip():
            return np.nan
        result = sentiment_pipeline(text[:512])[0] # BERT limit is 512 tokens
        # result['label'] will be "1 star", ..., "5 stars"
        score = int(result['label'][0]) # take the first character
        return score

    # 1 to 5
    df_sampled['Sentiment'] = [
        bert_sentiment_score(text)
        for text in tqdm(df_sampled['Text'], desc="BERT sentiment")
    ]
    # 0.0 to 1.0 (0.0, 0.25, 0.5, ...)
    df_sampled['SentimentNormalized'] = (df_sampled['Sentiment'] - 1) / 4

    perform_linear_regression(df_sampled, 'SentimentNormalized', 'HelpfulnessScore')

while True:
    # WARNING: (Input 'no' if you don't have a cuda-capable GPU unless you have 60 minutes of patience)
    execute_bert = input("Execute BERT for sentiment analysis (yes/no)? WARNING: Might take up to 30 minutes on a CPU (a few seconds on a T4 GPU). It's recommended to enter 'no' if you don't have a cuda-capable GPU, and you'll get the exact same result showed in the notebook (if you keep the random_state=73).\n*: ")
    if execute_bert.lower() not in ["yes", "no"]:
        print("Enter 'yes' or 'no'.")
        continue
    break

execute_bert = True if execute_bert.lower() == "yes" else False

if execute_bert:
    run_bert_sentiment_analysis()
    
# Score vs Helpfulness Score
perform_linear_regression(df, 'Score', 'HelpfulnessScore')