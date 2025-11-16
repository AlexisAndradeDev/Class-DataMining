# -*- coding: utf-8 -*-
"""
Read docs.md for an explanation of the practice.

P9: Text Analysis
Word clouds for sentiment analysis.
Generate and compare word clouds for positive and negative reviews.

Martín Alexis Martínez Andrade
"""
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

df = pd.read_csv("../Dataset/modified_clean_data.csv")

# Positive: score 4-5
# Negative: score 1-2
positive_reviews = df[df['Score'] >= 4]['Text']
negative_reviews = df[df['Score'] <= 2]['Text']

extra_stopwords = set([
    'product', 'amazon', 'would', 'one', 'get', 'can', 'like', 'just',
    'really', 'also', 'br' # br because of <br> html tag
])
stopwords = STOPWORDS.union(extra_stopwords)

def word_cloud(reviews: pd.Series, title: str):
    text = " ".join(str(r) for r in reviews)
    wc_pos = WordCloud(width=900, height=400, stopwords=stopwords, background_color='white', collocations=False).generate(text)
    plt.figure(figsize=(14, 7))
    plt.imshow(wc_pos, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.show()

word_cloud(positive_reviews, "Word Cloud - Positive Reviews (4-5 stars)")
word_cloud(negative_reviews, "Word Cloud - Negative Reviews (1-2 stars)")
