# -*- coding: utf-8 -*-
"""
Read docs.md for an in-depth explanation of the practice.

P7: Data Clustering

Create a model using K means and test it.

Group reviews using numerical features (sentiment & review length), 
to identify common feedback subgroups.

Martín Alexis Martínez Andrade
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# !pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from itertools import combinations

# In the root dir of the repo, the Dataset/ directory contains the
# data used for this practice.
# You can also download the .csv from dropbox
# !wget -O modified_clean_data.csv https://www.dropbox.com/scl/fi/8pnm0884bksvfcyxep4ec/modified_clean_data.csv?rlkey=n8qd7e299e1bydtfiwemsdeig&st=89yik26c&dl=1
df = pd.read_csv("../Dataset/modified_clean_data.csv")

# Keep only reviews with N or more votes
n_votes_threshold = 10
df = df[df['HelpfulnessDenominator'] >= n_votes_threshold]
print(f"Reviews kept (reviews with {n_votes_threshold} or more votes): {df.shape[0]}")

df = df.sample(3000, random_state=2)
print(f"Sample size: {df.shape[0]}")

# Sentiment analysis
analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    vs = analyzer.polarity_scores(str(text))
    return vs['compound']  # [-1 ... 1], best as continuous variable

# Compute sentiment and review length
df['Sentiment'] = df['Text'].apply(get_vader_sentiment)
df['ReviewLength'] = df['Text'].str.len()

# Use Sentiment and ReviewLength as features for clustering
X = df[['Sentiment', 'ReviewLength']].values

def within_cluster_variation(X, labels, cluster_number):
    """
    Compute the average pairwise squared Euclidean distance for all points in
    one cluster (as in ISLR 12.17).
    """
    indices = np.where(labels == cluster_number)[0]
    if len(indices) < 2:
        return 0  # variation is zero if only one point
    X_cluster = X[indices]
    pairwise_distances = [
        np.sum((X_cluster[i] - X_cluster[j]) ** 2)
        for i, j in combinations(range(X_cluster.shape[0]), 2)
    ]
    avg_within_variance = np.mean(pairwise_distances)
    return avg_within_variance

# Run iterations
results = []
k = 7

for it in range(5):
    kmeans = KMeans(n_clusters=k, random_state=it).fit(X)
    silhouette = silhouette_score(X, kmeans.labels_)
    total_W = 0
    cluster_means = []
    for cluster_num in range(k):
        w = within_cluster_variation(X, kmeans.labels_, cluster_num)
        total_W += w
        indices = (kmeans.labels_ == cluster_num)
        mean_sentiment = df.loc[indices, 'Sentiment'].mean()
        mean_length = df.loc[indices, 'ReviewLength'].mean()
        mean_score = df.loc[indices, 'Score'].mean()
        cluster_means.append((mean_sentiment, mean_length, mean_score))
    results.append({
        "random_state": it,
        "silhouette": silhouette,
        "total_W": total_W,
        "labels": kmeans.labels_.copy(),
        "cluster_means": cluster_means,
        "kmeans": kmeans,
    })
    print(f"\nIteration {it:02d} | Silhouette: {silhouette:.4f} | Total W: {total_W:.4f}")

# Select best result
best_idx = np.argmax([r["silhouette"] for r in results])
best = results[best_idx]
print("\n" + "="*60)
print(f"BEST RESULT: Iteration {best['random_state']} | Silhouette: {best['silhouette']:.4f}\n")
df['cluster'] = best["labels"]

# Print best cluster details
for i in range(k):
    cluster_reviews = df[df['cluster'] == i]
    m_sent, m_len, m_score = best["cluster_means"][i]
    print(f"\nCluster {i}:")
    print(cluster_reviews['Text'].sample(2).values)
    print(f"  Mean sentiment: {m_sent:.4f}, mean length: {m_len:.4f}, mean Score: {m_score:.4f}")
    print(f"  Cluster size: {cluster_reviews.shape[0]}")

# Print within-cluster variation details for the best result
for cluster_num in range(k):
    w = within_cluster_variation(X, best['labels'], cluster_num)
    print(f"Within-cluster variation for cluster {cluster_num}: {w:.4f}")
print(f"\nTotal within-cluster variation (ISLR Eqn 12.17): {best['total_W']:.4f}")