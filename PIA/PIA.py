# -*- coding: utf-8 -*-
"""
Final Project: Thematic Segmentation of Amazon Fine Foods Products using Text Embeddings and Clustering

Identifies subgroups of products/reviews from textual content,
and validates the robustness and semantic coherence of the clusters.

Martín Alexis Martínez Andrade
"""

print("Loading modules...")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
print("Modules loaded!")

def preprocess_row(row):
    """Prepare the text for embeddings. Use just the summary (kind of the
    title) of the reviews."""
    summary = str(row['Summary']) if pd.notnull(row['Summary']) else ""
    return summary

def kmeans_clustering(k, X, random_state):
    """Clustering with K-means."""
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
    silhouette = silhouette_score(X, kmeans.labels_)
    return kmeans.labels_, silhouette, kmeans

def top_words(corpus, n=10):
    """Extract top words and sample reviews per cluster."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
    X = vectorizer.fit_transform(corpus)
    scores = np.asarray(X.mean(axis=0)).ravel()
    top_ids = np.argsort(scores)[::-1][:n]
    words = np.array(vectorizer.get_feature_names_out())[top_ids]
    return words

df = pd.read_csv("../Dataset/modified_clean_data.csv")

n_reviews = 3000
df = df.sample(n_reviews, random_state=42)
print(f"Sample size: {df.shape[0]} | Unique products: {df['ProductId'].nunique()}")

df['TextForEmbeddings'] = df.apply(preprocess_row, axis=1)

# Text embeddings using BERT/SentenceTransformer
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)
print("Calculating sentence embeddings...")
embeddings = model.encode(df['TextForEmbeddings'].tolist(), batch_size=32, show_progress_bar=True)

ks = [i for i in range(3, 9+1)]
runs = 3

results = []

print("~ * ~ * Clustering iterations ~ * ~ *")
for k in ks:
    for iteration in range(runs):
        print(f"K={k}, iteration={iteration} ...")
        labels, silh_score, km = kmeans_clustering(k, embeddings, random_state=iteration)
        results.append({'k': k, 'iteration': iteration, 'labels': labels, 'silhouette': silh_score, 'kmeans': km})

# Select the clustering with the highest Silhouette score
best_result = max(results, key=lambda x: x['silhouette'])
labels = best_result['labels']
df['cluster'] = labels
best_k = best_result['k']
best_silh = best_result['silhouette']

print(f"\nBest result: K={best_k}, Silhouette score={best_silh:.4f}")

# PCA for cluster plotting in 3D
pca = PCA(n_components=3)
principal_components = pca.fit_transform(embeddings)
df['pc1'] = principal_components[:,0]
df['pc2'] = principal_components[:,1]
df['pc3'] = principal_components[:,2]

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    df['pc1'], df['pc2'], df['pc3'], c=df['cluster'],
    cmap='tab10', alpha=0.7,
)

ax.set_title(f"Clusters (K={best_k}) in 3D", fontsize=13)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

legend1 = ax.legend(
    *scatter.legend_elements(),
    title="Cluster"
)
ax.add_artist(legend1)

plt.tight_layout()
plt.savefig("clusters.png")
plt.show()

# Print best cluster info
for i in range(best_k):
    cluster_reviews = df[df['cluster'] == i]
    print(f"\n* ~ * ~ Cluster {i} | Size: {len(cluster_reviews)} * ~ * ~")
    print("Mean score:", round(cluster_reviews['Score'].mean(), 2))
    print("Unique products:", cluster_reviews['ProductId'].nunique())
    print("Top words:", ", ".join(top_words(cluster_reviews['TextForEmbeddings'], n=12)))
    print("\n- Example review:")
    print(cluster_reviews['TextForEmbeddings'].sample(1, random_state=2).values[0][:600], "...") # first 600 chars of review

from scipy.spatial.distance import cdist

print("\n* ~ * ~ Internal cluster cohesion (mean distance to centroid) ~ * ~ *")
for i in range(best_k):
    # Mask for this cluster
    mask = (df['cluster'] == i)
    cluster_embs = embeddings[mask]
    centroid = cluster_embs.mean(axis=0)
    mean_dist = cdist(cluster_embs, [centroid]).mean()
    print(f"Cluster {i}: Cohesion (mean dist to centroid) = {mean_dist:.4f} (n={mask.sum()})")
