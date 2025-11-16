# -*- coding: utf-8 -*-
"""
Read docs.md for an explanation of the practice.

P3: Data Visualization

Martín Alexis Martínez Andrade
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../Dataset/modified_clean_data.csv")

# Distribution of the variables
print(df.dtypes)

# Scores
score_counts = df['Score'].value_counts().sort_index()
plt.pie(score_counts, labels=score_counts.index, autopct='%1.1f%%')
plt.title("Distribution of Review Scores")
plt.show()

# Review lengths
sns.histplot(df['ReviewLength'], bins=50, color="skyblue")
plt.title("Histogram of Review Lengths")
plt.xlabel("Number of characters")
plt.show()

print(df['ReviewLength'].describe())

# As shown by the quartiles and the histogram plot, 75% of reviews do not
# surpass 529 characters. Let's see how 529+ characters reviews distribute.

df_review_length_greater_than = df[df['ReviewLength'] > 529]

plt.hist(df_review_length_greater_than["ReviewLength"], bins=50, color="skyblue")
plt.title("Histogram of Review Lengths")
plt.xlabel("Number of characters")
plt.show()

print(df_review_length_greater_than["ReviewLength"].describe())

# Still, the histogram shows a Pareto-like distribution.
# Let's run this plot a few more times.

df_review_length_greater_than = df.copy()

for i in range(6):
    # Calculate 75th percentile (upper quartile) for the current subset
    q75 = df_review_length_greater_than['ReviewLength'].quantile(0.75)

    # Filter reviews longer than this percentile
    df_review_length_greater_than = df_review_length_greater_than[df_review_length_greater_than['ReviewLength'] > q75]

    plt.hist(df_review_length_greater_than["ReviewLength"], bins=50, color="skyblue")
    plt.title(f"Iteration {i+1}")
    plt.xlabel("Number of characters")
    plt.show()

    print(f"Iteration {i+1} - Statistics for ReviewLength (N={len(df_review_length_greater_than)}, {df.shape[0]} total):")
    print(df_review_length_greater_than["ReviewLength"].describe())
    print('-'*50)
    
# Review's helpfulness score voted by other users and score given to the
# product by that review
sns.boxplot(x='Score', y='HelpfulnessScore', data=df)
plt.title("Helpfulness Score by Score given to the product in the review")
plt.xlabel("Score (stars)")
plt.ylabel("Helpfulness Score")
plt.show()

sns.catplot(x='Score', y='HelpfulnessScore', data=df, kind="boxen")
plt.title("Helpfulness Score by Score given to the product in the review")
plt.xlabel("Score (stars)")
plt.ylabel("Helpfulness Score")
plt.show()

print(df.groupby('Score')['HelpfulnessScore'].describe())

# Let's also plot this as histograms so that the boxplots are better explained.
# plot distribution of each grouped score
for score in range(1, 6):
    plt.figure(figsize=(10, 6))
    plt.hist(df[df['Score'] == score]['HelpfulnessScore'], bins=100, alpha=0.5, label=f'Score {score}')

    # draw quartiles in vertical lines
    plt.axvline(df[df['Score'] == score]['HelpfulnessScore'].quantile(0.25), color='r', linestyle='--')
    plt.axvline(df[df['Score'] == score]['HelpfulnessScore'].quantile(0.50), color='r', linestyle='--')
    plt.axvline(df[df['Score'] == score]['HelpfulnessScore'].quantile(0.75), color='r', linestyle='--')
    plt.axvline(df[df['Score'] == score]['HelpfulnessScore'].quantile(1.00), color='r', linestyle='--')

    plt.xlabel('Helpfulness Score')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Helpfulness Score for Score {score}')
    plt.show()
    print(df[df['Score'] == score]['HelpfulnessScore'].describe())

# Relationship between review length and helpfulness
sns.scatterplot(x='ReviewLength', y='HelpfulnessScore', data=df.sample(300000))
plt.title("Helpfulness vs Review Length")
plt.xlabel("Review Length (characters)")
plt.ylabel("Helpfulness Score")
plt.show()

