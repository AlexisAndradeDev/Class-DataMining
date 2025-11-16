# -*- coding: utf-8 -*-
"""
Read docs.md for an explanation of the practice.

P2: Descriptive Statistics

Martín Alexis Martínez Andrade
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("../Dataset/clean_data.csv")

print(df.describe(include="int64")[["HelpfulnessNumerator", "HelpfulnessDenominator", "Score"]])

print("\n**Variance**")
print(df[["HelpfulnessNumerator", "HelpfulnessDenominator", "Score"]].var())

print("\n**Interquartile Range**")
print(df[["HelpfulnessNumerator", "HelpfulnessDenominator", "Score"]].quantile(0.75) - df[["HelpfulnessNumerator", "HelpfulnessDenominator", "Score"]].quantile(0.25))

# create a HelpfulnessScore column
df["HelpfulnessScore"] = df["HelpfulnessNumerator"] / df["HelpfulnessDenominator"]

print(df["HelpfulnessScore"].head())
print(df["HelpfulnessScore"].describe())

num_of_nulls = df["HelpfulnessScore"].isna().sum()
# percentage of nulls
print(f"Number of NaN values in HelpfulnessScore: {num_of_nulls} | Percentage: {num_of_nulls / df.shape[0] * 100:.2f}%")

# Distributions
freq = df["Score"].value_counts().sort_index(inplace=False)
print(freq)

plt.bar(freq.index, freq.values)

# Helpfulness Score
# make intervals and then calculate frequency for HelpfulnessScore following
# the Sturges' rule
voted_reviews = df.dropna(subset=["HelpfulnessScore"], inplace=False)
n = voted_reviews.shape[0]
number_of_bins = int(1 + 3.322 * np.log10(n))
print(f"Number of bins: {number_of_bins}")

# frecuencies
freq = voted_reviews["HelpfulnessScore"].value_counts(bins=number_of_bins).sort_index(inplace=False)
print(freq)

plt.hist(df["HelpfulnessScore"], bins=number_of_bins)

num_of_nulls = df["HelpfulnessScore"].isna().sum()
n = df.shape[0]

print(f"Number of null HelpfulnessScore rows: {num_of_nulls} | Percentage: {num_of_nulls / df.shape[0] * 100:.2f}%")

plt.bar(["Null", "Non Null"], [num_of_nulls, n - num_of_nulls])

# Statistics from grouped data

# calculate statistics per user
user_stats = df.groupby('UserId').agg(
    total_reviews = ('Id', 'count'),
    avg_score = ('Score', 'mean'),
    median_score = ('Score', 'median'),
    std_score = ('Score', 'std'),
    avg_helpfulness_score = ('HelpfulnessScore', 'mean')
).sort_values('total_reviews', ascending=False)

print("User statistics")
print(user_stats.head())

# Get users with at least 10 reviews and sort them by mean helpfulness score
# get just the users with at least N reviews
n_rev = 10
users_with_considerable_reviews = user_stats[user_stats["total_reviews"] >= n_rev]

# drop users that haven't been voted and sort by helpfulness score
users_with_considerable_reviews = users_with_considerable_reviews.dropna(
    subset=["avg_helpfulness_score"], inplace=False
).sort_values('avg_helpfulness_score', ascending=True)

# categorize "troll" users
trolls = users_with_considerable_reviews[users_with_considerable_reviews["avg_helpfulness_score"] < 0.5]
num_trolls = trolls.shape[0]
num_of_users_with_considerable_reviews = users_with_considerable_reviews.shape[0]
print(f"Number of trolls: {num_trolls} | Percentage of trolls: {num_trolls / num_of_users_with_considerable_reviews * 100 : .2f}%")

# Calculate statistics per product
product_stats = df.groupby('ProductId').agg(
    total_reviews = ('Id', 'count'),
    avg_score = ('Score', 'mean'),
    median_score = ('Score', 'median'),
    mode_score = ('Score', lambda x: x.mode() if len(x.mode()) > 0 else None),
    std_score = ('Score', 'std')
).sort_values('total_reviews', ascending=False)

print("Product statistics")
print(product_stats.head())

# Get least controversial products
# We can use the standard deviation as a measure of how controversial a
# product is. Low std means there's a general agreement on the mean score;
# high std means the mean score is controversial.
# get products with at least 200 total_reviews
n_total_reviews = 200
products_by_std_score = product_stats[product_stats["total_reviews"] >= n_total_reviews]
# order by std_score
products_by_std_score = products_by_std_score.sort_values('std_score', ascending=True)

print(products_by_std_score.head())

# get the std_score that are less or equal to 1
least_controversial_products = products_by_std_score[products_by_std_score["std_score"] <= 0.75]

print(least_controversial_products)
print(f"Number of least controversial products: {least_controversial_products.shape[0]}")

# Are longer reviews more helpful?

# create a column "ReviewLength"
df["ReviewLength"] = df["Text"].apply(lambda x: len(x))

# create length categories (quartiles)
length_categories=['very_short', 'short', 'long', 'very_long']
df['LengthCategory'] = pd.qcut(df['ReviewLength'], q=4, labels=length_categories)

# Group reviews by length quantiles
# drop null HelpfulnessScore values
reviews_with_helpfulness = df.dropna(subset=["HelpfulnessScore"], inplace=False)

# get average helpfulness by length category
avg_helpfulness_by_length = reviews_with_helpfulness.groupby('LengthCategory')['HelpfulnessScore'].mean()
print("Average helpfulness score by review length category:")
print(avg_helpfulness_by_length)

plt.bar(length_categories, avg_helpfulness_by_length)
plt.title('Helpfulness vs Review Length Category')
plt.xlabel('Review Length Category')
plt.ylabel('Helpfulness Ratio')
plt.show()

# Person coefficient of length vs helpfulness score
corr = reviews_with_helpfulness["ReviewLength"].corr(reviews_with_helpfulness["HelpfulnessScore"])
print(f"Correlation between review length and helpfulness score: {corr:.3f}")

df.to_csv("../Dataset/modified_clean_data.csv", index=False)