# -*- coding: utf-8 -*-
"""
Read docs.md for an explanation of the practice.

P1: Data Cleaning

Martín Alexis Martínez Andrade
"""
import pandas as pd
import kagglehub

df = pd.read_csv("../Dataset/Reviews.csv")

# # Download latest version from Kaggle
# path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")
# df = pd.read_csv(f"{path}/Reviews.csv")

# 'Timestamp' is more clear
df.rename(columns={"Time": "Timestamp"}, inplace=True)
# Convert int timestamp to date
df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')

# Dataset summary
r, c = df.shape
print(f"{r} rows x {c} columns.")
print("\n*** COLUMNS ***")
print("*object* type is a string type in this dataset.")
print(df.dtypes)

print(df.head(n=40))

# missing values
print("\n*** MISSING VALUES ***")
print(df.isna().sum())

# Drop missing values
df = df.dropna()

print(f"Number of rows: {df.shape[0]} | Percentage of data kept: {df.shape[0] / r * 100:.2f}%")
r, c = df.shape

# Duplicate products
print(df[df["UserId"] == "A3HDKO7OW0QNK4"])

duplicable_columns = df.columns.difference(["ProductId", "Id"])

# Drop rows that have the exact same values for every column except for ProductId and Id
df = df.drop_duplicates(subset=duplicable_columns)

# Display for the same user
print(df[df["UserId"] == "A3HDKO7OW0QNK4"])

print(df.head())
print(f"Number of rows: {df.shape[0]} | Percentage of data kept: {df.shape[0] / r * 100:.2f}%")
r, c = df.shape

# HelpfulnessDenominator and HelpfulnessNumerator

print(df[df["HelpfulnessNumerator"] > df["HelpfulnessDenominator"]])

df = df[df["HelpfulnessNumerator"] <= df["HelpfulnessDenominator"]]
print(f"Number of rows: {df.shape[0]} | Percentage of data kept: {df.shape[0] / r * 100}%")
r, c = df.shape

# check if the Score is between 1 and 5, and the dates really range from Oct
# 1999 to Oct 2012.
# Also, make sure that the Helpfulness numerator and
# denominator are always greater than or equal to zero.
print(f"Min HelpfulnessNumerator: {df['HelpfulnessNumerator'].min()} | Max HelpfulnessNumerator: {df['HelpfulnessNumerator'].max()}")
print(f"Min HelpfulnessDenominator: {df['HelpfulnessDenominator'].min()} | Max HelpfulnessDenominator: {df['HelpfulnessDenominator'].max()}")

print(f"Min Score: {df['Score'].min()} | Max Score: {df['Score'].max()}")

print(f"Min Date: {df['Date'].min()} | Max Date: {df['Date'].max()}")

df.to_csv("../Dataset/clean_data.csv", index=False)
