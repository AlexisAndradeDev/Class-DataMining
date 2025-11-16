# -*- coding: utf-8 -*-
"""
Read docs.md for an explanation of the practice.

P4: Statistic Test

Prove that labeled data is different by running ANOVA + T test or Kruskall
Wallis test.

Martín Alexis Martínez Andrade
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pylab as py
from scipy import stats
from typing import List

df = pd.read_csv("../Dataset/modified_clean_data.csv")

df.dropna(subset=["HelpfulnessScore"], inplace=True)

# Test normality
def plot_distributions(df: pd.DataFrame, independent_var_column_name: str,
        dependent_var_column_name: str) -> List[pd.Series]:
    groups = [df[df[independent_var_column_name] == cat][dependent_var_column_name] for cat in df[independent_var_column_name].unique()]

    # plot distribution of each group
    for cat, group in zip(["very short", "short", "long", "very long"], groups):
        print(group.shape[0])
        plt.figure(figsize=(10, 6))
        plt.hist(group, bins=20, alpha=0.5, label=f'Group')
        plt.title(f'{cat} | Distribution of {dependent_var_column_name}')
        plt.xlabel(dependent_var_column_name)
        plt.ylabel('Frequency')
        plt.show()

    return groups

groups = plot_distributions(df, "LengthCategory", "HelpfulnessScore")

for group in groups:
    sm.qqplot(group, line ='45')
    py.show()

# Kruskal-Wallis
def kruskal_wallis(groups: List[pd.Series]):
    H, p = stats.kruskal(*groups)
    return H, p

def run_kruskal_wallis(df: pd.DataFrame or List[pd.Series],
        independent_var_column_name: str, dependent_var_column_name: str,
        significance_level: float = 0.05, plot: bool = False, describe: bool = True,
        drop_nulls: bool = True
    ):
    # get groups
    if isinstance(df, list) and all(isinstance(group, pd.Series) for group in df):
        groups = df
    else:
        groups = [df[df[independent_var_column_name] == cat][dependent_var_column_name] for cat in df[independent_var_column_name].unique()]

    if drop_nulls:
        # drop nulls
        print("\n~ ~ ~ NULLS ~ ~ ~")
        for group in groups:
          print(group.isna().sum(), "/", group.shape[0])
        groups = [group.dropna() for group in groups]
        print("Dropped")
        print("~ ~ ~ NULLS ~ ~ ~\n")

    if plot:
        print("\n~ ~ ~ PLOT ~ ~ ~")
        # plot distribution of each group
        for group in groups:
            print(group.shape[0])
            plt.figure(figsize=(10, 6))
            plt.hist(group, bins=20, alpha=0.5, label=f'Group')
            plt.title(f'Distribution of {dependent_var_column_name}')
            plt.xlabel(dependent_var_column_name)
            plt.ylabel('Frequency')
            plt.show()
        print("~ ~ ~ PLOT ~ ~ ~\n")

    if describe:
        print("\n~ ~ ~ DESCRIBE ~ ~ ~")
        for cat, group in zip(["very short", "short", "long", "very long"], groups):
            print(cat)
            print(group.describe())
        print("~ ~ ~ DESCRIBE ~ ~ ~\n")

    print("\n~ ~ ~ KRUSKAL-WALLIS ~ ~ ~")
    H, p = kruskal_wallis(groups)
    print(f"Kruskal-Wallis H = {H:.4f}, p = {p:.4e}")
    print("~ ~ ~ KRUSKAL-WALLIS ~ ~ ~\n")

    print("\n~ ~ ~ HYPOTHESIS ~ ~ ~")
    reject_null_hypothesis = p < significance_level
    if p < significance_level:
        print(f"Reject null hypothesis with significance level {significance_level}: there is at least one group that produces better (or worse) outcomes than the rest.")
    else:
        print("Fail to reject null hypothesis: all groups produce similar outcomes.")
    print("~ ~ ~ HYPOTHESIS ~ ~ ~\n")

    return H, p, reject_null_hypothesis

run_kruskal_wallis(df, "LengthCategory", "HelpfulnessScore", describe=True, drop_nulls=False)

