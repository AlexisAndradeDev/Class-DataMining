# Practice 2: Descriptive Statistics

**(In this practice, reading the notebook is more clear)**.

## Add a HelpfulnessScore column

It would be useful to have a HelpfulnessNumerator/Denominator column to avoid boilerplate code.

## Reviews not voted

~47% of the reviews haven't been voted by other users as helpful or not helpful. Dropping these reviews would reduce dramatically the statistical power of future analysis; also, it's not mandatory dropping them, as it's not an essential column for most analysis. They were not dropped.

## Distributions

### Score

Most reviews have a high score (5 stars category is predominant). The score distribution is biased towards positive values.

### HelpfulnessScore

Most reviews have a high helpfulness score (reviews are often voted as useful by other users). The helpfulness score distribution is highly biased towards positive values.

As mentioned before, ~47% of the reviews are not voted by other users.

## Statistics from grouped data

### User

**14.47% of users are trolls (consistently low usefulness scores in their reviews)**. Users with less than 0.5 mean helpfulness score and 10 or more reviews are considered as trolls.

### Products

**There are 9 products which have 200+ scores that are generally shared among buyers**. Also, we can see their average scores are no less than 4. Thus, we should promote them in the landing page: they're popular but also consistently highly valued by buyers.

### Reviews

There's not a significant difference in the mean helpfulness of each length category (very short, short, long, very long; using quartiles). Thus, the length of a review is not related to its helpfulness score.

We got a Pearson coefficient of 0.041. The Pearson coefficient confirms our suspicion: there isn't a *strong* correlation between the length of a review and its helpfulness; thus, **longer reviews are not necessarily more helpful**.


