# Practice 1: Data Cleaning

**(In this practice, reading the notebook is more clear)**.

https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review.

There are two variables that might need an explanation:
* **HelpfulnessNumerator**: Number of users who found the review helpful.
* **HelpfulnessDenominator**: Number of users who indicated whether they found the review helpful or not.

## Missing values

There are a few null values. They don't even represent 1% of the rows, so dropping them won't reduce the statistical power.

## Duplicated values

There are duplicated rows for some users that only differ in the ProductId and Id. It can be inferred that these duplicated rows belong to the same product, just for different particular characteristics (flavours or quantities); the same review is created at the same time for each variation of the product. In order to reduce redundancy, the duplicated reviews should be removed.

The duplicates were removed and there is now no redundancy. The first ProductId is kept.

Almost 30% of the rows consisted of duplicate product reviews. ~400,000 reviews are unique and kept.

## HelpfulnessDenominator and HelpfulnessNumerator

The number of users who indicated whether they found the review helpful or not (denominator) cannot be less than the number of users who found the review helpful (numerator). There are just two rows that do not comply with this rule; they were dropped.

The helpfulness numerator and denominator start at 0.

The scores range from 1 to 5.

Dates range from October 8th, 1999 to October 26th, 2012.

Both the score and date columns are clean.

## Conclusion

After cleaning the dataset, there are no null values and no duplicates; the Helpfulness numerator is always less than or equal to the denominator and are positive numbers or zero; scores and dates are in the correct range.