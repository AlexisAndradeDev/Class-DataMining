# Practice 3: Data Visualization

**(In this practice, reading the notebook is more clear)**.

## Distribution of the variables

### Score

~78% of reviews score 4 stars or higher. Also, 1 stars reviews are slightly more frequent than 2 stars reviews and 3 stars reviews.

### Review lengths

As shown by the quartiles and the histogram plot, 75% of reviews do not surpass 529 characters. Let's see how 529+ characters reviews distribute.

Even seeing the distribution of 529+ characters reviews, the histogram shows a Pareto-like distribution.

Running this iterations on the top length reviews, we can see the distribution seems to be a Pareto distribution even in the 6th iteration of getting the upper quartile.

25% of reviewers write reviews longer than 529 characters; 25% of 25% (6.25%) of reviewers use more than 1088 characters in their reviews. **Thus, 93.75% of reviewers' length needs can be covered by setting a maximum review length of 1088**.

## Review's helpfulness score voted by other users and score given to the product by that review

In the boxplots and boxen catplots, you can see the all quartiles get higher in Helpfulness Score the higher the review given by the score gets, and low helpfulness scores are shown as outliers in 4 and 5 stars reviews. This means the Helpfulness Score is frequently higher for higher review score values.

**This indicates that positive reviews are more positively voted by other users; negative reviews are more frequently voted as not useful**. In 5 stars reviews, Q1 shows up at 1.0 Helpfulness Score and points below 1.0 are showed as outliers, meaning almost all really positive reviews are voted as useful; in 1 star reviews, Q2 shows up at 0.55, meaning half the reviews (Q2 is 50% of the count) are voted as useful by half of the users (0.55 Helpfulness Score, meaning 55% of positive votes).

Maybe an important part of 1 star reviews are made by trolls or are not explained clearly; heavily negative reviews might also provoke negative feelings on voting users.

## Relationship between review length and helpfulness

**There seems to be no correlation between the length of a review and its helpfulness score**. This apparent relationship is showed by the fact that for each review length, there's a vertical area almost completely filled with points; thus, there's great diversity in Helpfulness Score for each review length and no clear shape/correlation.

In the previous practice (P2), we got a Pearson coefficient of 0.041, indicating no strong linear correlation between the length of a review and its voted helpfulness.