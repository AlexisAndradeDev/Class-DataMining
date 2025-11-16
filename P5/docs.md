# Practice 5: Linear model

**(In this practice, reading the notebook is more clear)**.

## What should we create a linear model for?

Let's see all of the variables we have, maybe we can have an initial hypothesis on linearity.

## After running the correlation matrix
Now, as you will realize later, there isn't a single linear relationship with the current variables. The highest $R^2$ gotten was $0.1$.

So, I had to experiment with *review text complexity vs helpfulness score* and *review text sentiment vs helpfulness score*. For shortness, in this practice I show the current variables' failed linear models and a linear model for the sentiment rating extracted by BERT model (TextBlob and VADER Sentiment were both tested, and their $R^2$ was around $0$).

Still, $R^2$ for *review text sentiment vs helpfulness score* is low ($R^2 ≃ 0.17$), but it's at least stronger than any of the other linear relationships.

## Testing relationships with our current variables

First, let's test if the amount of reviews a product has keeps a linear relationship with the amount of days passed since its first review. This might be useful to predict consumer interactions in the form of reviews, projecting since the first review made.

$R^2 ≃ 0.1$ indicates a really weak linear relationship.

Now, let's test if reviews with more votes tend to be more useful to other users. Let's use the helpfulness denominator (amount of votes) and the helpfulness score (positive votes / amount of votes).

There's an amazing $R^2$ of $0$. There's no linear relationship between number of votes and helpfulness of a review *at all*.

Now, let's test if there's a linear relationship between the length of a review and its helpfulness score.

There's an incredibly low linear relationship of $R^2 ≃ 0.0017$ between the length of a review and its helpfulness.

Let's change our approach.

## Review sentiment vs Helpfulness Score

We now have to resort to exploring totally different relationships.

We'll do a sentiment analysis on the text of the reviews, and try to find a linear relationship with how helpful the review is.

First, I used `TextBlob`. The results were poor, due to the lack of information captured by TextBlob's sentiment analysis.

$R^2 ≃ 0.05$ indicates a really weak linear relationship using TextBlob for sentiment analysis.

With vaderSentiment, there's a weak relationship ($R^2 ≃ 0.125$), but stronger than any other previous relationship we've tested so far. It seems like there's a dense area in the top-right corner, which indicates reviews with really positive sentiment usually have high helpfulness scores. Still, there are a lot of datapoints across all sentiment values and helpfulness score values, so it's not a really strong relationship.

### Using absurd amounts of computing (BERT)

So far, the best $R^2$ is $0.125$, gotten by VADER Sentiment analysis.

Now, running the next code is optional and not recommended unless you can use a GPU. The code was run in Google Colab, so a (free!) T4 GPU was used and the analysis lasted a few seconds; using a CPU, it might take several minutes (in CPU-only Colab, it takes up to 60 minutes).

You can see the results gotten with BERT in the notebook (`.ipynb` file).

The relationship is a bit stronger than with VADER Sentiment analysis ($R^2 ≃ 0.16$, using a sample of $n=2000$; using $n=1000$ with the same `random_state=73`, $R^2 ≃ 0.17$). Nonetheless, it's still a pretty weak linear relationship, and there's the possibility of $R^2$ being higher because of the discrete sentiment scoring (BERT outputs discrete values: 1, 2, 3, 4 and 5, which are then normalized to 0, 0.25, 0.50, 0.75 and 1.0). You can see the helpfulness score accumulates more and more in the upper values as you approach the highest possible sentiment score.

## Conclusion

Even in the first practices (P1, P2, P3), the descriptive statistics analysis, data exploration and plotting indicated there was no linear relationship between any of the variables. In this practice (P5), multiple linear models were tested, and none indicated a decent linear relationship between two variables. Even text complexity analysis (Flesch reading ease, type-token ratio, among other techniques) were tested separate from this practice (all of them had really low $R^2$).

The sentiment of a review vs its helpfulness score seems to be the strongest linear relationship there is between two variables in the `Amazon Fine Foods Reviews` dataset, with an $R^2$ of between $0.125$ (VADER Sentiment) and $0.17$ (BERT Multilingual Uncased), depending on the method used to analyze sentiment.

# (!) Post-analysis

There was a really obvious relationship I didn't notice for some reason: Score and Helpfulness Score. In the correlation matrix, there's a 0.37 Pearson coefficient for Score vs Helpfulness Score. Let's do a quick linear model.

There's a weak linear relationship ($R^2 ≃ 0.137$) between the score given by a review to the product and its helpfulness score ($\text{positive votes} / \text{total votes}$). It's stronger than *Sentiment vs Helpfulness Score* using VADER Sentiment Analysis, but weaker than *Sentiment vs Helpfulness Score* using BERT. Nonetheless, the conclusion is still the same (no strong linear relationship in the dataset).