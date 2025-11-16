# P9: Text Analysis (Word Clouds)

A comparison of word clouds between positive and negative reviews to discover which themes and motives are common in each extreme.

## Objective

To answer:
* What do people who leave 5-star reviews praise the most?
* What are the main issues (words/emotions) in 1-star reviews?

## Methodology

* Filter the reviews into clearly positive (score 4 or 5) and clearly negative (score 1 or 2) groups.
* Generate one word cloud for each group, removing standard stopwords and extra terms that would add noise.
* By visually comparing the two word clouds, identify the answer to the questions presented above.

## Results

### Word Cloud for Positive Reviews
![Positive word cloud](Word%20Cloud%20-%20Positive%20Reviews%20(4-5%20stars).png "Frequent words in positive reviews")

* The largest and most central word is "good", followed by "great", "taste", "love", "flavor", and "tea" and "coffee".
* These reviews often praise the taste and quality of the product, as well as general satisfaction and enjoyment ("love", "best", "favorite", "perfect").
* Even in positive reviews, "taste" and "flavor" are among the most discussed topics, suggesting that flavor is crucial for customer happiness.
* The strong presence of "tea" and "coffee" among the biggest words may indicate that these product categories generate substantial review volume, or that consumers have stronger opinions and loyalty towards these beverages.
* Words like "easy", "enjoy", and "nice" also appear, which could mean that ease of use, convenience, and perceived pleasantness contribute heavily to forming a positive perception.

### Word Cloud for Negative Reviews
![Negative word cloud](Word%20Cloud%20-%20Negative%20Reviews%20(1-2%20stars).png "Frequent words in negative reviews")

* The most prominent word is "taste", it is much larger than any other, indicating it's the main source of complaints. Other large words include "flavor", "coffee", "tea", "buy", "even", "bad", "price", "dog", "bag", "order", and "food".
* These reviews focus on disappointment with taste and flavor, as well as negative overall experiences such as "bad", "never", "waste", and "disappointed".
* There are frequent references to product categories ("dog", "coffee", "tea", "chocolate", "food"), suggesting that these are not only popular but may also be especially polarizing and sensitive to taste/quality perception.
* Words like "buy", "price", "waste", and "money" are frequent, highlighting buyers' regrets, perceived lack of value, or that the product did not meet expectations for its cost.
* Interestingly, "good" is still prominently used even in the negative cloud; this may suggest comparisons ("not as good as..."), or consumers using nuanced language ("would be good if...").

## Conclusion

The side-by-side word cloud visualization reveals that **taste** is the key topic in both positive and negative reviews, but in negative reviews it overshadows everything else. For positive ratings, "good", "great", and "love" are central, reflecting satisfaction, delight, and a match of expectations. For negative ratings, "taste" is the main reason for disappointment, with economic regret ("price", "waste", "money") amplifying dissatisfaction.

Several product categories, notably tea, coffee, and dog food, appear strongly in both types of reviews, suggesting both high sales volumes and high stakes for taste and preference.

Overall, these findings suggest that taste is the critical factor determining customer sentiment in food reviews, and that perceived value and ease of use also shape the extremes of satisfaction and dissatisfaction.

Word cloud text analysis is a powerful visual tool for companies to spot their product strengths and the main sources of customer dissatisfaction.