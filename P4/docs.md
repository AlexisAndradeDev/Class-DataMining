# Practice 4: Statistical Test

**(In this practice, reading the notebook is more clear)**.

## What Analysis of Variance will be used for

We want to know whether several categories of length of reviews (very short [bottom 25%], short [25%], long [25%], very long [top 25%]) produce similar helpfulness scores, or whether one among them may produce higher helpfulness scores than the rest.

For this practice, let's drop datapoints that have null values for the Helpfulness Score column.

## Assumptions

In the analysis of variance (ANOVA) procedure, the $F$ test is based on the assumption that independent random samples are taken from normal populations with equal variances.

Let's see if these assumptions are correct for our dataset.

### Normality

As you might instantly realize, the distributions are not normal. Because of this, the ANOVA test cannot be run (because of the F test, which requires normality, and having a non-normal distribution would lead to inaccuracies). Instead, the Kruskal-Wallis test will be used.

Let's prove non-normality with a Quantile-Quantile plot. All of the normality tests (Shapiro, Kolmogorov-Smirnov, etc.) almost certainly reject normality when applied to large datasets, even if they're normal; as we have hundreds of thousands of datapoints, it's better to use a Quantile-Quantile plot.

In the Quantile-Quantile plots, none of the distributions match the 45° line at all, so they do not align with a normal distribution.

Let's use the Kruskal-Wallist test.

## Kruskal-Wallis test

### Assumptions

Like other nonparametric techniques, the Kruskal-Wallis procedure requires no assumptions about the actual form of the probability distributions. Nonetheless, we still have to make the following assumptions:

1. [✔] **Independence of Observations**: Each observation within the dataset should be independent of all other observations, meaning one data point does not influence another.
2. [✔] **Ordinal or Continuous Dependent Variable**: The dependent variable should be measured on an ordinal scale or a continuous scale.
3. [✔] **Independence of Samples**: The samples from each group must be independent of each other.

### Procedure

We let the sample sizes to be unequal, $n_i$ for $i=1,2,...,k$ represents the size of the sample drawn from the $i$th population.

We combine all the $\sum\limits^k_{i=1}{n_i}=n$ observations and rank them from 1 (smallest) to $n$ (largest). If two or more observations are tied for the same rank, then the average of the ranks that would have been assigned to these observations is assigned to each member of the tied group.

Let $R_i$ denote the sum of the ranks of the observations from population $i$ and let $\bar{R_i}= \frac{R_i}{n_i}$ denote the average of the ranks in $i$.

Let $\bar{R}$ be the overall average of all of the ranks:

$
\bar{R} = \frac{1}{k} \cdot \sum\limits^k_{i=1}{\bar{R_i}}
$,

now we can consider $V$, the rank analogue () of SST (SST is the *total sum of squares* used in the F-test-based ANOVA), which is **computed by using the ranks rather than the actual values of the measurements**:

$
V = \sum\limits^k_{i=1}{n_i(\bar{R_i} - \bar{R})^2}.
$

---

If the null hypothesis is true and the populations do not differ, we would expect the $\bar{R_i}$ values to be approximately equal and the resulting value of V to be relatively small. If the alternative hypothesis is true, we would expect this to be exhibited in differences among the values of $\bar{R_i}$, leading to a large value for V. Notice that $\bar{R} =$ sum of the first $n$ integers $/n=[n(n+1)/2]/n=(n+1)/2$ and thus that:

$
V = \sum\limits^k_{i=1}{n_i(\bar{R_i}-\frac{n+1}{2})^2}
$

Instead of focusing on V, Kruskal and Wallis (1952) considered the statistic $H = \frac{12V}{n(n+1)}$, which, after performing in V the squaring of each term in the sum and adding the resulting values, may be rewritten as:

$H = \frac{12}{n(n+1)} \sum\limits^{k}_{i=1}{(\frac{R^2_i}{n_i})} - 3(n+1)$

The null hypothesis is rejected if the value of $H$ is large. The corresponding $\alpha$-level test calls for rejection of the null hypothesis in favor of the alternative if $H>h(\alpha)$, where $h(\alpha)$ is such that, when $H_0$ is true, $P[H>h(\alpha)]=\alpha$.

In short, Kruskal-Wallis tests whether several treatment produce similar outcomes, or whether one among them may produce higher or lower outcomes than the rest; its test statistics is $H$, and $H_0$ is rejected if $H>x^2_\alpha$ with $(k-1)$ degrees of freedom.

### After running Kruskal-Wallis
We have a $p$-value of almost 0. Kruskal-Wallis is sensitive to really small differences in the distributions when the dataset is too large. Now, this is not a problem; some statistical tests are misinterpreted. As explained before, Kruskal-Wallis determines whether at least one group produces better or worse outcomes than the other groups. As seen graphically in the second practice, there's a small difference in, for example, the mean of the Helpfulness Score of each of the length categories. Kruskal-Wallis finds with complete certainty this difference in the distributions; this doesn't mean that difference is *meaningful*, it's just *statistically significant*.

A *statistically significant* difference is a difference between poblations (groups) that is proved to not be due to a random coincidence in the samples. A *meaningful* difference (*practical significance*) is obtained by considering effect sizes and understanding the context of the study and with domain knowledge; it's the real-world importance of the difference. A difference (or finding, in general) can be statistically significant but not practically meaningful, such as a tiny, but real, improvement in a medication's effectiveness, or a practically meaningful difference may not be statistically significant due to small sample sizes.

In the case of this practice, each length category has, for example, really similar means: 0.79, 0.8, 0.74, 0.8; the $p$-value obtained in the Kruskal-Wallis test was $p=2.061 \times 10^{-151} ≈ 0$, so the null hypothesis is rejected, meaning that at least one of the length categories produces better (or worse) helpfulness scores than the other categories. Looking at the means, the *long* category produces significantly lower helpfulness scores than the other categories; this significance is highlighted because of the huge size of the dataset. Is this difference *meaningful*? It depends on the interpretation of the analyst; in this example, there's a 0.06 difference in the helpfulness score, which might not be a meaningful difference.

