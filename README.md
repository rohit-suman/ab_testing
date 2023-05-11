# ab_testing
AB testing for a new version of e-commerce site

The current conversion rate of thsi website is about 13% on average throughout the year, and that the team would be happy with an increase of 2%, meaning that the new design will be considered a success if it raises the conversion rate to 15%.

Formulating a hypothesis:

Given we don’t know if the new design will perform better or worse (or the same?) as our current design, we’ll choose a two-tailed test:
Hₒ: p = pₒ
Hₐ: p ≠ pₒ
where p and pₒ stand for the conversion rate of the new and old design, respectively. We’ll also set a confidence level of 95%:
α = 0.05
The α value is a threshold we set, by which we say “if the probability of observing a result as extreme or more (p-value) is lower than α, then we reject the Null hypothesis”. Since our α=0.05 (indicating 5% probability), our confidence (1 — α) is 95%.

•	A control group - They'll be shown the old design
•	A treatment (or experimental) group - They'll be shown the new design

For our Dependent Variable (i.e. what we are trying to measure), we are interested in capturing the conversion rate. A way we can code this is by each user session with a binary variable:
•	0 - The user did not buy the product during this user session
•	1 - The user bought the product during this user session
This way, we can easily calculate the mean for each group to get the conversion rate of each design.


Steps:
1.	Calculating the sample
2.	Collecting and preparing the data
3.	Sampling
4.	Visualising the results
5.	Testing the hypothesis
6.	Drawing conclusions

Conclusion: Since our p-value=0.732 is way above our α=0.05 threshold, we cannot reject the Null hypothesis Hₒ, which means that our new design did not perform significantly different (let alone better) than our old one.
