#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# # 1.Calculating the sample size

# In[1]:


# Packages imports
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.api as sms
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

get_ipython().run_line_magic('matplotlib', 'inline')

# Some plot styling preferences
plt.style.use('seaborn-whitegrid')
font = {'family' : 'Helvetica',
        'weight' : 'bold',
        'size'   : 14}

mpl.rc('font', **font)

#the current conversion rate is about 13% on average throughout the year, and that the team would be happy with an 
#increase of 2%, meaning that the new design will be considered a success if it raises the conversion rate to 15%.
# Calculating effect size based on our expected rates
effect_size = sms.proportion_effectsize(0.13, 0.15) 

#power of the test: This represents the probability of finding a statistical difference 
#between the groups in our test when a difference is actually present.
#This is usually set at 0.8 by convention.
required_n = sms.NormalIndPower().solve_power(
    effect_size, 
    power=0.8, 
    alpha=0.05, 
    ratio=1
    )                                                  # Calculating sample size needed

required_n = ceil(required_n)                          # Rounding up to next whole number                          

print(required_n)


# 
# # We’d need at least 4720 observations for each group.

# # 2. Collecting and preparing the data

# In[2]:


df = pd.read_csv("E:/TISS_Analytics_4th_Sem/AB_Testing/AB_test.csv")
df.head()


# In[3]:

df.info()


# In[4]:


# To make sure all the control group are seeing the old page and viceversa

pd.crosstab(df['group'], df['landing_page'])


# In[5]:


#Before we go ahead and sample the data to get our subset, 
#let’s make sure there are no users that have been sampled multiple times.
session_counts = df['user_id'].value_counts(ascending=False)
multi_users = session_counts[session_counts > 1].count()

print(f'There are {multi_users} users that appear multiple times in the dataset')


# In[6]:


#There are, in fact, 3894 users that appear more than once. Since the number is pretty low, 
#we’ll go ahead and remove them from the DataFrame to avoid sampling the same users twice.

users_to_drop = session_counts[session_counts > 1].index

df = df[~df['user_id'].isin(users_to_drop)]
print(f'The updated dataset now has {df.shape[0]} entries')


# # 3. Sampling

# In[9]:


control_sample = df[df['group'] == 'control'].sample(n=required_n, random_state=22) #required_n = 4720 calculated earlier
treatment_sample = df[df['group'] == 'treatment'].sample(n=required_n, random_state=22)
#I’ve set random_state=22 so that the results are reproducible

ab_test = pd.concat([control_sample, treatment_sample], axis=0)
ab_test.reset_index(drop=True, inplace=True)
ab_test


# In[10]:


ab_test.info()


# In[11]:


ab_test['group'].value_counts()


# # 4. Visualize the result

# In[12]:


conversion_rates = ab_test.groupby('group')['converted']

std_p = lambda x: np.std(x, ddof=0)              # Std. deviation of the proportion
se_p = lambda x: stats.sem(x, ddof=0)            # Std. error of the proportion (std / sqrt(n))

conversion_rates = conversion_rates.agg([np.mean, std_p, se_p])
conversion_rates.columns = ['conversion_rate', 'std_deviation', 'std_error']


conversion_rates.style.format('{:.3f}')


# In[13]:


#Judging by the stats above, it does look like our two designs performed very similarly, 
#with our new design performing slightly better, approx. 12.3% vs. 12.6% conversion rate.
#Plotting the data will make these results easier to grasp:
plt.figure(figsize=(8,6))

sns.barplot(x=ab_test['group'], y=ab_test['converted'], ci=False)

plt.ylim(0, 0.17)
plt.title('Conversion rate by group', pad=20)
plt.xlabel('Group', labelpad=15)
plt.ylabel('Converted (proportion)', labelpad=15);


# In[14]:


#So… the treatment group's value is higher.But is this difference statistically significant?


# # 5. Testing the Hypothesis

# In[15]:


#Since we have a very large sample, we can use the normal approximation for calculating our p-value (i.e. z-test).


# In[16]:


from statsmodels.stats.proportion import proportions_ztest, proportion_confint

control_results = ab_test[ab_test['group'] == 'control']['converted']
treatment_results = ab_test[ab_test['group'] == 'treatment']['converted']
n_con = control_results.count()
n_treat = treatment_results.count()
successes = [control_results.sum(), treatment_results.sum()]
nobs = [n_con, n_treat]

z_stat, pval = proportions_ztest(successes, nobs=nobs)
(lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(successes, nobs=nobs, alpha=0.05)

print(f'z statistic: {z_stat:.2f}')
print(f'p-value: {pval:.3f}')
print(f'ci 95% for control group: [{lower_con:.3f}, {upper_con:.3f}]')
print(f'ci 95% for treatment group: [{lower_treat:.3f}, {upper_treat:.3f}]')


# # 6. Drawing Conclusion

# In[ ]:


#Since our p-value=0.732 is way above our α=0.05 threshold, we cannot reject the Null hypothesis Hₒ, 
#which means that our new design did not perform significantly different (let alone better) than our old one.
#Additionally, if we look at the confidence interval for the treatment group ([0.116, 0.135], or 11.6-13.5%) we notice that:
#It includes our baseline value of 13% conversion rate
#It does not include our target value of 15% (the 2% uplift we were aiming for)
#What this means is that it is more likely that the true conversion rate of the new design is similar to our baseline, 
#rather than the 15% target we had hoped for. 
#This is further proof that our new design is not likely to be an improvement on our old design,
#and that unfortunately we are back to the drawing board!


# In[ ]:




