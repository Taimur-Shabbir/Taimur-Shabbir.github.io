---
title: "End to End ML Project: Salary Prediction"
date: 2021-24-05
tags: [Data Visualisation, Machine Learning, Feature Engineering]
#header:
image: j.jpg
#mathjax: "true"
---

# Part 1 - Defining the Problem

The business problem I am facing relates to predicting the salaries of a set of new job postings. I aim to train a machine learning model on data related to existing job postings and their salaries, and generalise that to accurately predict salaries for new postings.


```python
# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# change settings for plots
#plt.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
#plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('fivethirtyeight')




#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"


__author__ = "Taimur Shabbir"
__email__ = "alitaimurshabbir@hotmail.com"
```

## Part 2 - Discovering the Data

### Load the data


```python
data_features = pd.read_csv('/Users/alitaimurshabbir/Desktop/salary-prediction/data/train_features.csv')
data_outcomes = pd.read_csv('/Users/alitaimurshabbir/Desktop/salary-prediction/data/train_salaries.csv')
data_combined = pd.merge(data_features, data_outcomes, on = 'jobId')
```


```python
# checking whether 'merge' was executed successfully by examining number of rows

print(len(data_features))
print(len(data_outcomes))
print(len(data_combined))
```

    1000000
    1000000
    1000000


###  Cleaning data


```python
# As a first step, I will find what data is missing. We see that no data is missing:

data_combined.isnull().sum()
```




    jobId                  0
    companyId              0
    jobType                0
    degree                 0
    major                  0
    industry               0
    yearsExperience        0
    milesFromMetropolis    0
    salary                 0
    dtype: int64




```python
# checking for potentially incorrect data among numerical variables

data_combined.describe()

data_combined[data_combined['salary'] == 0]

# There are a few records for salaries with a value of 0. These must be incorrect
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>jobId</th>
      <th>companyId</th>
      <th>jobType</th>
      <th>degree</th>
      <th>major</th>
      <th>industry</th>
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>30559</td>
      <td>JOB1362684438246</td>
      <td>COMP44</td>
      <td>JUNIOR</td>
      <td>DOCTORAL</td>
      <td>MATH</td>
      <td>AUTO</td>
      <td>11</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <td>495984</td>
      <td>JOB1362684903671</td>
      <td>COMP34</td>
      <td>JUNIOR</td>
      <td>NONE</td>
      <td>NONE</td>
      <td>OIL</td>
      <td>1</td>
      <td>25</td>
      <td>0</td>
    </tr>
    <tr>
      <td>652076</td>
      <td>JOB1362685059763</td>
      <td>COMP25</td>
      <td>CTO</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>AUTO</td>
      <td>6</td>
      <td>60</td>
      <td>0</td>
    </tr>
    <tr>
      <td>816129</td>
      <td>JOB1362685223816</td>
      <td>COMP42</td>
      <td>MANAGER</td>
      <td>DOCTORAL</td>
      <td>ENGINEERING</td>
      <td>FINANCE</td>
      <td>18</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <td>828156</td>
      <td>JOB1362685235843</td>
      <td>COMP40</td>
      <td>VICE_PRESIDENT</td>
      <td>MASTERS</td>
      <td>ENGINEERING</td>
      <td>WEB</td>
      <td>3</td>
      <td>29</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



The best way to deal with this missing data is to drop the relevant rows, for two reasons.

- First, the missing component is the outcome variable, salary, so we cannot use the traditional methods of data replacement we would use with missing values of features


- Second, we have 1 million rows in our table. Dropping 5 rows is going to be a trivial loss of data.


```python
data_combined = data_combined.drop(data_combined[data_combined.salary == 0].index)
```

Next I want to check the unique values for a few columns with the 'object' data type. This is to see, for example, if there are misspellings for entries in the 'jobType' column.


```python
data_combined['jobType'].value_counts()
```




    SENIOR            125886
    VICE_PRESIDENT    125234
    MANAGER           125120
    CTO               125045
    JANITOR           124971
    CEO               124778
    JUNIOR            124592
    CFO               124369
    Name: jobType, dtype: int64




```python
data_combined['degree'].value_counts()
```




    HIGH_SCHOOL    236975
    NONE           236853
    BACHELORS      175495
    DOCTORAL       175362
    MASTERS        175310
    Name: degree, dtype: int64




```python
data_combined['major'].value_counts()
```




    NONE           532353
    CHEMISTRY       58875
    LITERATURE      58684
    ENGINEERING     58594
    BUSINESS        58518
    PHYSICS         58410
    COMPSCI         58382
    BIOLOGY         58379
    MATH            57800
    Name: major, dtype: int64



There are no misspellings for the values in any of the columns investigated.

###  Exploring data (EDA)

**Investigate 'salary':**


```python
plt.figure(figsize=(12,6))

sns.distplot(data_combined['salary'], kde=False, bins=40, color = 'teal')

plt.title('Distribution of Salary in Arbitrary Units', fontsize = 18)
plt.xlabel('Salary (Arbitrary Units)')
plt.ylabel('Frequency')

```




    Text(0, 0.5, 'Frequency')




![png](output_20_1.png)


**Investigate numerical variables 'yearsExperience' and 'milesFromMetropolis':**


```python
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 8))

ax1.boxplot(data_combined['yearsExperience'],
            showfliers = True, patch_artist = True,
            boxprops = dict(facecolor = 'cadetblue', color = 'cadetblue'))

ax1.set_title('Distribution of Years of Experience', fontsize = 16)
ax1.set_ylabel('Years')



ax2.boxplot(data_combined['milesFromMetropolis'],
            showfliers = True, patch_artist = True,
            boxprops = dict(facecolor = 'powderblue', color = 'powderblue'))

ax2.set_title('Distribution of Miles From Metropolis', fontsize = 16)
ax2.set_ylabel('Miles')
```




    Text(0, 0.5, 'Miles')




![png](output_22_1.png)


#### Investigate relationship between the above two interval variables and 'salary'


```python
# take a small random sample of data for better visualisation

small_sample_data = data_combined.sample(n = 3000, random_state = 42)

plt.figure(figsize = (12, 8))

plt.scatter(small_sample_data['milesFromMetropolis'],
            small_sample_data['salary'], alpha = 0.5,
            c = 'cadetblue')

plt.title('Miles From Metropolis Versus Salary')
plt.xlabel('Distance From Metropolis (Miles)')
plt.ylabel('Salary (Arbitrary Units)')
```




    Text(0, 0.5, 'Salary (Arbitrary Units)')




![png](output_24_1.png)



```python
plt.figure(figsize = (12, 8))

plt.scatter(small_sample_data['yearsExperience'],
            small_sample_data['salary'], alpha = 0.8,
            c = 'powderblue')

plt.title('Years of Experience Versus Salary')
plt.xlabel('Work Experience (Years)')
plt.ylabel('Salary (Arbitrary Units)')
```




    Text(0, 0.5, 'Salary (Arbitrary Units)')




![png](output_25_1.png)


**Interpretation:**

- 'Salary' is fairly normally distributed with a slight positive skew, which means the mean and the median are greater than the mode. The mean is being 'pulled' up by a few instances with very large values (above 250)


- The median of years of work experience required is 12. 50% of the postings require between approximately 6 and 17 years. There are cases where the job posting is aimed at those who are starting their careers, with 0 years of experience, and cases where postings require candidates who have been in the workforce for a long time, approaching 24 years


- On the other hand, there are a few jobs available in Metropolis (0 miles away from this city) and a few who require a long commute (nearly 100 miles). These are extreme points, as the majority of postings lie between 25 and 75 miles


- Distance from Metropolis and Salary have a very weak linear and negative relationship. Conversely, Years of Work experience and Salary have a very weak linear and positive relationship


- Since these are our two main interval features, we can question whether we need to scale them. From the y-axes of both boxplots, there is a noticeable difference in the magnitude of the data. We keep in mind this observation for now and will return to it if needed


----

#### Next I want to see the distributions of salary among different types industries.

I will choose 3 diverse values for this variable just to get an idea of the data:

'AUTO', 'OIL' and 'EDUCATION'


```python
# create dataframes

auto_salary_df = data_combined.loc[(data_combined['industry'] == 'AUTO')]

oil_salary_df = data_combined.loc[(data_combined['industry'] == 'OIL')]

education_salary_df = data_combined.loc[(data_combined['industry'] == 'EDUCATION')]
```


```python
# create plots

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 8), sharey = True)



ax1.boxplot(auto_salary_df['salary'], patch_artist = True,
            boxprops=dict(facecolor= 'firebrick', color='firebrick'))

ax1.set_xlabel('Automobile Industry', fontsize = 16)
ax1.set_ylabel('Salary (Arbitrary Units)', fontsize = 16)
ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax1.set_ylim(0, 250)


ax2.boxplot(oil_salary_df['salary'], patch_artist = True,
            boxprops = dict(facecolor = 'black', color = 'black'))

ax2.set_xlabel('Oil Industry', fontsize = 16)
ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)


ax3.boxplot(education_salary_df['salary'], patch_artist = True,
            boxprops = dict(facecolor = 'darkgreen', color = 'navy'))

ax3.set_xlabel('Education Industry', fontsize = 16)
ax3.tick_params(axis = 'both', which = 'major', labelsize = 12)


ax2.set_title('Distribution of Salary Among Automobile, Oil and Education Industries', fontsize = 22)

ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)
```


![png](output_29_0.png)


**Interpretation**

There are noticeable but small differences among the 3 chosen industry in terms of salary distribution.

- The highest-paying job postings in the Oil industry earn the most compared to their counterparts in the other two industries. The same can be said for the lowest-paying roles


- The middle 50% of job postings in Oil also pay more than the middle 50% in the Automobile and Education industries


- The salaries offered for jobs in the Automobile industry seem to lie in the middle of the other two industries


- Substantial numbers of outliers exist for all 3 industries

This visualisation suggests there may be a link between the type of industry one is in and the salary offered by the job. As a result, the type of industry may have some predictive power in computing new salaries.

---
#### Investigating mean salary per industry


```python
industry_salary_df = data_combined.groupby('industry').mean().reset_index()
```


```python
plt.figure(figsize = (12, 6))
plt.bar(industry_salary_df['industry'], industry_salary_df['salary'], color = 'mediumpurple')
plt.title('Average Salary Offered By Industry')
plt.ylabel('Salary (Arbitrary Units)')
plt.xlabel('Industry')
```




    Text(0.5, 0, 'Industry')




![png](output_33_1.png)


**Interpretation**

- Oil and Finance offer the highest paying jobs on average. Education and Service conversely offer the lowest paying jobs on average

----

### Investigating how salary differs with job type/seniority


```python
jobType_df = data_combined.loc[:, ['jobType', 'salary']]
```


```python
jobType_df.boxplot(by="jobType",
                   column="salary",
                   patch_artist = True,
                   showfliers = True,
                   figsize = (14, 8))

plt.title('How Salary Differs With Job Seniority')
plt.ylabel('Salary (Arbitrary Units)')
plt.xlabel('Type/Seniority of Job')
plt.tick_params(axis = 'both', which = 'major', labelsize = 9)
plt.show()
```


![png](output_37_0.png)


**Interpretation**

- The data suggests what one might expect; more senior roles usually pay higher salaries. The middle 50%, lowest paying and highest paying CEO, CFO and CTO jobs pay the most on average compared to other roles.


- Janitor and Junior jobs pay the least


- The distribution of salary is fairly well correlated with the seniority of the job overall

**Correlations between integer variables**


```python
fig, axes = plt.subplots(1, 1, figsize = (12, 10))
sns.heatmap(data_combined.corr(), annot = True, fmt = '.2f', cmap = 'mako')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f92fc507090>




![png](output_40_1.png)


**Interpretation**

- As suggested by our initial scatterplots, 'milesFromMetropolis' has a weak, negative correlation with 'salary' while 'yearsExperience' has a weak, positive correlation with 'salary', with coefficients of -0.3 and 0.38 respectively. This suggests both of these variables may have some predictive power

###  Establishing a baseline

For a baseline model, I will use the average salary per industry as the prediction.

I will then calculate RMSE to find a benchmark to improve upon.


```python
industry_salary_mean = data_combined.groupby('industry')['salary'].mean()
```


```python
# initialise variables

auto_salary = industry_salary_mean[0]
education_salary = industry_salary_mean[1]
finance_salary = industry_salary_mean[2]
health_salary = industry_salary_mean[3]
oil_salary = industry_salary_mean[4]
service_salary = industry_salary_mean[5]
web_salary = industry_salary_mean[6]
```


```python
# create new Baseline Prediction column and fill with NaNs, to be replaced shortly

data_combined['Baseline Prediction'] = np.nan
```


```python
# replace each NaN with the average salary of the industry for the relevant row

data_combined['Baseline Prediction'] = np.where(
                                      (data_combined['industry'] == 'AUTO'),
                                       auto_salary, data_combined['Baseline Prediction'])

data_combined['Baseline Prediction'] = np.where(
                                      (data_combined['industry'] == 'EDUCATION'),
                                       education_salary, data_combined['Baseline Prediction'])

data_combined['Baseline Prediction'] = np.where(
                                      (data_combined['industry'] == 'FINANCE'),
                                       finance_salary, data_combined['Baseline Prediction'])

data_combined['Baseline Prediction'] = np.where(
                                      (data_combined['industry'] == 'HEALTH'),
                                       health_salary, data_combined['Baseline Prediction'])

data_combined['Baseline Prediction'] = np.where(
                                      (data_combined['industry'] == 'OIL'),
                                       oil_salary, data_combined['Baseline Prediction'])

data_combined['Baseline Prediction'] = np.where(
                                      (data_combined['industry'] == 'SERVICE'),
                                       service_salary, data_combined['Baseline Prediction'])

data_combined['Baseline Prediction'] = np.where(
                                      (data_combined['industry'] == 'WEB'),
                                       web_salary, data_combined['Baseline Prediction'])
```


```python
data_combined
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>jobId</th>
      <th>companyId</th>
      <th>jobType</th>
      <th>degree</th>
      <th>major</th>
      <th>industry</th>
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
      <th>salary</th>
      <th>Baseline Prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>JOB1362684407687</td>
      <td>COMP37</td>
      <td>CFO</td>
      <td>MASTERS</td>
      <td>MATH</td>
      <td>HEALTH</td>
      <td>10</td>
      <td>83</td>
      <td>130</td>
      <td>115.735540</td>
    </tr>
    <tr>
      <td>1</td>
      <td>JOB1362684407688</td>
      <td>COMP19</td>
      <td>CEO</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>WEB</td>
      <td>3</td>
      <td>73</td>
      <td>101</td>
      <td>121.645362</td>
    </tr>
    <tr>
      <td>2</td>
      <td>JOB1362684407689</td>
      <td>COMP52</td>
      <td>VICE_PRESIDENT</td>
      <td>DOCTORAL</td>
      <td>PHYSICS</td>
      <td>HEALTH</td>
      <td>10</td>
      <td>38</td>
      <td>137</td>
      <td>115.735540</td>
    </tr>
    <tr>
      <td>3</td>
      <td>JOB1362684407690</td>
      <td>COMP38</td>
      <td>MANAGER</td>
      <td>DOCTORAL</td>
      <td>CHEMISTRY</td>
      <td>AUTO</td>
      <td>8</td>
      <td>17</td>
      <td>142</td>
      <td>109.435222</td>
    </tr>
    <tr>
      <td>4</td>
      <td>JOB1362684407691</td>
      <td>COMP7</td>
      <td>VICE_PRESIDENT</td>
      <td>BACHELORS</td>
      <td>PHYSICS</td>
      <td>FINANCE</td>
      <td>8</td>
      <td>16</td>
      <td>163</td>
      <td>130.747659</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>999995</td>
      <td>JOB1362685407682</td>
      <td>COMP56</td>
      <td>VICE_PRESIDENT</td>
      <td>BACHELORS</td>
      <td>CHEMISTRY</td>
      <td>HEALTH</td>
      <td>19</td>
      <td>94</td>
      <td>88</td>
      <td>115.735540</td>
    </tr>
    <tr>
      <td>999996</td>
      <td>JOB1362685407683</td>
      <td>COMP24</td>
      <td>CTO</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>FINANCE</td>
      <td>12</td>
      <td>35</td>
      <td>160</td>
      <td>130.747659</td>
    </tr>
    <tr>
      <td>999997</td>
      <td>JOB1362685407684</td>
      <td>COMP23</td>
      <td>JUNIOR</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>EDUCATION</td>
      <td>16</td>
      <td>81</td>
      <td>64</td>
      <td>99.448386</td>
    </tr>
    <tr>
      <td>999998</td>
      <td>JOB1362685407685</td>
      <td>COMP3</td>
      <td>CFO</td>
      <td>MASTERS</td>
      <td>NONE</td>
      <td>HEALTH</td>
      <td>6</td>
      <td>5</td>
      <td>149</td>
      <td>115.735540</td>
    </tr>
    <tr>
      <td>999999</td>
      <td>JOB1362685407686</td>
      <td>COMP59</td>
      <td>JUNIOR</td>
      <td>BACHELORS</td>
      <td>NONE</td>
      <td>EDUCATION</td>
      <td>20</td>
      <td>11</td>
      <td>88</td>
      <td>99.448386</td>
    </tr>
  </tbody>
</table>
<p>999995 rows × 10 columns</p>
</div>




```python
# separate features, output and baseline predictions from one another

X_train = data_combined[['jobId', 'companyId', 'jobType',
                 'degree', 'major', 'industry',
                 'yearsExperience', 'milesFromMetropolis']]

y_train = data_combined['salary']

y_predicted = data_combined['Baseline Prediction']
```


```python
# calculate MSE

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_train, y_predicted))

print('The baseline model RMSE is {}'.format(rmse))
```

    The baseline model RMSE is 36.974625769373986


For referencial purposes, I will also calculate MSE:


```python
mse = rmse**2
print('The baseline model MSE is {}'.format(mse))
```

    The baseline model MSE is 1367.122950785255


### Hypothesising A Solution

The three models I have chosen are

**Linear and Polynomial Regression**: Linear regression is a simple and effective model that serves as a good place to start. We saw from our EDA that the two interval variables we have a weak linear relationship with salary.

Combined with the newly encoded categorical variables that are engineered in the following cells, which linear regression can handle easily, it is worthwhile to see how Linear Regression performs as a simple model, over which we can use more advanced models or ensemble methods if the need arises


**Linear SVR**: Support Vector Machines are powerful and versatile models. Although SVMs are primarily used for classification tasks, we use the SVM Regression version (scikit-learn's LinearSVR class) which is used for regression tasks. This is also powerful as it can handle both linear and nonlinear regression.

However, we will not explore nonlinear regression with SVR. This is because the kernalised SVM model, which can be used for nonlinear regression, scales poorly with data size, unlike the LinearSVR class which scales linearly with data size

**Gradient Boosting Regressor**: Finally, I would like to use an ensemble method, and a Gradient Boosting Regressor would be an appropriate choice. Ensemble methods can combined several weak learners into a strong learner. Gradient Boosting in particular trains individual learners sequentially, with each subsequent learner being fitted on the residuals of the prior learner.

Morever, an ensemble method such as Gradient Boosting generally trades a bit more bias for less variance as well, which is useful in generalising a model to unseen data

----

In terms of new features, the first priority is to encode our categorical features 'major', 'degreeType', 'jobType' and 'industry'

Two of these features are ordinal and two are nominal. As a result, different encoding techniques will be required

Just as we created the mean salary per industry as a baseline prediction, it could be valuable to create a mean salary per job type feature

On the other hand, the data does not lend itself to any meaningful interaction variables, so these will not be explored

## Part 3 - Engineering Features & Developing Models

### Convert categorical features to numerical features: 'degree'

Let us first encode the 'degree' column type.

'Degree' can be considered a ordinal variable (the order of the data matters) more so than a nominal one (the order does not matter); clearly, a Doctoral degree is more advanced than a Master's degree, which itself is more advanced than a Bachelor's degree

As a result, it would seem the best way to convert this categorical feature into numbers is to manually encode them, where 0 may correspond to 'NONE' (no degree), 1 to 'HIGH_SCHOOl', 2 to 'BACHELORS' and so on.

We could use Label Encoding, but this option does not guarantee the order we want. For example, 'HIGH_SCHOOL' may be assigned to '3' (undesirable) and not '0' (desirable)


```python
data_combined.degree.value_counts()
```




    HIGH_SCHOOL    236975
    NONE           236853
    BACHELORS      175495
    DOCTORAL       175362
    MASTERS        175310
    Name: degree, dtype: int64




```python
data_combined['Degree Category'] = data_combined['degree']
```


```python
data_combined = data_combined.replace({'Degree Category':
                                       {'NONE':0, 'HIGH_SCHOOL':1,
                                        'BACHELORS':2, 'MASTERS':3,
                                        'DOCTORAL':4}})
```

### Convert categorical features to numerical features: 'jobType'


Similar to 'degree', 'jobType' can also be considered an ordinal variable, not a nominal one, because a CEO role has the highest possible seniority, followed by a CFO role and so on until a janitor role\*


Therefore, we perform the same transformation here as we did above

\**There is some ambiguity regarding which of CFO or CTO is the more senior rank as this often depends on company context; I will simply consider CFO to be the more senior rank since this has historically been the case*


```python
data_combined.jobType.value_counts()
```




    SENIOR            125886
    VICE_PRESIDENT    125234
    MANAGER           125120
    CTO               125045
    JANITOR           124971
    CEO               124778
    JUNIOR            124592
    CFO               124369
    Name: jobType, dtype: int64




```python
data_combined['Job Type Category'] = data_combined['jobType']

data_combined = data_combined.replace({'Job Type Category':
                                       {'JANITOR':0, 'JUNIOR':1,
                                        'SENIOR':2, 'MANAGER':3,
                                        'VICE_PRESIDENT':4, 'CTO':5,
                                        'CFO':6, 'CEO':7}})
```


```python
data_combined
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>jobId</th>
      <th>companyId</th>
      <th>jobType</th>
      <th>degree</th>
      <th>major</th>
      <th>industry</th>
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
      <th>salary</th>
      <th>Baseline Prediction</th>
      <th>Degree Category</th>
      <th>Job Type Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>JOB1362684407687</td>
      <td>COMP37</td>
      <td>CFO</td>
      <td>MASTERS</td>
      <td>MATH</td>
      <td>HEALTH</td>
      <td>10</td>
      <td>83</td>
      <td>130</td>
      <td>115.735540</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <td>1</td>
      <td>JOB1362684407688</td>
      <td>COMP19</td>
      <td>CEO</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>WEB</td>
      <td>3</td>
      <td>73</td>
      <td>101</td>
      <td>121.645362</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <td>2</td>
      <td>JOB1362684407689</td>
      <td>COMP52</td>
      <td>VICE_PRESIDENT</td>
      <td>DOCTORAL</td>
      <td>PHYSICS</td>
      <td>HEALTH</td>
      <td>10</td>
      <td>38</td>
      <td>137</td>
      <td>115.735540</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>JOB1362684407690</td>
      <td>COMP38</td>
      <td>MANAGER</td>
      <td>DOCTORAL</td>
      <td>CHEMISTRY</td>
      <td>AUTO</td>
      <td>8</td>
      <td>17</td>
      <td>142</td>
      <td>109.435222</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <td>4</td>
      <td>JOB1362684407691</td>
      <td>COMP7</td>
      <td>VICE_PRESIDENT</td>
      <td>BACHELORS</td>
      <td>PHYSICS</td>
      <td>FINANCE</td>
      <td>8</td>
      <td>16</td>
      <td>163</td>
      <td>130.747659</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>999995</td>
      <td>JOB1362685407682</td>
      <td>COMP56</td>
      <td>VICE_PRESIDENT</td>
      <td>BACHELORS</td>
      <td>CHEMISTRY</td>
      <td>HEALTH</td>
      <td>19</td>
      <td>94</td>
      <td>88</td>
      <td>115.735540</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <td>999996</td>
      <td>JOB1362685407683</td>
      <td>COMP24</td>
      <td>CTO</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>FINANCE</td>
      <td>12</td>
      <td>35</td>
      <td>160</td>
      <td>130.747659</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <td>999997</td>
      <td>JOB1362685407684</td>
      <td>COMP23</td>
      <td>JUNIOR</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>EDUCATION</td>
      <td>16</td>
      <td>81</td>
      <td>64</td>
      <td>99.448386</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>999998</td>
      <td>JOB1362685407685</td>
      <td>COMP3</td>
      <td>CFO</td>
      <td>MASTERS</td>
      <td>NONE</td>
      <td>HEALTH</td>
      <td>6</td>
      <td>5</td>
      <td>149</td>
      <td>115.735540</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <td>999999</td>
      <td>JOB1362685407686</td>
      <td>COMP59</td>
      <td>JUNIOR</td>
      <td>BACHELORS</td>
      <td>NONE</td>
      <td>EDUCATION</td>
      <td>20</td>
      <td>11</td>
      <td>88</td>
      <td>99.448386</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>999995 rows × 12 columns</p>
</div>



### Convert categorical features to numerical features: 'major'

Unlike 'degree' and 'jobType', 'major' cannot be considered to be an ordinal variable. 'Physics' cannot be said to be greater or lesser in some intuitive way than 'Engineering'. As a result, manual label encoding that maps options to different numbers (1, 2, 3...) is not the optimal approach here


Hence, it would be better to use dummy variables. One disadvantage of this, as referred to before, is that this will add many sparse columns to our dataframe. This may slow down our model training


```python
major_dummy_data = pd.get_dummies(data_combined['major'])
major_dummy_data = major_dummy_data.rename({'BIOLOGY':'Major_Biology',
                                            'BUSINESS':'Major_Business',
                                            'CHEMISTRY':'Major_Chemistry',
                                            'COMPSCI':'Major_CompSci',
                                            'ENGINEERING':'Major_Engineering',
                                            'LITERATURE':'Major_Literature',
                                            'MATH':'Major_Math',
                                            'NONE':'Major_None',
                                            'PHYSICS':'Major_Physics'},
                                             axis = 1)

major_dummy_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Major_Biology</th>
      <th>Major_Business</th>
      <th>Major_Chemistry</th>
      <th>Major_CompSci</th>
      <th>Major_Engineering</th>
      <th>Major_Literature</th>
      <th>Major_Math</th>
      <th>Major_None</th>
      <th>Major_Physics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_combined = pd.concat([data_combined, major_dummy_data], axis = 1)
```

### Convert categorical features to numerical features: 'industry'

Similar to the 'major' variable, 'industry' is also nominal; we cannot intuitively order its values. Again, we will create dummy variables


```python
industry_dummy_data = pd.get_dummies(data_combined['industry'])

industry_dummy_data = industry_dummy_data.rename({'AUTO':'Industry_Auto',
                                            'EDUCATION':'Industry_Education',
                                            'FINANCE':'Industry_Finance',
                                            'HEALTH':'Industry_Health',
                                            'OIL':'Industry_Oil',
                                            'SERVICE':'Industry_Service',
                                            'WEB':'Industry_Web'},
                                             axis = 1)



industry_dummy_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Industry_Auto</th>
      <th>Industry_Education</th>
      <th>Industry_Finance</th>
      <th>Industry_Health</th>
      <th>Industry_Oil</th>
      <th>Industry_Service</th>
      <th>Industry_Web</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_combined = pd.concat([data_combined, industry_dummy_data], axis = 1)
```


```python
data_combined.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>jobId</th>
      <th>companyId</th>
      <th>jobType</th>
      <th>degree</th>
      <th>major</th>
      <th>industry</th>
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
      <th>salary</th>
      <th>Baseline Prediction</th>
      <th>...</th>
      <th>Major_Math</th>
      <th>Major_None</th>
      <th>Major_Physics</th>
      <th>Industry_Auto</th>
      <th>Industry_Education</th>
      <th>Industry_Finance</th>
      <th>Industry_Health</th>
      <th>Industry_Oil</th>
      <th>Industry_Service</th>
      <th>Industry_Web</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>999995</td>
      <td>JOB1362685407682</td>
      <td>COMP56</td>
      <td>VICE_PRESIDENT</td>
      <td>BACHELORS</td>
      <td>CHEMISTRY</td>
      <td>HEALTH</td>
      <td>19</td>
      <td>94</td>
      <td>88</td>
      <td>115.735540</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>999996</td>
      <td>JOB1362685407683</td>
      <td>COMP24</td>
      <td>CTO</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>FINANCE</td>
      <td>12</td>
      <td>35</td>
      <td>160</td>
      <td>130.747659</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>999997</td>
      <td>JOB1362685407684</td>
      <td>COMP23</td>
      <td>JUNIOR</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>EDUCATION</td>
      <td>16</td>
      <td>81</td>
      <td>64</td>
      <td>99.448386</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>999998</td>
      <td>JOB1362685407685</td>
      <td>COMP3</td>
      <td>CFO</td>
      <td>MASTERS</td>
      <td>NONE</td>
      <td>HEALTH</td>
      <td>6</td>
      <td>5</td>
      <td>149</td>
      <td>115.735540</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>999999</td>
      <td>JOB1362685407686</td>
      <td>COMP59</td>
      <td>JUNIOR</td>
      <td>BACHELORS</td>
      <td>NONE</td>
      <td>EDUCATION</td>
      <td>20</td>
      <td>11</td>
      <td>88</td>
      <td>99.448386</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



### Create mean salary for each job type


```python
job_mean_salary = data_combined.groupby('jobType')['salary'].mean()

job_mean_salary

# intialise variables

ceo_salary = job_mean_salary[0]
cfo_salary = job_mean_salary[1]
cto_salary = job_mean_salary[2]
janitor_salary = job_mean_salary[3]
junior_salary = job_mean_salary[4]
manager_salary = job_mean_salary[5]
senior_salary = job_mean_salary[6]
vp_salary = job_mean_salary[7]

```


```python
data_combined['Mean Salary Per Job Type'] = np.NaN
```


```python
# replace each NaN with the average salary of the industry for the relevant row

data_combined['Mean Salary Per Job Type'] = np.where(
                                      (data_combined['jobType'] == 'CEO'),
                                       ceo_salary, data_combined['Mean Salary Per Job Type'])

data_combined['Mean Salary Per Job Type'] = np.where(
                                      (data_combined['jobType'] == 'CFO'),
                                       cfo_salary, data_combined['Mean Salary Per Job Type'])

data_combined['Mean Salary Per Job Type'] = np.where(
                                      (data_combined['jobType'] == 'CTO'),
                                       cto_salary, data_combined['Mean Salary Per Job Type'])

data_combined['Mean Salary Per Job Type'] = np.where(
                                      (data_combined['jobType'] == 'JANITOR'),
                                       janitor_salary, data_combined['Mean Salary Per Job Type'])

data_combined['Mean Salary Per Job Type'] = np.where(
                                      (data_combined['jobType'] == 'JUNIOR'),
                                       junior_salary, data_combined['Mean Salary Per Job Type'])

data_combined['Mean Salary Per Job Type'] = np.where(
                                      (data_combined['jobType'] == 'MANAGER'),
                                       manager_salary, data_combined['Mean Salary Per Job Type'])

data_combined['Mean Salary Per Job Type'] = np.where(
                                      (data_combined['jobType'] == 'SENIOR'),
                                       senior_salary, data_combined['Mean Salary Per Job Type'])

data_combined['Mean Salary Per Job Type'] = np.where(
                                      (data_combined['jobType'] == 'VICE_PRESIDENT'),
                                       vp_salary, data_combined['Mean Salary Per Job Type'])
```

### Checking for correlations between selected newly engineered features and 'salary'

We will investigate only the ordinal features and mean salary per job type. Including dummy variables in the correlation heatmap will lead to a congested visualisation


```python
plt.figure(figsize = (8, 6))

new_features_data = data_combined[['Degree Category', 'Job Type Category',
                     'Mean Salary Per Job Type', 'salary']]

sns.heatmap(new_features_data.corr(), annot = True, fmt = '.2f', cmap = 'mako')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f92ff38bfd0>




![png](output_76_1.png)


Thankfully, our newly created features seem to have decent predictive power, as suggested by the correlation coefficients.


These coefficients are 0.38, 0.58 and 0.6 for Degree Category, Job Type Category and Mean Salary Per Job Type, respectively

### Create and Test models


```python
# repeat step of separating features, output and
# baseline predictions from one another to include
# newly created features

X_train = data_combined
X_train = X_train.drop(['salary', 'jobId', 'companyId',
                        'jobType', 'degree', 'major',
                        'industry'], axis = 1)

y_train = data_combined['salary']
```

#### My metric will be MSE and my goal is <360


```python
# import selected models

from sklearn import linear_model
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# initialise models
lr = linear_model.LinearRegression()
svm_reg = LinearSVR(epsilon = 0.1)
tree_reg = DecisionTreeRegressor()
rf_reg = RandomForestRegressor()

# create simple function to find 5-fold cross-validation MSE score mean

def cross_val_mse(model, X, y):
    model_scores = cross_val_score(model, X, y,
                                  scoring = 'neg_mean_squared_error',
                                  cv = 5)
    print('The mean cross-validation score is {}'.format(model_scores.mean()))
```

### Linear Regression


```python
cross_val_mse(linear_model.LinearRegression(), X_train, y_train)
```

    The mean cross-validation score is -386.63858899427606


### Polynomial Regression with Degree = 2


```python
# create polynomial features with degree = 2

from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree = 2, include_bias = False)

x_polly = pr.fit_transform(X_train)

cross_val_mse(lr, x_polly, y_train)
```

### Linear SVR


```python
# scale data as SVM is sensitive to different scales

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

cross_val_mse(svm_reg, X_train_scaled, y_train)
```

    The mean cross-validation score is -387.88605390076617


### Gradient Boosting Regressor

Gradient Boosting Regressor trains each individual learner sequentially. As a result, it does not scale well with data. Let's take a smaller sample of our data to train it


```python
sample_data = data_combined.sample(n = 500000, random_state = 42)

X_train_sample_500k = sample_data.drop(['salary', 'jobId', 'companyId',
                        'jobType', 'degree', 'major',
                        'industry'], axis = 1)

y_train_sample_500k = sample_data['salary']

```


```python
from sklearn.ensemble import GradientBoostingRegressor

gb_reg = GradientBoostingRegressor()

cross_val_mse(gb_reg, X_train_sample_500k, y_train_sample_500k)
```

    The mean cross-validation score is -366.4483854053291


Now it is a good idea to perform a grid search and find the optimal hyperparameter values for our Gradient Boosting Regressor model. I chose to do this for this model only, as opposed to using grid search for Linear SVR and Decision Trees Regressor also, because of the following:

1. With no hyperparameter tuning, Gradient Boosting Regressor performs the best, compare to the other 2 models


2. Grid Search can be computationally expensive and take a long time to find solutions, so it is best performed on the most promising model, which in reference to point 1, is Gradient Boosting Regressor

To perform a grid search, I will take another, smaller sample of data to speed this process up


```python
sample_data = data_combined.sample(n = 10000, random_state = 42)

X_train_sample_10k = sample_data.drop(['salary', 'jobId', 'companyId',
                        'jobType', 'degree', 'major',
                        'industry'], axis = 1)

y_train_sample_10k = sample_data['salary']
```


```python
from sklearn.model_selection import GridSearchCV

# initialise parameters
param_grid = [ {'max_features': [5, 10, 15], 'min_samples_split': [10, 100, 1000],
               'learning_rate':[0.5, 1, 1.5], 'max_depth':[4, 8, 12]}]

gb_reg_1 = GradientBoostingRegressor(n_estimators = 100)

# initialise Grid Search
Grid_gb = GridSearchCV(gb_reg_1, param_grid, cv = 5, scoring = 'neg_mean_squared_error')

# fit
Grid_gb.fit(X_train_sample_10k, y_train_sample_10k)

# find best parameters
print(Grid_gb.best_estimator_)

#find scores
gb_grid_scores = Grid_gb.cv_results_
gb_grid_scores['mean_test_score']
```


```python
#cross validation score of tuned Gradient Boosting Regressor

gb_reg_2 = GradientBoostingRegressor(n_estimators = 160, learning_rate = 0.1,
                                     max_depth = 4, max_features = 10, min_samples_split = 1000)

cross_val_mse(gb_reg_2, X_train_sample_500k, y_train_sample_500k)
```

    The mean cross-validation score is -358.5245795133168


### Selecting the best model

The best model is polynomial regression with degree = 2. It achieved an MSE of 354, a 74.1% improvement over the baseline model MSE of 1367.12

## Part 4 - Deploy

### Automating our pipeline


```python
#write script that trains model on entire training set, saves model to disk,
#and scores the "test" dataset

def train_test_model(gb_reg_tuned, X_train, y_train, X_test, y_test):

    # apply quadratic transform to train set
    pr_train = PolynomialFeatures(degree = 2, include_bias = False)
    x_polly_train = pr_train.fit_transform(X_train)

    # initialise tuned model
    gb_reg_tuned = GradientBoostingRegressor(n_estimators = 160, learning_rate = 0.1,
                                         max_depth = 4, max_features = 10,
                                         min_samples_split = 1000)

    # fit model on X_polly_train and y_train
    gb_reg_tuned.fit(x_polly_train, y_train)

    # apply quadratic transform to test set
    pr_test = PolynomialFeatures(degree = 2, include_bias = False)
    x_polly_test = pr_test.fit_transform(X_test)

    # predict y_predicted using trained model
    y_predicted = gb_reg_tuned.predict(x_polly_test)

    # test model and print mse
    mse = metrics.mean_squared_error(y_test, y_predicted)

    print('The MSE score on the test set is {}'.format(mse))
```

### Summary of Model Performance - MSE


```python
# create dataframe
mse_performance_data = pd.DataFrame(columns = ['Model', 'MSE'])
mse_performance_data['Model'] = pd.Series(['Linear Regression',
                                           'Polynomial Regression (n = 2)',
                                           'Linear SVR', 'Decision Tree Regressor',
                                           'GB Regressor',
                                           'GB Regressor (tuned)'])
mse_performance_data['MSE'] = pd.Series([386.64, 354.13, 387.89, 689.75, 380.30,
                                         358.43])
```


```python
# plot

plt.figure(figsize = (20, 8))

plt.bar(mse_performance_data['Model'], mse_performance_data['MSE'], color = 'teal')
plt.title('5-fold MSE for 5 different models')
plt.xlabel('Model Name')
plt.ylabel('MSE')
```




    Text(0, 0.5, 'MSE')




![png](output_103_1.png)


### Save model to CSV file


```python
from pandas import to_csv

# save predictions

predicted_salary = y_predicted

# save to CSV file

predicted_salary.to_csv('/Users/User Name/Desktop/Predicted Salaries.csv')
```
