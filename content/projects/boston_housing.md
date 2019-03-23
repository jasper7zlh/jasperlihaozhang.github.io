+++
categories = ["MovieLens EDA"]
comments = false
date = "2018-03-18T22:28:13-04:00"
draft = false
showpagemeta = true
showcomments = true
slug = ""
tags = ["boston-housing-price-prediction", "grid-search", "hyperparameter-tuning", "cross-validation", "eda", "complexity-curve"]
title = "Boston Housing"
description = "Model Evaluation and Validation"

+++

## Model Evaluation & Validation
## Project: Predicting Boston Housing Prices


## Getting Started
In this project, you will evaluate the performance and predictive power of a model that has been trained and tested on data collected from homes in suburbs of Boston, Massachusetts. A model trained on this data that is seen as a *good fit* could then be used to make certain predictions about a home — in particular, its monetary value. This model would prove to be invaluable for someone like a real estate agent who could make use of such information on a daily basis.

The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing). The Boston housing data was collected in 1978 and each of the 506 entries represent aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts. For the purposes of this project, the following preprocessing steps have been made to the dataset:
- 16 data points have an `'MEDV'` value of 50.0. These data points likely contain **missing or censored values** and have been removed.
- 1 data point has an `'RM'` value of 8.78. This data point can be considered an **outlier** and has been removed.
- The features `'RM'`, `'LSTAT'`, `'PTRATIO'`, and `'MEDV'` are essential. The remaining **non-relevant features** have been excluded.
- The feature `'MEDV'` has been **multiplicatively scaled** to account for 35 years of market inflation.

Run the code cell below to load the Boston housing dataset, along with a few of the necessary Python libraries required for this project. You will know the dataset loaded successfully if the size of the dataset is reported.


```python
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
%matplotlib inline

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))
```

    Boston housing dataset has 489 data points with 4 variables each.


    /opt/conda/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    /opt/conda/lib/python3.6/site-packages/sklearn/learning_curve.py:22: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the functions are moved. This module will be removed in 0.20
      DeprecationWarning)


## Data Exploration
In this first section of this project, you will make a cursory investigation about the Boston housing data and provide your observations. Familiarizing yourself with the data through an explorative process is a fundamental practice to help you better understand and justify your results.

Since the main goal of this project is to construct a working model which has the capability of predicting the value of houses, we will need to separate the dataset into **features** and the **target variable**. The **features**, `'RM'`, `'LSTAT'`, and `'PTRATIO'`, give us quantitative information about each data point. The **target variable**, `'MEDV'`, will be the variable we seek to predict. These are stored in `features` and `prices`, respectively.

### Implementation: Calculate Statistics
For your very first coding implementation, you will calculate descriptive statistics about the Boston housing prices. Since `numpy` has already been imported for you, use this library to perform the necessary calculations. These statistics will be extremely important later on to analyze various prediction results from the constructed model.

In the code cell below, you will need to implement the following:
- Calculate the minimum, maximum, mean, median, and standard deviation of `'MEDV'`, which is stored in `prices`.
  - Store each calculation in their respective variable.


```python
# TODO: Minimum price of the data
minimum_price = np.min(prices)

# TODO: Maximum price of the data
maximum_price = np.max(prices)

# TODO: Mean price of the data
mean_price = np.mean(prices)

# TODO: Median price of the data
median_price = np.median(prices)

# TODO: Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${:,.2f}".format(minimum_price)) 
print("Maximum price: ${:,.2f}".format(maximum_price))
print("Mean price: ${:,.2f}".format(mean_price))
print("Median price ${:,.2f}".format(median_price))
print("Standard deviation of prices: ${:,.2f}".format(std_price))
```

    Statistics for Boston housing dataset:
    
    Minimum price: $105,000.00
    Maximum price: $1,024,800.00
    Mean price: $454,342.94
    Median price $438,900.00
    Standard deviation of prices: $165,171.13


### Question 1 - Feature Observation
As a reminder, we are using three features from the Boston housing dataset: `'RM'`, `'LSTAT'`, and `'PTRATIO'`. For each data point (neighborhood):
- `'RM'` is the average number of rooms among homes in the neighborhood.
- `'LSTAT'` is the percentage of homeowners in the neighborhood considered "lower class" (working poor).
- `'PTRATIO'` is the ratio of students to teachers in primary and secondary schools in the neighborhood.


** Using your intuition, for each of the three features above, do you think that an increase in the value of that feature would lead to an **increase** in the value of `'MEDV'` or a **decrease** in the value of `'MEDV'`? Justify your answer for each.**

**Hint:** This problem can phrased using examples like below.  
* Would you expect a home that has an `'RM'` value(number of rooms) of 6 be worth more or less than a home that has an `'RM'` value of 7?
* Would you expect a neighborhood that has an `'LSTAT'` value(percent of lower class workers) of 15 have home prices be worth more or less than a neighborhood that has an `'LSTAT'` value of 20?
* Would you expect a neighborhood that has an `'PTRATIO'` value(ratio of students to teachers) of 10 have home prices be worth more or less than a neighborhood that has an `'PTRATIO'` value of 15?

**Answer: **

`'RM'`
- Increase in the value of RM would lead to an increase in the value of MEDV
- Because a higher number of rooms takes more space, which usually would make the home price higher given other factors remaining the same.

`'LSTAT'`
- Increase in the value of LSTAT would lead to an decrease in the value of MEDV
- Since the more the lower class workers, the more affordable the homes are in the neighborhood, meaning the home price is expected to be lower.

`'PTRATIO'`
- Increase in the value of PTRATIO would lead to an decrease in the value of MEDV
- Since a higher PTRATIO means in average, a teacher is pairing with more students, it could reflect that the educational resources are becoming fewer. Typically, it's the scenario in less-rich area compared to rich area, so the corresponding home price in such neighborhood is expected to decrease accordingly.

`Note:` Both the scatter plot and the correlation matrix below would visually support that the relationship between MEDV and RM is possitive, the relationship between MEDV and LSTAT is negative, and the relationship between MEDV and PTRATIO is also negative.

**Scatter Plot**


```python
# Use pyplot and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.figure(figsize = (18, 5))

for i, col in enumerate(features.columns):
    # Plots for 3 features
    plt.subplot(1, 3, i+1)
    x = features[col]
    y = prices
    plt.plot(x, y, 'o')
    
    # Overlay regression line
    a, b = np.polyfit(x, y, 1)
    plt.plot(x, a*x + b, '-')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('Prices')
```


![png](img/output_9_0.png)


**Correlation Matrix**


```python
import numpy as np
import pandas as pd

# Put MEDV as the first column to better visualize correlation matrix plot
data_c = pd.concat([pd.DataFrame(prices), features], axis = 1)
corr = data_c.corr()

# Plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, vmin = -1, vmax = 1)
fig.colorbar(cax)
ticks = np.arange(0, 4, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(data_c.columns.values)
ax.set_yticklabels(data_c.columns.values)
ax.set_title('Correlation Matrix between Prices and Features', y = 1.08)
plt.show()
```


![png](img/output_11_0.png)


----

## Developing a Model
In this second section of the project, you will develop the tools and techniques necessary for a model to make a prediction. Being able to make accurate evaluations of each model's performance through the use of these tools and techniques helps to greatly reinforce the confidence in your predictions.

### Implementation: Define a Performance Metric
It is difficult to measure the quality of a given model without quantifying its performance over training and testing. This is typically done using some type of performance metric, whether it is through calculating some type of error, the goodness of fit, or some other useful measurement. For this project, you will be calculating the [*coefficient of determination*](http://stattrek.com/statistics/dictionary.aspx?definition=coefficient_of_determination), R<sup>2</sup>, to quantify your model's performance. The coefficient of determination for a model is a useful statistic in regression analysis, as it often describes how "good" that model is at making predictions. 

The values for R<sup>2</sup> range from 0 to 1, which captures the percentage of squared correlation between the predicted and actual values of the **target variable**. A model with an R<sup>2</sup> of 0 is no better than a model that always predicts the *mean* of the target variable, whereas a model with an R<sup>2</sup> of 1 perfectly predicts the target variable. Any value between 0 and 1 indicates what percentage of the target variable, using this model, can be explained by the **features**. _A model can be given a negative R<sup>2</sup> as well, which indicates that the model is **arbitrarily worse** than one that always predicts the mean of the target variable._

For the `performance_metric` function in the code cell below, you will need to implement the following:
- Use `r2_score` from `sklearn.metrics` to perform a performance calculation between `y_true` and `y_predict`.
- Assign the performance score to the `score` variable.


```python
# TODO: Import 'r2_score'
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)
    
    # Return the score
    return score
```

### Question 2 - Goodness of Fit
Assume that a dataset contains five data points and a model made the following predictions for the target variable:

| True Value | Prediction |
| :-------------: | :--------: |
| 3.0 | 2.5 |
| -0.5 | 0.0 |
| 2.0 | 2.1 |
| 7.0 | 7.8 |
| 4.2 | 5.3 |

Run the code cell below to use the `performance_metric` function and calculate this model's coefficient of determination.


```python
# Calculate the performance of this model
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))
```

    Model has a coefficient of determination, R^2, of 0.923.


* Would you consider this model to have successfully captured the variation of the target variable? 
* Why or why not?

** Hint: **  The R2 score is the proportion of the variance in the dependent variable that is predictable from the independent variable. In other words:
* R2 score of 0 means that the dependent variable cannot be predicted from the independent variable.
* R2 score of 1 means the dependent variable can be predicted from the independent variable.
* R2 score between 0 and 1 indicates the extent to which the dependent variable is predictable. An 
* R2 score of 0.40 means that 40 percent of the variance in Y is predictable from X.

**Answer:**

- This model have relatively successfully captured the variation of the target variable in the 5 observations
- R<sup>2</sup> we got is 0.923, seems pretty good and very close to 1, meaning 92.3% of the variance in the dependent variable is predictable from the independent variable. However, the score is based on 5 obervations which the sample size is very small, so it is hard to draw conclusions that it is statistically significant.

### Implementation: Shuffle and Split Data
Your next implementation requires that you take the Boston housing dataset and split the data into training and testing subsets. Typically, the data is also shuffled into a random order when creating the training and testing subsets to remove any bias in the ordering of the dataset.

For the code cell below, you will need to implement the following:
- Use `train_test_split` from `sklearn.cross_validation` to shuffle and split the `features` and `prices` data into training and testing sets.
  - Split the data into 80% training and 20% testing.
  - Set the `random_state` for `train_test_split` to a value of your choice. This ensures results are consistent.
- Assign the train and testing splits to `X_train`, `X_test`, `y_train`, and `y_test`.


```python
# TODO: Import 'train_test_split'
from sklearn.cross_validation import train_test_split
# TODO: Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size = 0.2, random_state = 23)

# Success
print("Training and testing split was successful.")
```

    Training and testing split was successful.


### Question 3 - Training and Testing

* What is the benefit to splitting a dataset into some ratio of training and testing subsets for a learning algorithm?

**Hint:** Think about how overfitting or underfitting is contingent upon how splits on data is done.

**Answer: **

- A good learning algorithm should be good at both **prediction** and **generalization**. If the prediction accuracy based on the given dataset is low, then it will lead to underfitting, that means such learning algorithm's prediction capability is not good; if the prediction accuracy based on the given dataset is very high, but the model's prediction accuracy for a new dataset is low, meaning the prediction capability cannot generalize well, and this leads to overfitting.
- The benefit to splitting a dataset into training and testing will help a learning algorithm to improve the overall prediction power based on the trade-off of prediction capability and generalization capability. We use the training set to train the model to have a high prediction accuracy to avoid underfitting, and use the testing set to make sure it can also generalize well to avoid overfitting.

----

## Analyzing Model Performance
In this third section of the project, you'll take a look at several models' learning and testing performances on various subsets of training data. Additionally, you'll investigate one particular algorithm with an increasing `'max_depth'` parameter on the full training set to observe how model complexity affects performance. Graphing your model's performance based on varying criteria can be beneficial in the analysis process, such as visualizing behavior that may not have been apparent from the results alone.

### Learning Curves
The following code cell produces four graphs for a decision tree model with different maximum depths. Each graph visualizes the learning curves of the model for both training and testing as the size of the training set is increased. Note that the shaded region of a learning curve denotes the uncertainty of that curve (measured as the standard deviation). The model is scored on both the training and testing sets using R<sup>2</sup>, the coefficient of determination.  

Run the code cell below and use these graphs to answer the following question.


```python
# Produce learning curves for varying training set sizes and maximum depths
vs.ModelLearning(features, prices)
```


![png](img/output_25_0.png)


### Question 4 - Learning the Data
* Choose one of the graphs above and state the maximum depth for the model. 
* What happens to the score of the training curve as more training points are added? What about the testing curve? 
* Would having more training points benefit the model? 

**Hint:** Are the learning curves converging to particular scores? Generally speaking, the more data you have, the better. But if your training and testing curves are converging with a score above your benchmark threshold, would this be necessary?
Think about the pros and cons of adding more training points based on if the training and testing curves are converging.

**Answer: **

- max_depth = 10, a high variance scenario
- Score of the training curve remains around 1 and barely decreases as more training points are added, it is overfitting; score of testing curve firstly increases to 0.6 and then it is converging to around 0.7, not a very high score, meaning the model does not generalize well
- Having more training points will not benefit the model as it is already overfitting the dataset, and the testing curve already converges to a not-high score 0.7, in such case, adding more training points will mostly make computations more intensive without improving much of the testing score or training score. For such scenario, we need to lower the model complexity to avoid overfitting, and improve the model's generalization capability.
- When max_depth = 1, it's a high bias scenario; when max_depth = 6, it's a slightly high variance scenario; when max_depth = 3, it's a relatively ideal scenario.

### Complexity Curves
The following code cell produces a graph for a decision tree model that has been trained and validated on the training data using different maximum depths. The graph produces two complexity curves — one for training and one for validation. Similar to the **learning curves**, the shaded regions of both the complexity curves denote the uncertainty in those curves, and the model is scored on both the training and validation sets using the `performance_metric` function.  

** Run the code cell below and use this graph to answer the following two questions Q5 and Q6. **


```python
vs.ModelComplexity(X_train, y_train)
```


![png](img/output_29_0.png)


### Question 5 - Bias-Variance Tradeoff
* When the model is trained with a maximum depth of 1, does the model suffer from high bias or from high variance? 
* How about when the model is trained with a maximum depth of 10? What visual cues in the graph justify your conclusions?

**Hint:** High bias is a sign of underfitting(model is not complex enough to pick up the nuances in the data) and high variance is a sign of overfitting(model is by-hearting the data and cannot generalize well). Think about which model(depth 1 or 10) aligns with which part of the tradeoff.

**Answer: **

- When the model is trained with a maximum depth of 1, the model suffers from high bias since both the training score (< 0.5) and validation score (< 0.4) are low in terms of accuracy, and the gap between training score and validation score is small. So, the model is not complex enough to pick up the nuances in the data, it is suffering from high bias.
- When the model is trained with a maximum depth of 10, the model suffers from high variance since the training score is almost 1, nearly perfect, while the validation score is around 0.7, not a high score, and the gap between training score and validation score is big. The model complexity is too high, it is overfitting and cannot generalize well, so, it suffers from high variance.

### Question 6 - Best-Guess Optimal Model
* Which maximum depth do you think results in a model that best generalizes to unseen data? 
* What intuition lead you to this answer?

** Hint: ** Look at the graph above Question 5 and see where the validation scores lie for the various depths that have been assigned to the model. Does it get better with increased depth? At what point do we get our best validation score without overcomplicating our model? And remember, Occams Razor states "Among competing hypotheses, the one with the fewest assumptions should be selected."

**Answer: **

- The maximum depth is 4
- The increase speed of training score after maximum depth of 4 starts to slow, indicating that maximum depth of 4 has an optimal training score for the model's ability to generalize to unseen data, and it is not suffer from underfitting the dataset.
- Validation score at maximum depth of 4 starts to hit its peak with around 0.8, the score does not increase much afterwards, and the gap between training score and validation score is relatively small, meaning the model does not suffer from overfitting the dataset.

-----

## Evaluating Model Performance
In this final section of the project, you will construct a model and make a prediction on the client's feature set using an optimized model from `fit_model`.

### Question 7 - Grid Search
* What is the grid search technique?
* How it can be applied to optimize a learning algorithm?

** Hint: ** When explaining the Grid Search technique, be sure to touch upon why it is used,  what the 'grid' entails and what the end goal of this method is. To solidify your answer, you can also give an example of a parameter in a model that can be optimized using this approach.

**Answer: **
- Grid search is a traditional way of performing hyperparameter optimization, which is an exhaustive searching through a manually specified subset of the hyperparameter space of a learning algorithm. The manually specified subset of the hyperparameter space is usually determined by heuristics.
- A grid search algorithm must be guided by some performance metric, typically measured by cross-validation on the training set or evaluation on a held-out validation set. Cross validations is not always needed for grid search.
- For example, we could have a grid with the following values for a parameter pair (gamma, C): (0.1, 1), (0.1, 10), (0.1, 100), (1, 1), (1, 10) and (1, 100) for a SVM model. It's a grid because it's similar to a product of [0.1, 1] for gamma and [1, 10, 100] for C. In this example, grid search would basically train a SVM model for each of these six pairs of (gamma, C) values, then we can evaluate it using cross-validation, and select the pair of (gamma, C) that makes the SVM model best fit the data.

### Question 8 - Cross-Validation

* What is the k-fold cross-validation training technique? 

* What benefit does this technique provide for grid search when optimizing a model?

**Hint:** When explaining the k-fold cross validation technique, be sure to touch upon what 'k' is, how the dataset is split into different parts for training and testing and the number of times it is run based on the 'k' value.

When thinking about how k-fold cross validation helps grid search, think about the main drawbacks of grid search which are hinged upon **using a particular subset of data for training or testing** and how k-fold cv could help alleviate that. You can refer to the [docs](http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation) for your answer.

**Answer: **
- K-fold cross-validation training technique
    - K-fold cross-validation training technique partitions the original training data set into K equal subsets, i.e. K folds.
    - Each fold acts as the validation set exactly once, and acts as the cross validation training set K-1 times.
    - Each time, train the machine learning model using the cross validation training set and calculate the accuracy of the model by validating the predicted results against the validation set.
    - Estimate the accuracy of the machine learning model by **averaging** the accuracies derived in all the K cases of cross validation.
    
- Benefit of CV for grid search
    - Cross validation can help grid search find the best combination of hyperparameters of the learning algorithm that fits the data the most.
    - It can reduce variance when optimizing a model simply based on a train/test dataset split

### Implementation: Fitting a Model
Your final implementation requires that you bring everything together and train a model using the **decision tree algorithm**. To ensure that you are producing an optimized model, you will train the model using the grid search technique to optimize the `'max_depth'` parameter for the decision tree. The `'max_depth'` parameter can be thought of as how many questions the decision tree algorithm is allowed to ask about the data before making a prediction. Decision trees are part of a class of algorithms called *supervised learning algorithms*.

In addition, you will find your implementation is using `ShuffleSplit()` for an alternative form of cross-validation (see the `'cv_sets'` variable). While it is not the K-Fold cross-validation technique you describe in **Question 8**, this type of cross-validation technique is just as useful!. The `ShuffleSplit()` implementation below will create 10 (`'n_splits'`) shuffled sets, and for each shuffle, 20% (`'test_size'`) of the data will be used as the *validation set*. While you're working on your implementation, think about the contrasts and similarities it has to the K-fold cross-validation technique.

Please note that ShuffleSplit has different parameters in scikit-learn versions 0.17 and 0.18.
For the `fit_model` function in the code cell below, you will need to implement the following:
- Use [`DecisionTreeRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) from `sklearn.tree` to create a decision tree regressor object.
  - Assign this object to the `'regressor'` variable.
- Create a dictionary for `'max_depth'` with the values from 1 to 10, and assign this to the `'params'` variable.
- Use [`make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html) from `sklearn.metrics` to create a scoring function object.
  - Pass the `performance_metric` function as a parameter to the object.
  - Assign this scoring function to the `'scoring_fnc'` variable.
- Use [`GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) from `sklearn.grid_search` to create a grid search object.
  - Pass the variables `'regressor'`, `'params'`, `'scoring_fnc'`, and `'cv_sets'` as parameters to the object. 
  - Assign the `GridSearchCV` object to the `'grid'` variable.


```python
# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': list(range(1, 11))}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search cv object --> GridSearchCV()
    # Make sure to include the right parameters in the object:
    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
    grid = GridSearchCV(regressor, params, scoring_fnc, cv = cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_
```

    /opt/conda/lib/python3.6/site-packages/sklearn/grid_search.py:42: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
      DeprecationWarning)


### Making Predictions
Once a model has been trained on a given set of data, it can now be used to make predictions on new sets of input data. In the case of a *decision tree regressor*, the model has learned *what the best questions to ask about the input data are*, and can respond with a prediction for the **target variable**. You can use these predictions to gain information about data where the value of the target variable is unknown — such as data the model was not trained on.

### Question 9 - Optimal Model

* What maximum depth does the optimal model have? How does this result compare to your guess in **Question 6**?  

Run the code block below to fit the decision tree regressor to the training data and produce an optimal model.


```python
# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))
```

    Parameter 'max_depth' is 5 for the optimal model.



```python
# Get all parameters for the best model
reg.get_params()
```




    {'criterion': 'mse',
     'max_depth': 5,
     'max_features': None,
     'max_leaf_nodes': None,
     'min_impurity_decrease': 0.0,
     'min_impurity_split': None,
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'min_weight_fraction_leaf': 0.0,
     'presort': False,
     'random_state': None,
     'splitter': 'best'}



** Hint: ** The answer comes from the output of the code snipped above.

**Answer: **

- The optimal model has a maximum depth of 5
- The result has 1 depth deeper than my guess in Question 6
- It does not suprise me that much, because it is why we need to use grid search and cross-validation or similiar validation process to make our choise of hyperparameters.
- Also, I found that the optimal max_depth is not always 5 when I run the model multiple times, I think it's because we did not assign a random state when creating the decision tree regressor.

### Question 10 - Predicting Selling Prices
Imagine that you were a real estate agent in the Boston area looking to use this model to help price homes owned by your clients that they wish to sell. You have collected the following information from three of your clients:

| Feature | Client 1 | Client 2 | Client 3 |
| :---: | :---: | :---: | :---: |
| Total number of rooms in home | 5 rooms | 4 rooms | 8 rooms |
| Neighborhood poverty level (as %) | 17% | 32% | 3% |
| Student-teacher ratio of nearby schools | 15-to-1 | 22-to-1 | 12-to-1 |

* What price would you recommend each client sell his/her home at? 
* Do these prices seem reasonable given the values for the respective features? 

**Hint:** Use the statistics you calculated in the **Data Exploration** section to help justify your response.  Of the three clients, client 3 has has the biggest house, in the best public school neighborhood with the lowest poverty level; while client 2 has the smallest house, in a neighborhood with a relatively high poverty rate and not the best public schools.

Run the code block below to have your optimized model make predictions for each client's home.


```python
# Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))
```

    Predicted selling price for Client 1's home: $421,095.65
    Predicted selling price for Client 2's home: $230,522.73
    Predicted selling price for Client 3's home: $964,162.50


**Answer: **

- Recommended home price
    - Client 1: \$ 421099
    - Client 2: \$ 230499
    - Client 3: \$ 964199
- Data Exploration Summary
    - Minimum price: \$ 105000.0
    - Maximum price: \$ 1024800.0
    - Mean price: \$ 454342.9
    - Median price \$ 438900.0
    - Standard deviation of prices: \$ 165171.1
- Price Reasonableness
    - Prices in the results are firstly rounded to the nearest hundreds since values in the MEDV field in the raw data are rounded to the nearest hundreds, then I subtract 1 for each of the three as it's a common and useful selling/pricing strategy.
    - Predicted selling price for both **Client 1's and Client 2's homes is below (Client 1's is slightly below) the mean price and median price** in the dataset, and predicted selling price for **Client 3 is far above both the mean and median price in the dataset, and it's close to the maximum price value in the dataset**:
        * For Client 1: The predicted selling price is in the middle of the 3, and it is reasonable becuase his/her home's average number of rooms, neighborhood poverty level and student-teacher ratio
        * For Client 2: The predicted selling price is the lowest of the 3, and it is reasonable since his/her home has the least number of rooms, highest neighborhood poverty level and highest student-teacher ratio
        * For Client 3: The predicted selling price is the highest among the 3, and it is reasonable because his/her home has the most number of rooms, very low neighborhood poverty level and the lowest student-teacher ratio

### Sensitivity
An optimal model is not necessarily a robust model. Sometimes, a model is either too complex or too simple to sufficiently generalize to new data. Sometimes, a model could use a learning algorithm that is not appropriate for the structure of the data given. Other times, the data itself could be too noisy or contain too few samples to allow a model to adequately capture the target variable — i.e., the model is underfitted. 

**Run the code cell below to run the `fit_model` function ten times with different training and testing sets to see how the prediction for a specific client changes with respect to the data it's trained on.**


```python
vs.PredictTrials(features, prices, fit_model, client_data)
```

    Trial 1: $391,183.33
    Trial 2: $419,700.00
    Trial 3: $415,800.00
    Trial 4: $420,622.22
    Trial 5: $418,377.27
    Trial 6: $411,931.58
    Trial 7: $399,663.16
    Trial 8: $407,232.00
    Trial 9: $351,577.61
    Trial 10: $413,700.00
    
    Range in prices: $69,044.61


### Question 11 - Applicability

* In a few sentences, discuss whether the constructed model should or should not be used in a real-world setting.  

**Hint:** Take a look at the range in prices as calculated in the code snippet above. Some questions to answering:
- How relevant today is data that was collected from 1978? How important is inflation?
- Are the features present in the data sufficient to describe a home? Do you think factors like quality of apppliances in the home, square feet of the plot area, presence of pool or not etc should factor in?
- Is the model robust enough to make consistent predictions?
- Would data collected in an urban city like Boston be applicable in a rural city?
- Is it fair to judge the price of an individual home based on the characteristics of the entire neighborhood?

**Answer: **

- The range in predicted prices is 16.4% of the maximum value and 19.6% of the minimum value in the 10 trials, that means this model is not that robust when predicting a specific client's house price using different training and testing datasets from the same original data, so the model is not robust enough to make consistent predictions.
- Leaning algorithms learned based on data collected 40 years ago is hard to generalize to new data nowadays in terms of prediction accuracy, relations between number of rooms, neighborhood poverty level, student-teacher ratio and home prices may have changed a lot during the 40 years. Inflation is an important factor, if we did not deal with inflation in the data pre-processing section, our predicted house prices will be much lower than they should be today as prices have been inflated a lot since 40 years ago. Luckily, the MEDV in the dataset already accounted for 35 years' market inflation.
- The three features used are not sufficient enough to describe a home, additionally, crime rate, neighborhood public school rating, neighborhood private school rating, public transportation accessibility, quality of apppliances in the home, square feet of the plot area, presence of pool or not etc should also factor in.
- Data collected in an urban city like Boston may not be applicable in a rural city since the features of a home may change a lot in rural areas.
- It is not that fair to judge the price of an individual home based on the characteristics of the entire neighborhood, since within neighborhood, features of each home may vary a lot. So we may need to account for within-neighborhood variations when predicting the price for individual homes.

