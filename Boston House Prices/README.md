
# Boston House Prices

This project aims to train an algorithm to accurately predict the price of a Boston House given X amount of features. Inside this repository you'll find a range of statistical technqiues and visualizations to give you insight on the problem at hand and appropiate applied solutions that we'll explore. 

## Background
This dataset contains information collected by the U.S Census Service concerning housing in the area of Boston Mass. It was obtained from the StatLib archive (http://lib.stat.cmu.edu/datasets/boston), and has been used extensively throughout the literature to benchmark algorithms. The dataset is small in size with only 506 cases.

## The Dataset
The dataset is a multivariate regression problem that contains 14 attributes, 506 rows and a all numerical data type. As such, we will have to manually split the data into train and test sets, to later test our model on unseen data.

## Project Layout 
This project uses 6 notebooks that are divided by each stage/topic of our analysis. To begin, start at notebook "01_basic_exploration" then work your way up sequentially until you reach the 6th and final notebook that deploys the model. The functions and plots files can be found under the "functions" folder which contains all the functions used inside our notebooks.

## Features
* ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
* INDUS: proportion of non-retail business acres per town
* CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
* NOX: nitric oxides concentration (parts per 10 million)
* RM: average number of rooms per dwelling
* AGE: proportion of owner-occupied units built prior to 1940
* DIS: weighted distances to ﬁve Boston employment centers
* RAD: index of accessibility to radial highways
* TAX: full-value property-tax rate per $10,000
* PTRATIO: pupil-teacher ratio by town 12. B: 1000(Bk−0.63)2 where Bk is the proportion of blacks by town 13. LSTAT: % lower status of the population
* MEDV: Median value of owner-occupied homes in $1000s

## Model Deployment
From the conclusion and findings from our project, we found out that the Random Forest Regressor was the best performing model that achieved the lowest loss function from our sample of algorithms. You can find the final pipeline with all the necessary transformers and model inside the pickle file "RandomForestRegressor.pkl".


## Authors

- [@aaron-chew](https://github.com/aaron-chew)

