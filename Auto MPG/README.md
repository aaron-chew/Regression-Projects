
# Auto MPG

This project aims to train an algorithm to accurately predict the miles per gallon of an automobile given X amount of features. Inside this repository you'll find a range of statistical technqiues and visualizations to give you insight on the problem at hand and appropiate applied solutions that we'll explore.

## Background
This dataset is a slightly modified version of the dataset provided in
the StatLib library. In line with the use by Ross Quinlan in
predicting the attribute "mpg", 8 of the original instances were removed
because they had unknown values for the "mpg" attribute. The original
dataset is available in the file "auto-mpg.data-original".

## The Dataset
The dataset is a multivariate regression problem that contains 8 attributes, 398 rows and categorical and real numbers. As such, we will have to manually split the data into train and test sets, to later test our model on unseen data.

## Project Layout
This project uses 6 notebooks that are divided by each stage/topic of our analysis. To begin, start at notebook "01_basic_exploration" then work your way up sequentially until you reach the 6th and final notebook that deploys the model. The functions and plots files can be found under the "functions" folder which contains all the functions used inside our notebooks.

## Features
1. mpg: continuous
2. cylinders: multi-valued discrete
3. displacement: continuous
4. horsepower: continuous
5. weight: continuous
6. acceleration: continuous
7. model year: multi-valued discrete
8. origin: multi-valued discrete
9. car name: string (unique for each instance)

## Model Deployment
From the conclusion and findings from our project, we found out that the XGB Regressor was the best performing model that achieved the lowest loss function from our sample of algorithms. You can find the final pipeline with all the necessary transformers and model inside the pickle file "XGBRegressor.pkl".


## Authors

- [@aaron-chew](https://github.com/aaron-chew)

