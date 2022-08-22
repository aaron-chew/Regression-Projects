# Importing standard libraries. 
import pandas as pd
import os
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

# Import transformers.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Import validation.
from sklearn.model_selection import KFold, cross_val_score, cross_validate, train_test_split, StratifiedKFold, cross_val_predict 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import feature selection.
from sklearn.feature_selection import RFECV

# Importing pickle library. 
from pickle import dump
from pickle import load


 # ============================== 3_Feature_Engineering ============================== #


def hypothesis_testing(h0, benchmark_variable, new_benchmark_name):
    # Stores the key inside a variable. 
    old_dict_key = list(benchmark_variable.keys())[0]
    
    # If our h0 scores a better error metric than h1: 
    if h0 < benchmark_variable[list(benchmark_variable.keys())[0]]:
        # Deletes the old key, and replaces it with a new key name, specified by the user.
        benchmark_variable[new_benchmark_name] = benchmark_variable.pop(old_dict_key)
        # The new h0 value is assigned to the new key name. 
        benchmark_variable[new_benchmark_name] = h0
        print("We reject the null hypothesis with the new benchmark for %s: %.4f" % (list(benchmark_variable.keys())[0],h0))
    else:
        print("We accept the null hypothesis.")
        
class model_evaluation:
    def __init__(self):
        # Placeholder for parameter grid for grid search. 
        self.params = dict()
        
    def preprocessing(self, data):
        self.numerical = list(data.select_dtypes(exclude=['object']).columns)
        self.numerical.remove("mpg") # Remove label from data.
        self.categorical = list(data.select_dtypes(['object']).columns)
        
        self.cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))])

        # Define numerical pipeline
        self.num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))])

        # Combine categorical and numerical pipelines
        self.preprocessor = ColumnTransformer([
            ('cat', self.cat_pipe, self.categorical),
            ('num', self.num_pipe, self.numerical)])
        
        self.estimators = [('preprocessor', self.preprocessor)]

    def add_pipe_component(self, string, instance):
        # Function to add pipeline steps. 
        self.estimators.append((string, instance))
        
    def cross_validation(self, data):
        # Splitting the data into features and labels. 
        self.X = data.iloc[:,:-1]
        self.y = data.iloc[:,-1] 
        
        # Defining estimator with pipeline applied steps.  
        self.pipe = Pipeline(steps=self.estimators)
        self.pipe.fit(self.X, self.y)
        # Setting up cross validation strategy. 
        self.cv = KFold(n_splits=5, random_state=42, shuffle=True)
        # Evaluate results. 
        self.results = cross_val_score(self.pipe, self.X, self.y, cv=self.cv, scoring='neg_mean_squared_error', n_jobs=-1)
        # Average out the cv array and display absloute values to remove the "neg". 
        self.cv_result = abs(self.results.mean()) 
            
    def RFE_cross_validate(self, data, model):
        self.X = data.iloc[:,:-1]
        self.y = data.iloc[:,-1] 
        
        self.cv = KFold(n_splits=5, random_state=42, shuffle=True)
        self.rfe = RFECV(estimator=model, cv=self.cv, scoring='neg_mean_squared_error', n_jobs=-1) 
        self.rfe_result = self.rfe.fit(self.X, self.y)

    def grid_search(self, X, y, model): 
        # Setting up cross validation strategy. 
        self.cv = KFold(n_splits=5, random_state=42, shuffle=True)
        # Evaluate results. 
        self.grid = GridSearchCV(estimator=model, param_grid=self.params, cv=self.cv, scoring='neg_mean_squared_error', n_jobs=-1)
        self.grid_results = self.grid.fit(X, y)
        
        print('Best: %f using %s' % (self.grid_results.best_score_, self.grid_results.best_params_))
        self.means = self.grid_results.cv_results_['mean_test_score'] 
        self.params = self.grid_results.cv_results_['params']

        #for mean, param in zip(self.means, self.params):
            #print("%f with: %r" % (mean,param))
            
    def add_params_component(self, key, value):
        # Function to add pipeline steps. 
        self.params[key] = value

    def overfitting_checker(self, X, y, model):
        # Cross validation strategy, with 5 number of splits. 
        self.cv = KFold(n_splits=5, random_state=42, shuffle=True)
        # We set "return_train_score" to True to see for overfitting. 
        self.results = cross_validate(model, X, y, cv=self.cv, scoring='neg_mean_squared_error', return_train_score=True)  
        print('Model scored an Train_MSE_Score of: %.2f and a Validation_MSE of: %.2f' % 
              (self.results["train_score"].mean(), self.results["test_score"].mean()))
        
    def overfitting_checker_no_print(self, X, y, model):
        # Cross validation strategy, with 5 number of splits. 
        self.cv = KFold(n_splits=5, random_state=42, shuffle=True)
        # We set "return_train_score" to True to see for overfitting. 
        self.results = cross_validate(model, X, y, cv=self.cv, scoring='neg_mean_squared_error', return_train_score=True)   
   

 # ============================== 4_Optimization ============================== #


def optimal_components(model, dictResults, df):
    for num in range(1, 51):
        pca = model_evaluation()
        pca.preprocessing(df)
        pca.add_pipe_component("pca", PCA(n_components=num))
        pca.add_pipe_component("clf", model)
        pca.cross_validation(df)

        dictResults[str(num)] = pca.cv_result  
        
   
 # ============================== 5_Regularization ============================== #
        

class Early_Stopping:
    def __init__(self, X, y, model):
        # Split the data into train and validation sets. 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        # Create our model instance with optimal hyperparameters. 
        self.eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        
        self.clf = model
        self.clf.fit(self.X_train, self.y_train, eval_set=self.eval_set, verbose=False)
        # Make predictions for test data.
        self.y_pred = self.clf.predict(self.X_test)

        # Evaluate predictions.
        self.mae = mean_absolute_error(self.y_test, self.y_pred)

        # Retrieve performance metrics.
        self.results = self.clf.evals_result() # Log loss scores. 
        self.epochs = len(self.results['validation_0']['mae']) # no. of training epochs/no. of trees. 
        self.x_axis = range(0, self.epochs) # x-axis (no. of trees).
        
def LogLoss_Curve(x_axis, y_axis_train, y_axis_test):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(x_axis), y=y_axis_train,
                        name='Train'))
    fig.add_trace(go.Scatter(x=list(x_axis), y=y_axis_test,
                        name='Test'))

    fig.update_layout(title='XGB Regressor Loss Curve',
                       xaxis_title='Epochs',
                       yaxis_title='MAE',
                       height=455, width=900)
    fig.show() 
    
    
 # ============================== 6_Final_Pipeline ============================== #


def holdout_set_evaluation(y_axis, train_data, holdout_features, holdout_labels, model, model_string):
    numerical = list(train_data.select_dtypes(exclude=['object']).columns)
    categorical = list(train_data.select_dtypes(['object']).columns)
    numerical.remove("mpg") # Remove label from data.
    
    for i in range(1, 31):
        cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))])

        # Define numerical pipeline
        num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))])

        # Combine categorical and numerical pipelines
        preprocessor = ColumnTransformer([
            ('cat', cat_pipe, categorical),
            ('num', num_pipe, numerical)])

        estimators = [('preprocessor', preprocessor), ("clf", model)]

        # Split the train dataset.
        trainX = train_data.iloc[:,:-1]
        trainy = train_data.iloc[:,-1]

        pipe = Pipeline(steps=estimators)
        pipe = pipe.fit(trainX, trainy) 

        yhat = pipe.predict(holdout_features)
        mse =  abs(mean_squared_error(holdout_labels, yhat))
        y_axis.append(mse)
        
def MSE_plot(MSE_scores_GBR, MSE_scores_RFR, MSE_scores_XGB, score_string):
    fig = make_subplots(rows=3, cols=1)

    fig.append_trace(go.Scatter(
        x=list(range(1,31)),
        y=MSE_scores_GBR,
        name="Gradient Boosting Regressor"
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=list(range(1,31)),
        y=MSE_scores_RFR,
        name="Random Forest Regressor"
    ), row=2, col=1)

    fig.append_trace(go.Scatter(
        x=list(range(1,31)),
        y=MSE_scores_XGB,
        name="XGB Regressor",
    ), row=3, col=1)


    fig.update_layout(title_text="Model Performance over 30 Iterations", width=900)

    fig['layout']['yaxis']['title']=score_string
    fig['layout']['xaxis3']['title']='Iterations'

    fig.show()
    
def saving_model(train_data, model):
    numerical = list(train_data.select_dtypes(exclude=['object']).columns)
    numerical.remove("mpg") # Remove label from data.
    categorical = list(train_data.select_dtypes(['object']).columns)

    # Define categorical pipeline.
    cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))])

    # Define numerical pipeline.
    num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))])

    # Combine categorical and numerical pipelines.
    preprocessor = ColumnTransformer([
        ('cat', cat_pipe, categorical),
        ('num', num_pipe, numerical)])

    estimators = [('preprocessor', preprocessor), ("clf", model)]
    
    # Split the train dataset.
    trainX = train_data.iloc[:,:-1]
    trainy = train_data.iloc[:,-1]

    pipe = Pipeline(steps=estimators)
    pipe = pipe.fit(trainX, trainy) 
    
    dump(pipe, open("XGB_Regressor.pkl", "wb"))

        