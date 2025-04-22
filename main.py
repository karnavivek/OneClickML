import os
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import time

start = time.time()

class ModelTraining:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def model_selection(result): #append all the MSE results from each model & make a list -> Select the least MSE's model & move it forward for model embeddings
        results = []
        results.append(result)
        return min(results)

    def results(self, y_test, y_pred, binary=False):
        if binary == False:
            mse = mean_squared_error(y_test, y_pred)
            return mse
        else:
            roc_auc = roc_auc_score(y_test, y_pred)
        return roc_auc
    
    def grid_searchcv(self, model, X_train, y_train, params_list, cv=5):
        scores = []
        gs = GridSearchCV(estimator=model, param_grid=params_list, cv=cv, return_train_score=False)
        gs.fit(X_train, y_train)
        scores.append(
                {'model': model,
                'best_score': gs.best_score_,
                'best_params': gs.best_params_
                }
            )
        scores = pd.DataFrame(scores)
        # print(f'Your {model} has been Trained Successfully!')
        return scores

    def linear(self, X_train, y_train, X_test=None, y_test=None, binary=False):

        if X_test is None or y_test is None:
            raise ValueError("X_test & y_test must be provided for evaluation")
            
        if binary == False:
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("\nYour Linear Regression model has been trained!")
            return y_pred
        else:
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("\nYour Logistic Regression model has been trained!")
            return y_pred
    

    def DT(self, X_train, y_train, X_test=None, y_test=None, binary=False):

        if X_test is None or y_test is None:
            raise ValueError("X_test & y_test must be provided for evaluation")
        
        if binary is False:
            model = DecisionTreeRegressor() #make sure to add params settings
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("\nYour Decision Tree Regressor model has been trained!")
            return y_pred
        else:
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("\nYour Decision Tree Classifier model has been trained!")
            return y_pred

    def MLP(self, X_train, y_train, X_test=None, y_test=None, binary=False):
        
        if X_test is None or y_test is None:
            raise ValueError("X_test & y_test must be provided for evaluation")
        
        if binary is False:
            model = MLPRegressor() #make sure to add params settings
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("\nYour MLP Regressor model has been trained!")
            return y_pred
        else:
            model = MLPClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("\nYour MLP Classifier model has been trained!")
            return y_pred
        
    def GBM(self, X_train, y_train, X_test=None, y_test=None, binary=False):

        if X_test is None or y_test is None:
            raise ValueError("X_test & y_test must be provided for evaluation")
        
        if binary is False:
            model = GradientBoostingRegressor() #make sure to add params settings
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("\nYour GBM Regressor model has been trained!")
            return y_pred
        else:
            model = GradientBoostingClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("\nYour GBM Classifier model has been trained!")
            return y_pred
        
    def RF(self, X_train, y_train, X_test=None, y_test=None, binary=False):

        if X_test is None or y_test is None:
            raise ValueError("X_test & y_test must be provided for evaluation")
        
        if binary is False:
            model = RandomForestRegressor() #make sure to add params settings
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("\nYour RF Regressor model has been trained!")
            return y_pred
        else:
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("\nYour RF Classifier model has been trained!")
            return y_pred

    def SVM(self, X_train, y_train, X_test=None, y_test=None, binary=False):

        if X_test is None or y_test is None:
            raise ValueError("X_test & y_test must be provided for evaluation")
        
        if binary is False:
            model = SVR() #make sure to add params settings
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("\nYour SVM Regressor model has been trained!")
            return y_pred
        else:
            model = SVC()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("\nYour SVM Classifier model has been trained!")
            return y_pred


    def run_models(self, X_train, y_train, model_choice, seed, X_test=None, y_test=None, binary=False): 
        '''If we run through the list of algo's names in model_choice, we can run all the algo at the same time.
        if we to choose specific, we can specify by single name'''
        models = ['linear','DT','MLP','GBM','RF','SVM','BLR']

        if model_choice not in models:
            print(f"{model_choice} is not supported by SCLTool :( | Try with supported models!")
        
        if model_choice == "linear" and binary == True:
            print("\nRunning 'Logistic Regression' Model | Metric: 'ROC_AUC' ...")
            print('\n')
            print("Training...")
            np.random.seed(seed)
            y_pred = self.linear(X_train, y_train, X_test, y_test, True)
            result = self.results(y_test, y_pred, True)
            params_list =  {'C': np.arange(0.001, 1, 0.05), 'penalty': ['l2','l1']}
            performance = self.grid_searchcv(LogisticRegression(), X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                'Model': 'Logistic Regression',
                'Metric': "MSE",
                "ROC Score": result,
                "Grid Search CV Score": performance['best_score'][0],
                'Best Params': performance['best_params'][0]
            }
        elif model_choice == "linear" and binary == False:
            print("\nRunning 'Linear Regression' Model | Metric: 'MSE' ...")
            print("\nTraining...")
            np.random.seed(seed)
            y_pred = self.linear(X_train, y_train, X_test, y_test, False)
            result = self.results(y_test, y_pred, False)
            params_list =  {'fit_intercept': [True,False], 'n_jobs': [1,5,10,15,None], 'positive': [True,False]}
            performance = self.grid_searchcv(LinearRegression(), X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                'Model': 'Linear Regression',
                'Metric': "MSE",
                "MSE Score": result,
                "Grid Search CV Score": performance['best_score'][0],
                'Best Params': performance['best_params'][0]
            }
        
        
        elif model_choice == 'DT' and binary == True:
            print(f"Running 'DT Classifier' Model | Metric: 'ROC_AUC' ...")
            print("Training...")
            np.random.seed(seed)
            y_pred = self.DT(X_train, y_train, X_test, y_test, True)
            result = self.results(y_test, y_pred, True)
            params_list =  {"max_depth": [3,4,5,6,7,8,9,10],
                            "min_samples_leaf": [0.02, 0.04, 0.06],
                            "max_features": [0.4, 0.6, 0.8, 1.0]}
            performance = self.grid_searchcv(DecisionTreeClassifier(), X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                    'Model': 'Decision Tree Classifier',
                    'Metric': "ROC",
                    "ROC Score": result,
                    "Grid Search CV Score": performance['best_score'][0],
                    'Best Params': performance['best_params'][0]
                }
        elif model_choice == 'DT' and binary == False:
            print("\nRunning 'DT Regressor' Model | Metric: 'MSE' ...")
            print("\nTraining...")
            np.random.seed(seed)
            y_pred = self.DT(X_train, y_train, X_test, y_test, False)
            result = self.results(y_test, y_pred, False)
            params_list =  {"max_depth": [3,4,5,6,7,8,9,10],
                            "min_samples_leaf": [0.02, 0.04, 0.06],
                            "max_features": [0.4, 0.6, 0.8, 1.0]}
            performance = self.grid_searchcv(DecisionTreeRegressor(), X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                    'Model': 'Decision Tree Regression',
                    'Metric': "MSE",
                    "MSE Score": result,
                    "Grid Search CV Score": performance['best_score'][0],
                    'Best Params': performance['best_params'][0]
                }
        
        elif model_choice == 'MLP' and binary == True:
            print("\nRunning 'MLP Classifier' Model | Metric: 'ROC_AUC' ...")
            print("\nTraining...")
            np.random.seed(seed)
            y_pred = self.MLP(X_train, y_train, X_test, y_test, True)
            result = self.results(y_test, y_pred, True)
            params_list =  {'hidden_layer_sizes': [(10,),(20,),(50,),(100,)]}
            performance = self.grid_searchcv(MLPClassifier(), X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                    'Model': 'MLP Classifier',
                    'Metric': "ROC",
                    "ROC Score": result,
                    "Grid Search CV Score": performance['best_score'][0],
                    'Best Params': performance['best_params'][0]
                }
        elif model_choice == 'MLP' and binary == False:
            print("\nRunning 'MLP Regressor' Model | Metric: 'MSE' ...")
            print("\nTraining...")
            np.random.seed(seed)
            y_pred = self.MLP(X_train, y_train, X_test, y_test, False)
            result = self.results(y_test, y_pred, False)
            params_list =  {'hidden_layer_sizes': [(10,),(20,),(50,),(100,)]}
            performance = self.grid_searchcv(MLPRegressor(), X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                    'Model': 'MLP Regression',
                    'Metric': "MSE",
                    "MSE Score": result,
                    "Grid Search CV Score": performance['best_score'][0],
                    'Best Params': performance['best_params'][0]
                }

        elif model_choice == 'GBM' and binary == True:
            print("\nRunning 'GBM Classifier' Model | Metric: 'ROC_AUC' ...")
            print("\nTraining...")
            y_pred = self.GBM(X_train, y_train, X_test, y_test, True)
            result = self.results(y_test, y_pred, True)
            params_list =  {"learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                              "max_depth": [2,3,4,5],
                              "n_estimators": [20]}
            performance = self.grid_searchcv(GradientBoostingClassifier(), X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                    'Model': 'GBM Regression',
                    'Metric': "ROC",
                    "ROC Score": result,
                    "Grid Search CV Score": performance['best_score'][0],
                    'Best Params': performance['best_params'][0]
                }
        elif model_choice == 'GBM' and binary == False:
            print("\nRunning 'GBM Regression' Model | Metric: 'MSE' ...")
            print("\nTraining...")
            y_pred = self.GBM(X_train, y_train, X_test, y_test, False)
            result = self.results(y_test, y_pred, False)
            params_list =  {"learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                              "max_depth": [2,3,4,5],
                              "n_estimators": [20]}
            performance = self.grid_searchcv(GradientBoostingRegressor(), X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                    'Model': 'GBM Regression',
                    'Metric': "MSE",
                    "MSE Score": result,
                    "Grid Search CV Score": performance['best_score'][0],
                    'Best Params': performance['best_params'][0]
                }
        
        elif model_choice == 'RF' and binary == True:
            print("\nRunning 'RF Classifier' Model | Metric: 'ROC_AUC' ...")
            print("\nTraining...")
            y_pred = self.RF(X_train, y_train, X_test, y_test, True)
            result = self.results(y_test, y_pred, True)
            params_list =  {'max_depth': [1,2,3,4],
                            'n_estimators': [1,5,10]}
            performance = self.grid_searchcv(RandomForestClassifier(), X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                    'Model': 'RF Classifier',
                    'Metric': "ROC",
                    "ROC Score": result,
                    "Grid Search CV Score": performance['best_score'][0],
                    'Best Params': performance['best_params'][0]
                }
        elif model_choice == 'RF' and binary == False:
            print("\nRunning 'RF Regression' Model | Metric: 'MSE' ...")
            print("\nTraining...")
            y_pred = self.RF(X_train, y_train, X_test, y_test, False)
            result = self.results(y_test, y_pred, False)
            params_list =  {'max_depth': [1,2,3,4],
                            'n_estimators': [1,5,10]}
            performance = self.grid_searchcv(RandomForestRegressor(), X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                    'Model': 'RF Regression',
                    'Metric': "MSE",
                    "MSE Score": result,
                    "Grid Search CV Score": performance['best_score'][0],
                    'Best Params': performance['best_params'][0]
                }
        
        elif model_choice == 'SVM' and binary == True:
            print("\nRunning 'SVM Classification' Model | Metric: 'ROC_AUC' ...")
            print("\nTraining...")
            y_pred = self.SVM(X_train, y_train, X_test, y_test, True)
            result = self.results(y_test, y_pred, True)
            params_list =  {'C': [.1,1,10,100]}
            performance = self.grid_searchcv(SVC(), X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                    'Model': 'SVM Classifier',
                    'Metric': "ROC",
                    "ROC Score": result,
                    "Grid Search CV Score": performance['best_score'][0],
                    'Best Params': performance['best_params'][0]
                }
        elif model_choice == 'SVM' and binary == False:
            print("\nRunning 'SVM Regression' Model | Metric: 'MSE' ...")
            print("\nTraining...")
            y_pred = self.SVM(X_train, y_train, X_test, y_test, False)
            result = self.results(y_test, y_pred, False)
            params_list =  {'C': [.1,1,10,100]}
            performance = self.grid_searchcv(SVR(), X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                    'Model': 'SVM Regression',
                    'Metric': "MSE",
                    "MSE Score": result,
                    "Grid Search CV Score": performance['best_score'][0],
                    'Best Params': performance['best_params'][0]
                }

    def performances(self, performance):
        perf = []
        perf.append(performance)
        perf = pd.DataFrame(perf)
        return perf


#(------------------------------Implementation---------------------------------)

data = pd.read_csv("/Users/karnavivek/SCLTool/Data/WFP_dataset.csv")
X = data.drop('label', axis=1)
y = data.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

seed = 17

#For running ONE model in a single iteration:
# model_list = 'linear' 

#for running ALL models in a single iteration:
model_list = ['linear','DT','MLP','GBM','RF','SVM']
# model_list = ['linear','DT','MLP']
# model_list = ['linear'] #for running single model in a single iteration

all_results = []

for alg in model_list:
    model = ModelTraining(X, y)
    perf = model.run_models(X_train, y_train, alg, seed=seed, X_test=X_test, y_test=y_test, binary=False)
    all_results.append(perf)

results_df = pd.DataFrame(all_results)
print("\nModel Performance Summary:\n")
print(results_df)
print(f'\nOneClickML Recommeds "{results_df.loc[results_df['MSE Score'].idxmin(), 'Model']}" Model with Minimum score: {min(results_df['MSE Score'])}')

'''
we need Max(Grid Search CV Score) & Min(MSE Score) to select the model for model_embeddings.
By performing model_training.py to Palatable WFP Problem, we found out that best model recommended by OneClickML is GBM Regressor,
Which is the similar to opticl's answer! :)

Next step is to Look at the Best Params selected by GridSearchCV and convert these params into something which is readable by model_embeddings
'''

end = time.time()
print(f'\nTime taken for OneClickML to train the model(s): {end-start} sec')
