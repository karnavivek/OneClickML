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
from sklearn.svm import LinearSVR
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import time

start = time.time()

class ModelTraining:
    def __init__(self):
        return
        # self.X = X
        # self.y = y

    def results(self, y_test, y_pred, binary=False):
        if binary == False:
            mse = mean_squared_error(y_test, y_pred)
            # r2 = r2_score(y_test, y_pred)
            return mse
        else:
            roc_auc = roc_auc_score(y_test, y_pred)
            # r2 = r2_score(y_test, y_pred)
        return roc_auc
    
    def r2(self, y_true, y_pred, y_mean):
        SSR = ((y_true - y_pred)**2).sum()
        SST = ((y_true - y_mean)**2).sum()
        return (1-(SSR/SST))
    
    def grid_searchcv(self, model_name, X_train, y_train, params_list, cv=5):
        scores = []
        gs = GridSearchCV(estimator=model_name, param_grid=params_list, cv=cv, n_jobs=-1, return_train_score=False)
        gs.fit(X_train, y_train)
        scores.append(
                {'model_name': model_name,
                'best_score': gs.best_score_,
                'best_params': gs.best_params_,
                'model_best_estimator': gs.best_estimator_
                }
            )
        scores = pd.DataFrame(scores)
        # print(f'Your {model} has been Trained Successfully!')
        return scores

    # def pickle_models(self, model, model_name: str):
    #     directory = '/Users/karnavivek/SCLTool/pickledmodels/'
    #     filename = f'{model_name}_model.pkl'
    #     filepath = os.path.join(directory, filename)
    #     os.makedirs(directory, exist_ok=True)
    #     with open(filepath, 'wb') as file:
    #         pickle.dump(model, file)

    def linear(self, X_train, y_train, X_test=None, y_test=None, binary=False):

        if X_test is None or y_test is None:
            raise ValueError("X_test & y_test must be provided for evaluation")
            
        if binary == False:
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # self.pickle_models(model, 'linear')
            print("\nYour Linear Regression model has been trained!")
            return model, y_pred
        else:
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # self.pickle_models(model, 'logistic')
            print("\nYour Logistic Regression model has been trained!")
            return model, y_pred
    

    def DT(self, X_train, y_train, X_test=None, y_test=None, binary=False):

        if X_test is None or y_test is None:
            raise ValueError("X_test & y_test must be provided for evaluation")
        
        if binary is False:
            model = DecisionTreeRegressor() #make sure to add params settings
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # self.pickle_models(model, 'DT')
            print("\nYour Decision Tree Regressor model has been trained!")
            return model, y_pred
        else:
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # self.pickle_models(model, 'DT')
            print("\nYour Decision Tree Classifier model has been trained!")
            return model, y_pred

    def MLP(self, X_train, y_train, X_test=None, y_test=None, binary=False):
        
        if X_test is None or y_test is None:
            raise ValueError("X_test & y_test must be provided for evaluation")
        
        if binary is False:
            model = MLPRegressor() #make sure to add params settings
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # self.pickle_models(model, 'MLP')
            print("\nYour MLP Regressor model has been trained!")
            return model, y_pred
        else:
            model = MLPClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # self.pickle_models(model, 'MLP')
            print("\nYour MLP Classifier model has been trained!")
            return model, y_pred
        
    def GBM(self, X_train, y_train, X_test=None, y_test=None, binary=False):

        if X_test is None or y_test is None:
            raise ValueError("X_test & y_test must be provided for evaluation")
        
        if binary is False:
            model = GradientBoostingRegressor() #make sure to add params settings
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # self.pickle_models(model, 'GBM')
            print("\nYour GBM Regressor model has been trained!")
            return model, y_pred
        else:
            model = GradientBoostingClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # self.pickle_models(model, 'GBM')
            print("\nYour GBM Classifier model has been trained!")
            return model, y_pred
        
    def RF(self, X_train, y_train, X_test=None, y_test=None, binary=False):

        if X_test is None or y_test is None:
            raise ValueError("X_test & y_test must be provided for evaluation")
        
        if binary is False:
            model = RandomForestRegressor() #make sure to add params settings
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # self.pickle_models(model, 'RF')
            print("\nYour RF Regressor model has been trained!")
            return model, y_pred
        else:
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # self.pickle_models(model, 'RF')
            print("\nYour RF Classifier model has been trained!")
            return model, y_pred

    def SVM(self, X_train, y_train, X_test=None, y_test=None, binary=False):

        if X_test is None or y_test is None:
            raise ValueError("X_test & y_test must be provided for evaluation")
        
        if binary is False:
            model = LinearSVR(max_iter=int(1e5), dual=False, loss='squared_epsilon_insensitive') #make sure to add params settings
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # self.pickle_models(model, 'SVM')
            print("\nYour SVM Regressor model has been trained!")
            return model, y_pred
        else:
            model = LinearSVC(max_iter=1e5, dual=False, penalty='l2')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # self.pickle_models(model, 'SVM')
            print("\nYour SVM Classifier model has been trained!")
            return model, y_pred


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
            _model, y_pred = self.linear(X_train, y_train, X_test, y_test, True)
            result = self.results(y_test, y_pred, True)
            y_mean = y_test.mean()
            r2 = self.r2(y_test, y_pred, y_mean)
            print(f'\nTest MSE: {result}')
            print(f'Test R2:  {r2}')
            params_list =  {'C': np.arange(0.001, 1, 0.05), 'penalty': ['l2','l1']}
            performance = self.grid_searchcv(_model, X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                'Model': 'Logistic Regression',
                'Metric': "ROC",
                "ROC Score": result,
                "GS CV Score": performance['best_score'][0],
                'Best Params': performance['best_params'][0],
            }
        elif model_choice == "linear" and binary == False:
            print("\nRunning 'Linear Regression' Model | Metric: 'MSE' ...")
            print("\nTraining...")
            np.random.seed(seed)
            _model, y_pred = self.linear(X_train, y_train, X_test, y_test, False)
            result = self.results(y_test, y_pred, False)
            y_mean = y_test.mean()
            # r2 = self.r2(y_test, y_pred, y_mean)
            print(f'\nTest MSE: {result}')
            # print(f'Test R2:  {r2}')
            params_list = {'fit_intercept': [True,False], 'n_jobs': [1,5,10,15,None], 'positive': [True,False]}
            performance = self.grid_searchcv(_model, 
                                             X_train, 
                                             y_train, 
                                             params_list)
            print("\n-------------------------------------------------------")
            return {
                'Model': 'Linear Regression',
                'Metric': "MSE",
                "MSE Score": result,
                # 'R2 Score': r2,
                "GS CV Score": performance['best_score'][0],
                'Best Params': str(performance['best_params'][0]),
                'Best Estimator': str(performance['model_best_estimator'][0])
            }
 

        elif model_choice == 'DT' and binary == True:
            print(f"Running 'DT Classifier' Model | Metric: 'ROC_AUC' ...")
            print("Training...")
            np.random.seed(seed)
            _model, y_pred = self.DT(X_train, y_train, X_test, y_test, True)
            result = self.results(y_test, y_pred, True)
            y_mean = y_test.mean()
            r2 = self.r2(y_test, y_pred, y_mean)
            print(f'\nTest MSE: {result}')
            print(f'Test R2:  {r2}')
            params_list =  {"max_depth": [3,4,5,6,7,8,9,10],
                            "min_samples_leaf": [0.02, 0.04, 0.06],
                            "max_features": [0.4, 0.6, 0.8, 1.0]}
            performance = self.grid_searchcv(_model, X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                    'Model': 'Decision Tree Classifier',
                    'Metric': "ROC",
                    "ROC Score": result,
                    "GS CV Score": performance['best_score'][0],
                    'Best Params': performance['best_params'][0],
                'Best Estimator': performance['model_best_estimator'][0]
                }
        elif model_choice == 'DT' and binary == False:
            print("\nRunning 'DT Regressor' Model | Metric: 'MSE' ...")
            print("\nTraining...")
            np.random.seed(seed)
            _model, y_pred = self.DT(X_train, y_train, X_test, y_test, False)
            result = self.results(y_test, y_pred, False)
            y_mean = y_test.mean()
            # r2 = self.r2(y_test, y_pred, y_mean)
            print(f'\nTest MSE: {result}')
            # print(f'Test R2:  {r2}')
            params_list =  {"max_depth": [3,4,5,6,7,8,9,10],
                            "min_samples_leaf": [0.02, 0.04, 0.06],
                            "max_features": [0.4, 0.6, 0.8, 1.0]}
            performance = self.grid_searchcv(_model, X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                    'Model': 'Decision Tree Regression',
                    'Metric': "MSE",
                    "MSE Score": result,
                    # 'R2 Score': r2,
                    "GS CV Score": performance['best_score'][0],
                    'Best Params': str(performance['best_params'][0]),
                'Best Estimator': str(performance['model_best_estimator'][0])
                }
        
        elif model_choice == 'MLP' and binary == True:
            print("\nRunning 'MLP Classifier' Model | Metric: 'ROC_AUC' ...")
            print("\nTraining...")
            np.random.seed(seed)
            _model, y_pred = self.MLP(X_train, y_train, X_test, y_test, True)
            result = self.results(y_test, y_pred, True)
            y_mean = y_test.mean()
            r2 = self.r2(y_test, y_pred, y_mean)
            print(f'\nTest MSE: {result}')
            print(f'Test R2:  {r2}')
            params_list =  {'hidden_layer_sizes': [(10,),(20,),(50,),(100,)]}
            performance = self.grid_searchcv(_model, X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                    'Model': 'MLP Classifier',
                    'Metric': "ROC",
                    "ROC Score": result,
                    "GS CV Score": performance['best_score'][0],
                    'Best Params': performance['best_params'][0]
                }
        elif model_choice == 'MLP' and binary == False:
            print("\nRunning 'MLP Regressor' Model | Metric: 'MSE' ...")
            print("\nTraining...")
            np.random.seed(seed)
            _model, y_pred = self.MLP(X_train, y_train, X_test, y_test, False)
            result = self.results(y_test, y_pred, False)
            y_mean = y_test.mean()
            # r2 = self.r2(y_test, y_pred, y_mean)
            print(f'\nTest MSE: {result}')
            # print(f'Test R2:  {r2}')
            params_list =  {'hidden_layer_sizes': [(10,),(20,),(50,),(100,)]}
            performance = self.grid_searchcv(_model, X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                    'Model': 'MLP Regression',
                    'Metric': "MSE",
                    "MSE Score": result,
                    # 'R2 Score': r2,
                    "GS CV Score": performance['best_score'][0],
                    'Best Params': str(performance['best_params'][0]),
                'Best Estimator': str(performance['model_best_estimator'][0])
                }

        elif model_choice == 'GBM' and binary == True:
            print("\nRunning 'GBM Classifier' Model | Metric: 'ROC_AUC' ...")
            print("\nTraining...")
            _model, y_pred = self.GBM(X_train, y_train, X_test, y_test, True)
            result = self.results(y_test, y_pred, True)
            y_mean = y_test.mean()
            r2 = self.r2(y_test, y_pred, y_mean)
            print(f'\nTest MSE: {result}')
            print(f'Test R2:  {r2}')
            params_list =  {"learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                              "max_depth": [2,3,4,5],
                              "n_estimators": [20]}
            performance = self.grid_searchcv(_model, X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                    'Model': 'GBM Classifier',
                    'Metric': "ROC",
                    "ROC Score": result,
                    "GS CV Score": performance['best_score'][0],
                    'Best Params': performance['best_params'][0]
                }
        elif model_choice == 'GBM' and binary == False:
            print("\nRunning 'GBM Regression' Model | Metric: 'MSE' ...")
            print("\nTraining...")
            _model, y_pred = self.GBM(X_train, y_train, X_test, y_test, False)
            result = self.results(y_test, y_pred, False)
            y_mean = y_test.mean()
            # r2 = self.r2(y_test, y_pred, y_mean)
            print(f'\nTest MSE: {result}')
            # print(f'Test R2:  {r2}')
            params_list =  {"learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                              "max_depth": [2,3,4,5],
                              "n_estimators": [20]}
            performance = self.grid_searchcv(_model, X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                    'Model': 'GBM Regression',
                    'Metric': "MSE",
                    "MSE Score": result,
                    # 'R2 Score': r2,
                    "GS CV Score": performance['best_score'][0],
                    'Best Params': str(performance['best_params'][0]),
                'Best Estimator': str(performance['model_best_estimator'][0])
                }
        
        elif model_choice == 'RF' and binary == True:
            print("\nRunning 'RF Classifier' Model | Metric: 'ROC_AUC' ...")
            print("\nTraining...")
            _model, y_pred = self.RF(X_train, y_train, X_test, y_test, True)
            result = self.results(y_test, y_pred, True)
            y_mean = y_test.mean()
            r2 = self.r2(y_test, y_pred, y_mean)
            print(f'\nTest MSE: {result}')
            print(f'Test R2:  {r2}')
            params_list =  {'max_depth': [1,2,3,4],
                            'n_estimators': [1,5,10]}
            performance = self.grid_searchcv(_model, X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                    'Model': 'RF Classifier',
                    'Metric': "ROC",
                    "ROC Score": result,
                    "GS CV Score": performance['best_score'][0],
                    'Best Params': performance['best_params'][0]
                }
        elif model_choice == 'RF' and binary == False:
            print("\nRunning 'RF Regression' Model | Metric: 'MSE' ...")
            print("\nTraining...")
            _model, y_pred = self.RF(X_train, y_train, X_test, y_test, False)
            result = self.results(y_test, y_pred, False)
            y_mean = y_test.mean()
            # r2 = self.r2(y_test, y_pred, y_mean)
            print(f'\nTest MSE: {result}')
            # print(f'Test R2:  {r2}')
            params_list =  {'max_depth': [1,2,3,4],
                            'n_estimators': [1,5,10]}
            performance = self.grid_searchcv(_model, X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                    'Model': 'RF Regression',
                    'Metric': "MSE",
                    "MSE Score": result,
                    # 'R2 Score': r2,
                    "GS CV Score": performance['best_score'][0],
                    'Best Params': str(performance['best_params'][0]),
                'Best Estimator': str(performance['model_best_estimator'][0])
                }
        
        elif model_choice == 'SVM' and binary == True:
            print("\nRunning 'SVM Classification' Model | Metric: 'ROC_AUC' ...")
            print("\nTraining...")
            _model, y_pred = self.SVM(X_train, y_train, X_test, y_test, True)
            result = self.results(y_test, y_pred, True)
            y_mean = y_test.mean()
            r2 = self.r2(y_test, y_pred, y_mean)
            print(f'\nTest MSE: {result}')
            print(f'Test R2:  {r2}')
            params_list =  {'C': [.1,1,10,100]}
            performance = self.grid_searchcv(_model, X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                    'Model': 'SVM Classifier',
                    'Metric': "ROC",
                    "ROC Score": result,
                    'R2 Score': r2,
                    "GS CV Score": performance['best_score'][0],
                    'Best Params': performance['best_params'][0]
                }
        elif model_choice == 'SVM' and binary == False:
            print("\nRunning 'SVM Regression' Model | Metric: 'MSE' ...")
            print("\nTraining...")
            _model, y_pred = self.SVM(X_train, y_train, X_test, y_test, False)
            result = self.results(y_test, y_pred, False)
            y_mean = y_test.mean()
            # r2 = self.r2(y_test, y_pred, y_mean)
            print(f'\nTest MSE: {result}')
            # print(f'Test R2:  {r2}')
            params_list =  {'C': [.1,1,10,100]}
            performance = self.grid_searchcv(_model, X_train, y_train, params_list)
            print("\n-------------------------------------------------------")
            return {
                    'Model': 'SVM Regression',
                    'Metric': "MSE",
                    "MSE Score": result,
                    # 'R2 Score': r2,
                    "GS CV Score": performance['best_score'][0],
                    'Best Params': str(performance['best_params'][0]),
                    'Best Estimator': str(performance['model_best_estimator'][0])
                }
         
        elif model_choice == 'GPR' and binary == True:
            print("GP in Classification not supported as of now, Please check again in later updates")
        elif model_choice == 'GPR' and binary == False:
            print("\nRunning 'Gaussian Process Regression' Model | Metric: 'MSE' ...")
            print("\nTraining...")  







    def save_models(self, enter_var, model, save_path):
        enter_var = pd.DataFrame(enter_var, index=[0])
        enter_var.to_csv(save_path+'%s_results.csv' % model, index=False)

