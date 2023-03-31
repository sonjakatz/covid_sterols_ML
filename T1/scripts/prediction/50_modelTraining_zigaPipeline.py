''' 
Start remotely to run in the background like this:

ssh skatz@212.235.237.86 -p 62636 'source miniconda3/bin/activate sklearn | /home/skatz/miniconda3/envs/sklearn/bin/python /home/skatz/PROJECTS/covid/disease_severity/discoveryValidation/T1/scripts/prediction/50_modelTraining_zigaPipeline.py'
'''


import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt 
import seaborn as sns
import random
import sys

from classification_ziga import classify_leave_one_out_cv
from preprocessing import imputation_scaling, SupervisedSelector

sns.set_theme(style="whitegrid", palette=None, font_scale=1.2)

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier 

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel

import warnings
warnings.filterwarnings('ignore')

PATH_base = "/home/skatz/PROJECTS/covid/disease_severity/discoveryValidation/T1"

''' Prepare data '''
datasetTimepoint = "12_sterol_discoveryValidation_corrected"
target = "disease_severity"
#### Feature selection:
vars = "iterativeBoruta"   
varPath = f"{PATH_base}/results/featureSelection/{datasetTimepoint}/{vars}.txt"

''' 
Define paths
'''
resultsPath = f"{PATH_base}/results/prediction/featureSelection/{datasetTimepoint}/{vars}/modelComparison/zigaPipeline"
os.makedirs(resultsPath, exist_ok=True)
dataPath = f"{PATH_base}/results/preprocessing/cleaned"
dataset = f"{datasetTimepoint}_{target}_cleaned.csv"

models = {
      #    'rfc': RandomForestClassifier(),
      #     ###'svc': SVC(probability=True),
      #     'gpr': GaussianProcessClassifier(),
      #     'abc': AdaBoostClassifier(base_estimator = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "balanced",max_depth = None)),
      #     'log': LogisticRegression(),
      #     'knn': KNeighborsClassifier(),
          'mlp': MLPClassifier(),
          'gnb': GaussianNB(),
          'qda': QuadraticDiscriminantAnalysis(),
          'mcl': DummyClassifier(strategy="most_frequent"),
         }               


grids = {'rfc':{
               'n_estimators': [100, 300, 1000],      ### changed
               'max_depth': [2,4,6],         
               'max_features': [2,4,6],  
               'ccp_alpha':  list(np.linspace(0, 0.025, 2)),   
               },
         'svc':{'C': [0.1, 1, 10, 100],  
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                'kernel': ['rbf', 'poly', 'linear']     
               },
         'gpr':{'kernel':[1*RBF(), 1*DotProduct(), 1*Matern(),  1*RationalQuadratic(), 1*WhiteKernel()]},
         'abc':{"base_estimator__criterion" : ["gini", "entropy"],
                "base_estimator__splitter" :   ["best", "random"],
                "n_estimators": [1, 2]
               },
         'log':{'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10,100,1000]},     
         'knn':{'n_neighbors': list(range(1, 15)),
               'weights': ['uniform', 'distance'],
               'metric': ['euclidean', 'manhattan']},
         'mlp':{'solver': ['adam'],    
                'max_iter': [50, 100, 200],        
                'alpha': 10.0 ** -np.arange(0, 5),            
                'hidden_layer_sizes': [(random.randrange(15, 41), random.randrange(5, 16)) for i in range(5)],           
               },  
         'gnb': {'var_smoothing': np.logspace(-9,9, num=100)},
         'qda': {'reg_param': (0.00001, 0.0001, 0.001,0.01, 0.1), 
                 'store_covariance': (True, False),
                 'tol': (0.0001, 0.001,0.01, 0.1)},
         'mcl': {},  
         }   

''' 
Read data
'''
data = pd.read_csv(f"{dataPath}/{dataset}", index_col=0)
X = data.drop(target, axis=1)
y = data[target]

#### FOR DEVELOPMENT PURPOSES: smaller dataset
# X = X.iloc[:8,:]
# y = y[:8]

''' 
Read in variables
'''
sel_variables = pd.read_csv(varPath, header=None)[0].tolist()

''' 
Impute & scale X (according to Sonja's pipeline)
only works like this with LOOCV (no MICE possible on test split!)
'''
num_columns = X.select_dtypes(include=["float64"]).columns
bin_columns = X.select_dtypes(include=["int64"]).columns
cat_columns = X.select_dtypes(include=["object"]).columns
preprocessor = imputation_scaling(num_columns, bin_columns, cat_columns, X)     
X_imputed = preprocessor.fit_transform(X)
X_imputed = SupervisedSelector(preprocessor, sel_variables).transform(X_imputed)

''' 
Run Pipeline
'''
for model in models.keys():
      print(model)
      df_before = pd.DataFrame()    
      df_features = pd.DataFrame()
      df_importances = pd.DataFrame() 

      saveIndivdualPred = True

      clf = GridSearchCV(models[model], grids[model], scoring='balanced_accuracy', verbose=0, cv=3, n_jobs=-1)  ##cv=5
      result = classify_leave_one_out_cv(clf, 
                              X_imputed, 
                              y,
                              model=model, 
                              save_to = resultsPath + f"/{model}", 
                              select_features=True, 
                              permutation_repeats=100, 
                              scale_features = False,
                              saveIndivdualPred = saveIndivdualPred,
                              logfile = f"log_T1_{vars}")
      print(result)
      ''' Prepare results '''
      result['model'] = model        
      df_before = df_before.append(result['df_results'], ignore_index=True)  
      df_importances = df_importances.append(result['importances_df'], ignore_index=True)     
      if saveIndivdualPred:
            df_indPred = pd.DataFrame()
            df_indPred = df_indPred.append(result["df_indPred"], ignore_index=True)

      del result['df_results']      
      del result['importances_df']   
      del result['df_indPred']
      df_features = df_features.append(result, ignore_index=True)  

      ''' Save to file '''
      df_before.to_csv((resultsPath+f"/prediction_cv_test_{model}.csv"), index=False)          
      df_features.to_csv((resultsPath+f"/features_test_{model}.csv"), index=False)  
      df_importances.to_csv((resultsPath+f"/importances_test_{model}.csv"), index=False)                                 
      df_indPred.to_csv((resultsPath+f"/individualPredictions_test_{model}.csv"), index=False)      
      
