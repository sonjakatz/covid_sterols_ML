import os
import pandas as pd
import numpy as np
import json

from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

def run_iterativeBoruta(X, y, cols, perc=100, n_iter=100):

    dict_boruta = {}

    for i in range(n_iter):
        print(f"Round {i+1} of {n_iter}")

        ''' 
        Setup and run Boruta
        '''
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=None, perc=perc)
        feat_selector.fit(X, y)

        ''' 
        Get selected variables and save in dict
        '''
        selVars = np.array(cols)[feat_selector.support_]
        for var in selVars: 
            if var in dict_boruta.keys():
                dict_boruta[var] += 1
            else: 
                dict_boruta[var] = 1

    ### Normalise regarding number of iterations
    dict_boruta.update((x, y/n_iter) for x, y in dict_boruta.items())
    
    return dict_boruta


def imputation_scaling(num_columns, bin_columns, cat_columns, X):

    '''
    ### Imputation ###
    - Numerical features (float64):
        - MICE
        - MinMaxScaler
    - Categorical features (int64):
        - KNN
    - Categorical features (objects):
        - SimpleImputer("most_frequent")
    '''

    num_transformer = Pipeline([
        ("scaler", MinMaxScaler()),
        ("imputer", IterativeImputer(random_state=11,       ### Set random state so models can be compared with same split!
                           max_iter=10,
                           verbose=0,
                           tol=0.001,
                           sample_posterior=True,
                           n_nearest_features=5))])

    bin_transformer = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5))])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy='most_frequent'))])

    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_columns),
                                                   ("bin", bin_transformer, bin_columns),
                                                   ("cat", cat_transformer, cat_columns)])

    return preprocessor.fit(X)



class SupervisedSelector():
    '''
    Part of prediction pipeline - only parses the variables it receives and returns pruned dataset
    ''' 
    def __init__(self, preprocessor, features, argument=None):
        self.preprocessor = preprocessor
        self.features = features
        self.argument = argument

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        columns = [ele[5:] for ele in self.preprocessor.get_feature_names_out()]
        df_X = pd.DataFrame(X, columns=columns)
        self.X_ = df_X.loc[:,self.features]
        return self.X_

    def get_feature_names(self):
        return self.X_.columns.tolist()

