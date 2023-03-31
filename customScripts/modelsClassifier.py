import os
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, recall_score, precision_score, fbeta_score, roc_curve, auc

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from preprocessing import imputation_scaling, SupervisedSelector


def fit_GS(X,
             y,
             target,
             sel_variables,
             model, 
             gridSearchScorer = "f1_macro",
             innerCV=3, 
             njobs=9,
             randomState=None):

    ''' 
    Only innerCV loop (for CV - AUC ROC plotting!)
    '''
    

    
    '''
    Prepare preprocessing
    '''
    num_columns = X.select_dtypes(include=["float64"]).columns
    bin_columns = X.select_dtypes(include=["int64"]).columns
    cat_columns = X.select_dtypes(include=["object"]).columns
    preprocessor = imputation_scaling(num_columns, bin_columns, cat_columns, X)        

    '''
    Implement: dict with HP for each model
    '''
    with open("/home/skatz/PROJECTS/covid/disease_severity/customScripts/HP_grid.json", "r") as f: hp_grid = json.load(f)
    clf = eval(hp_grid[f"model_{model}"])
    param_grid = hp_grid[f"grid_{model}"]    
    

    ### assemble pipe ###
    pipe_imputation = Pipeline([("imputation", preprocessor)])
    pipe_clf = Pipeline([*pipe_imputation.steps,
                        ("selector", SupervisedSelector(pipe_imputation["imputation"], 
                                                        sel_variables)),
                        (model, clf)])

    ### do GridSearch ###
    gs = GridSearchCV(pipe_clf,
                      param_grid,
                      cv=StratifiedKFold(n_splits=innerCV, shuffle=True, random_state=randomState),        ### Set random state so models can be compared with same split!
                      n_jobs=njobs,
                      scoring=gridSearchScorer,
                      verbose=0)
    gs.fit(X,y)

    return gs.best_estimator_


def fit_modelComparison(X,
                        y,
                        target,
                        sel_variables,
                        model,
                        gridSearchScorer = "f1_macro",
                        innerCV=3,
                        outerCV=3,
                        njobs=9,
                        randomState=None):

    print("\nFitting estimator...\n")

    ### Prepare Scoring ###
    scoring = {"acc":"accuracy", \
               "b_acc":"balanced_accuracy", \
               "recall":"recall", \
               "precision":"precision", \
               "roc_auc_micro":make_scorer(roc_auc_score,average="micro", needs_proba=True), \
               "roc_auc_macro":make_scorer(roc_auc_score,average="macro", needs_proba=True), \
               "f1":"f1", \
               "f1_micro":"f1_micro", \
               "f1_macro":"f1_macro", \
               "f1_weighted":"f1_weighted", \
               "fbeta":make_scorer(fbeta_score,beta=2)}

    '''
    Prepare preprocessing
    '''
    num_columns = X.select_dtypes(include=["float64"]).columns
    bin_columns = X.select_dtypes(include=["int64"]).columns
    cat_columns = X.select_dtypes(include=["object"]).columns
    preprocessor = imputation_scaling(num_columns, bin_columns, cat_columns, X)

    '''
    Implement: dict with HP for each model
    '''
    with open("/home/skatz/PROJECTS/covid/disease_severity/customScripts/HP_grid.json", "r") as f: hp_grid = json.load(f)
    clf = eval(hp_grid[f"model_{model}"])
    param_grid = hp_grid[f"grid_{model}"]


    ### assemble pipe ###
    pipe_imputation = Pipeline([("imputation", preprocessor)])
    pipe_clf = Pipeline([*pipe_imputation.steps,
                        ("selector", SupervisedSelector(pipe_imputation["imputation"], 
                                                        sel_variables)),
                        (model, clf)])


    ### do GridSearch ###
    gs = GridSearchCV(pipe_clf,
                      param_grid,
                      cv=StratifiedKFold(n_splits=innerCV, shuffle=True, random_state=randomState),        ### Set random state so models can be compared with same split!
                      n_jobs=njobs,
                      scoring=gridSearchScorer,
                      verbose=0)

    ### Set-up cross_validation scoring ###
    score = cross_validate(gs, X, y,
                           cv = StratifiedKFold(n_splits=outerCV, shuffle=True, random_state=randomState),   ### Set random state so models can be compared with same split!
                           scoring=scoring,
                           return_train_score=False,
                           return_estimator=True)

    estimators = score["estimator"]
    score.pop("estimator")


    return estimators, score



def fit_modelComparison_LOOCV(X,
                        y,
                        target,
                        sel_variables,
                        model,
                        gridSearchScorer = "f1_macro",
                        innerCV=3,
                        njobs=9,
                        randomState=None):

    print("\nFitting estimator...\n")

    ### Prepare Scoring ###
    scoring = {"acc":"accuracy", \
               "b_acc":"balanced_accuracy", \
               "recall":"recall", \
               "precision":"precision", \
               "auc":make_scorer(roc_auc_score), \
               "f1":"f1", \
               "f1_micro":"f1_micro", \
               "f1_macro":"f1_macro", \
               "f1_weighted":"f1_weighted", \
               "fbeta":make_scorer(fbeta_score,beta=2)}

    '''
    Prepare preprocessing
    '''
    num_columns = X.select_dtypes(include=["float64"]).columns
    bin_columns = X.select_dtypes(include=["int64"]).columns
    cat_columns = X.select_dtypes(include=["object"]).columns
    preprocessor = imputation_scaling(num_columns, bin_columns, cat_columns, X)

    '''
    Implement: dict with HP for each model
    '''
    with open("/home/skatz/PROJECTS/covid/disease_severity/customScripts/HP_grid.json", "r") as f: hp_grid = json.load(f)
    clf = eval(hp_grid[f"model_{model}"])
    param_grid = hp_grid[f"grid_{model}"]


    ### assemble pipe ###
    pipe_imputation = Pipeline([("imputation", preprocessor)])
    pipe_clf = Pipeline([*pipe_imputation.steps,
                        ("selector", SupervisedSelector(pipe_imputation["imputation"], 
                                                        sel_variables)),
                        (model, clf)])


    ### do GridSearch ###
    gs = GridSearchCV(pipe_clf,
                      param_grid,
                      cv=StratifiedKFold(n_splits=innerCV, shuffle=True, random_state=randomState),        ### Set random state so models can be compared with same split!
                      n_jobs=njobs,
                      scoring=gridSearchScorer,
                      verbose=0)

    ### Set-up cross_validation scoring ###
    score = cross_validate(gs, X, y,
                           cv = LeaveOneOut(),  
                           scoring=scoring,
                           return_train_score=False,
                           return_estimator=True)

    estimators = score["estimator"]
    score.pop("estimator")


    return estimators, score