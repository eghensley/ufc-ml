#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:30:48 2020

@author: ehens86
"""
import lightgbm
from sklearn.model_selection import StratifiedKFold, KFold
import platform
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import log_loss, mean_squared_error

def _single_core_solver(input_vals):
#   trainx, testx, trainy, testy, model = job
    trainx, testx, trainy, testy, model = input_vals
    if len(trainy.unique()) == 2:
        obj = 'class'
    else:
        obj = 'reg'        
    model.fit(trainx, trainy)       
    if obj == 'class':    
        pred = model.predict_proba(testx)
        pred = [i[1] for i in pred]
    else:
        pred = model.predict(testx)
    return(pd.DataFrame(pred, testy.index))
    
def _single_core_eval(input_vals):
#   trainx, testx, trainy, testy, model = job
    trainx, testx, trainy, testy, model = input_vals
    if len(trainy.unique()) == 2:
        obj = 'class'
    else:
        obj = 'reg'
    if obj == 'class':
        test_weights = class_weight.compute_class_weight('balanced',
                                    np.unique(trainy),trainy)    
        test_weights_dict = {i:j for i,j in zip(np.unique(trainy), test_weights)}            
    model.fit(trainx, trainy)       
    if obj == 'class':    
        pred = model.predict_proba(testx)
    else:
        pred = model.predict(testx)
    if obj == 'class':
        score = log_loss(testy, pred, sample_weight = [test_weights_dict[i] for i in testy]) * -1
    else:
        score = mean_squared_error(testy, pred) * -1
    return(score)
    
def cross_validate(x,y,est, only_scores = True, njobs = -1, verbose = False): 
#    x,y,est, only_scores, njobs, verbose = x,Y,estimator, True, -1, True
    if len(y.unique()) == 2:
        splitter = StratifiedKFold(n_splits = 8, random_state = 1108)
    else:
        splitter = KFold(n_splits = 8, random_state = 1108)        
    if (est.steps[-1][1].__class__ == lightgbm.sklearn.LGBMClassifier or est.steps[-1][1].__class__ == lightgbm.sklearn.LGBMRegressor) and platform.system() == 'Linux':
        njobs = 1        
    all_folds = []
    for fold in splitter.split(x, y):
        all_folds.append(fold)    
    jobs = []
    for train, test in all_folds:
        jobs.append([x.iloc[train], x.iloc[test], y.iloc[train], y.iloc[test], est])    
    if njobs == 1:
        cv_results = []
        for job in jobs:
            if only_scores:
                cv_results.append(_single_core_eval(job)) 
            else:
                cv_results.append(_single_core_solver(job))
    else:
        if only_scores:
            if verbose:
                cv_results = Parallel(n_jobs = njobs, verbose = 25)(delayed(_single_core_eval) (i) for i in jobs)
            else:
                cv_results = Parallel(n_jobs = njobs)(delayed(_single_core_eval) (i) for i in jobs)
        else:
            if verbose:
                cv_results = Parallel(n_jobs = njobs, verbose = 25)(delayed(_single_core_solver) (i) for i in jobs)
            else:
                cv_results = Parallel(n_jobs = njobs)(delayed(_single_core_solver) (i) for i in jobs)            
    if only_scores:
        results = np.mean(cv_results)
    else:
        results = pd.DataFrame()
        for df in cv_results:
            results = results.append(df)
    return(results)