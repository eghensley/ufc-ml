#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:04:52 2020

@author: ehens86
"""
if __name__ == "__main__":
    import sys
    sys.path.append("..")
    
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import uuid
import numpy as np
import json
from joblib import dump

from utils.general import progress
import random
import optuna
from training.training_utils import cross_validate, form_to_domain, retrieve_reduced_domain_features
import warnings
warnings.filterwarnings("ignore") 

def _logRegTuning(x, y, post_feat = False, domain = 'strike'):
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    for slvr in solvers:
        for c in np.logspace(-3, 3, 10):
            model_id = str(uuid.uuid4())
            mod = LogisticRegression(random_state = 2, solver = slvr, C = c)
            score = cross_validate(x, y, mod)
            score['id'] = model_id
            
            if post_feat:
                with open('./training/ml/%s/post_feat/scores/%s.json' % (domain, model_id), 'w') as f:
                    json.dump(score, f)
                dump(mod, "./training/ml/%s/post_feat/models/%s.joblib" % (domain, model_id)) 
            else:
                with open('./training/ml/%s/pre_feat/scores/%s.json' % (domain, model_id), 'w') as f:
                    json.dump(score, f)
                dump(mod, "./training/ml/%s/pre_feat/models/%s.joblib" % (domain, model_id)) 
       
def _lgbTuning(x, y, post_feat = False, domain = 'strike', n_iter = 5000):
#    param_grid = {'num_leaves': sp_randint(6, 50), 
#             'min_child_samples': sp_randint(100, 500), 
#             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
#             'subsample': sp_uniform(loc=0.2, scale=0.8), 
#             'colsample_bytree': sp_uniform(loc=0.4, scale=0.8),
#             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
#             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
    
#    param_grid = {
#        'objective': 'binary',
#        'metric': 'binary_logloss',
#        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
#        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
#        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
#        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
#        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
#        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
#        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#    }
    if post_feat:
        param_grid = {
            'colsample_bytree': np.linspace(.3,.9, 10),
            'num_leaves': [int(i) for i in np.linspace(15, 500, 50)],
            'reg_alpha': np.logspace(-3,3, 10),
            'reg_lambda': np.logspace(-3,3, 10),
            'min_split_gain': np.linspace(.1,.8, 10),
            'subsample': np.linspace(.5,1, 10),
            'subsample_freq': [int(i) for i in np.linspace(5, 100, 10)]
        }
    else:
        param_grid = {
            'colsample_bytree': [.5, 0.7, 0.8],
            'num_leaves': [25, 50, 100, 150, 200],
            'reg_alpha': [1e-2, 1e-1, 1, 1e1],
            'reg_lambda': [1e-2, 1e-1, 1, 1e1],
            'min_split_gain': [0.2, 0.3, 0.4, 0.5],
            'subsample': [0.7, 0.8, 0.9],
            'subsample_freq': [5, 20, 50]
            }

    for i in range(n_iter):
        mod = None
        model_id = str(uuid.uuid4())
        params = {}
        for k,v in param_grid.items():
            params[k] = random.choice(v)
        mod = lgb.LGBMClassifier(random_state = 1108, n_estimators = 500, verbose=-1, is_unbalance = True)
        mod = mod.set_params(**params)
        score = cross_validate(x, y, mod, njobs = 1)
        score['id'] = model_id
        
        if post_feat:
            with open('./training/ml/%s/post_feat/scores/%s.json' % (domain, model_id), 'w') as f:
                json.dump(score, f)
            dump(mod, "./training/ml/%s/post_feat/models/%s.joblib" % (domain, model_id)) 
        else:
            with open('./training/ml/%s/pre_feat/scores/%s.json' % (domain, model_id), 'w') as f:
                json.dump(score, f)
            dump(mod, "./training/ml/%s/pre_feat/models/%s.joblib" % (domain, model_id)) 
        progress(i+1, n_iter)
                
def _dartTuning(x, y, post_feat = False, domain = 'strike', n_iter = 5000):
#    param_grid = {'num_leaves': sp_randint(6, 50), 
#             'min_child_samples': sp_randint(100, 500), 
#             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
#             'subsample': sp_uniform(loc=0.2, scale=0.8), 
#             'colsample_bytree': sp_uniform(loc=0.4, scale=0.8),
#             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
#             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
    
#    param_grid = {
#        'objective': 'binary',
#        'metric': 'binary_logloss',
#        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
#        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
#        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
#        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
#        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
#        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
#        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#    }
    if post_feat:
        param_grid = {
            'colsample_bytree': np.linspace(.3,.9, 10),
            'num_leaves': [int(i) for i in np.linspace(15, 500, 50)],
            'reg_alpha': np.logspace(-3,3, 10),
            'reg_lambda': np.logspace(-3,3, 10),
            'min_split_gain': np.linspace(.1,.8, 10),
            'subsample': np.linspace(.5,1, 10),
            'subsample_freq': [int(i) for i in np.linspace(5, 100, 10)]
        }
    else:
        param_grid = {
            'colsample_bytree': [.5, 0.7, 0.8],
            'num_leaves': [25, 50, 100, 150, 200],
            'reg_alpha': [1e-2, 1e-1, 1, 1e1],
            'reg_lambda': [1e-2, 1e-1, 1, 1e1],
            'min_split_gain': [0.2, 0.3, 0.4, 0.5],
            'subsample': [0.7, 0.8, 0.9],
            'subsample_freq': [5, 20, 50]
            }

    for i in range(n_iter):
        mod = None
        model_id = str(uuid.uuid4())
        params = {}
        for k,v in param_grid.items():
            params[k] = random.choice(v)
        mod = lgb.LGBMClassifier(boosting_type = 'dart', random_state = 1108, n_estimators = 500, verbose=-1, is_unbalance = True)
        mod = mod.set_params(**params)
        score = cross_validate(x, y, mod, njobs = 1)
        score['id'] = model_id
        
        if post_feat:
            with open('./training/ml/%s/post_feat/scores/%s.json' % (domain, model_id), 'w') as f:
                json.dump(score, f)
            dump(mod, "./training/ml/%s/post_feat/models/%s.joblib" % (domain, model_id)) 
        else:
            with open('./training/ml/%s/pre_feat/scores/%s.json' % (domain, model_id), 'w') as f:
                json.dump(score, f)
            dump(mod, "./training/ml/%s/pre_feat/models/%s.joblib" % (domain, model_id)) 
        progress(i+1, n_iter)            
    
#def _optimize_light_gbm():
#    study = optuna.create_study(direction='maximize')
#    study.optimize(objective, n_trials=100)
#    
    
def _opt_light(x, y, param, post_feat = False, domain = 'strike'):
    model_id = str(uuid.uuid4())
    mod = lgb.LGBMClassifier(random_state = 1108, n_estimators = 500, verbose = -1, is_unbalance = True, silent = True)
    mod = mod.set_params(**param)
    score = cross_validate(x, y, mod, njobs = 1)
    score['id'] = model_id
    
    if post_feat:
        with open('./training/ml/%s/post_feat/scores/%s.json' % (domain, model_id), 'w') as f:
            json.dump(score, f)
        dump(mod, "./training/ml/%s/post_feat/models/%s.joblib" % (domain, model_id)) 
    else:
        with open('./training/ml/%s/pre_feat/scores/%s.json' % (domain, model_id), 'w') as f:
            json.dump(score, f)
        dump(mod, "./training/ml/%s/pre_feat/models/%s.joblib" % (domain, model_id))    
    return score['logloss']

def _opt_dart(x, y, param, post_feat = False, domain = 'strike'):
    model_id = str(uuid.uuid4())
    mod = lgb.LGBMClassifier(boosting_type = 'dart', random_state = 1108, n_estimators = 500, verbose=-1, is_unbalance = True)
    mod = mod.set_params(**param)
    score = cross_validate(x, y, mod, njobs = 1)
    score['id'] = model_id
    
    if post_feat:
        with open('./training/ml/%s/post_feat/scores/%s.json' % (domain, model_id), 'w') as f:
            json.dump(score, f)
        dump(mod, "./training/ml/%s/post_feat/models/%s.joblib" % (domain, model_id)) 
    else:
        with open('./training/ml/%s/pre_feat/scores/%s.json' % (domain, model_id), 'w') as f:
            json.dump(score, f)
        dump(mod, "./training/ml/%s/pre_feat/models/%s.joblib" % (domain, model_id))    
    return score['logloss']

def _opt_light_all_post(trial):
    domain = 'all'
    X, Y = form_to_domain(domain = domain)
    red_feats = retrieve_reduced_domain_features(domain = domain)
    X = X[red_feats['features']]
    param = {
    'reg_alpha': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
    'reg_lambda': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
    'num_leaves': trial.suggest_int('num_leaves', 2, 256),
    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 1.0),
    'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
    'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
    'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
    }
    return _opt_light(X, Y, param, post_feat = True, domain = domain)

def _opt_dart_all_post(trial):
    domain = 'all'
    X, Y = form_to_domain(domain = domain)
    red_feats = retrieve_reduced_domain_features(domain = domain)
    X = X[red_feats['features']]
    param = {
    'reg_alpha': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
    'reg_lambda': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
    'num_leaves': trial.suggest_int('num_leaves', 2, 256),
    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 1.0),
    'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
    'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
    'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
    }
    return _opt_dart(X, Y, param, post_feat = True, domain = domain)

def _opt_light_strike_post(trial):
    domain = 'strike'
    X, Y = form_to_domain(domain = domain)
    red_feats = retrieve_reduced_domain_features(domain = domain)
    X = X[red_feats['features']]
    param = {
    'reg_alpha': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
    'reg_lambda': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
    'num_leaves': trial.suggest_int('num_leaves', 2, 256),
    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 1.0),
    'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
    'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
    'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
    }
    return _opt_light(X, Y, param, post_feat = True, domain = domain)

def _opt_dart_strike_post(trial):
    domain = 'strike'
    X, Y = form_to_domain(domain = domain)
    red_feats = retrieve_reduced_domain_features(domain = domain)
    X = X[red_feats['features']]
    param = {
    'reg_alpha': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
    'reg_lambda': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
    'num_leaves': trial.suggest_int('num_leaves', 2, 256),
    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 1.0),
    'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
    'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
    'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
    }
    return _opt_dart(X, Y, param, post_feat = True, domain = domain)

def _opt_light_grapp_post(trial):
    domain = 'grapp'
    X, Y = form_to_domain(domain = domain)
    red_feats = retrieve_reduced_domain_features(domain = domain)
    X = X[red_feats['features']]
    param = {
    'reg_alpha': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
    'reg_lambda': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
    'num_leaves': trial.suggest_int('num_leaves', 2, 256),
    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 1.0),
    'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
    'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
    'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
    }
    return _opt_light(X, Y, param, post_feat = True, domain = domain)

def _opt_dart_grapp_post(trial):
    domain = 'grapp'
    X, Y = form_to_domain(domain = domain)
    red_feats = retrieve_reduced_domain_features(domain = domain)
    X = X[red_feats['features']]
    param = {
    'reg_alpha': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
    'reg_lambda': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
    'num_leaves': trial.suggest_int('num_leaves', 2, 256),
    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 1.0),
    'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
    'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
    'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
    }
    return _opt_dart(X, Y, param, post_feat = True, domain = domain)

def _optimize(clf = 'light', domain = 'strike', trials = 1000):
    study = optuna.create_study(direction='maximize')
    if clf == 'light':
        if domain == 'strike':
            study.optimize(_opt_light_strike_post, n_trials=trials)
        elif domain == 'grapp':
            study.optimize(_opt_light_grapp_post, n_trials=trials)      
        elif domain == 'all':
            study.optimize(_opt_light_all_post, n_trials = trials)
    elif clf == 'dart':
        if domain == 'strike':
            study.optimize(_opt_dart_strike_post, n_trials=trials)
        elif domain == 'grapp':
            study.optimize(_opt_dart_grapp_post, n_trials=trials)      
        elif domain == 'all':
            study.optimize(_opt_dart_all_post, n_trials = trials)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
        
#    clf, post_feat, domain, refit, opt = 'log', True, 'all', False, False
def tune_ml(clf = 'log', post_feat = False, domain = 'strike', refit = False, opt = False, trials = 5000):
    if opt:
        _optimize(clf = clf, domain = domain, trials = trials)
    else:
        X, Y = form_to_domain(domain = domain)
        if post_feat:
            red_feats = retrieve_reduced_domain_features(domain = domain, refit = refit)
            X = X[red_feats['features']]
        
        if not post_feat and domain == 'all':
            X = X[[i for i in list(X) if i != 'date']]
        if clf == 'log':
            _logRegTuning(X, Y, post_feat = post_feat, domain = domain)
        elif clf == 'light':
            _lgbTuning(X, Y, post_feat = post_feat, domain = domain, n_iter = trials)
        elif clf == 'dart':
            _dartTuning(X, Y, post_feat = post_feat, domain = domain, n_iter = trials)
        