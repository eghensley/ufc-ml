#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:30:48 2020

@author: ehens86
"""

from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed, load, dump
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import json
from scipy.stats import percentileofscore
from sklearn.metrics import log_loss, f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from utils.general import reduce_mem_usage
from common import raw_features
from data import pull_ml_training_corpus
from training.create_elo_training_set import pull_ml_training

#    x, y, domain = X, Y, domain 
def _feat_reduction(x, y, domain = 'strike', refit = False):
    print('Calculating best features')
    feat_red_model = retrieve_best_model(post_feat = False, domain = domain, refit = refit)
    feat_red_model.fit(x, y)

    if hasattr(feat_red_model, 'feature_importances_'):
        cv_varimp_df = pd.DataFrame([list(x), feat_red_model.feature_importances_]).T
    elif hasattr(feat_red_model, 'coef_'):
        cv_varimp_df = pd.DataFrame([list(x), feat_red_model.coef_[0]]).T
    
    cv_varimp_df.columns = ['feature_name', 'varimp']
    cv_varimp_df.sort_values(by='varimp', ascending=False, inplace=True)    
    
    feature_search = RFECV(estimator=feat_red_model, step=1, cv=StratifiedKFold(10, random_state=0),
                          scoring='neg_log_loss', n_jobs = 1)
    feature_search.fit(x, y)
    print("Optimal number of features : %d (%.4f)" % (feature_search.n_features_, feature_search.grid_scores_[feature_search.n_features_-1]))
    print('Best features :', x.columns[feature_search.support_])   
    
    best_feats = {'features': list(x.columns[feature_search.support_])}
    with open('src/predictors/%s/opt.json' % (domain), 'w') as f:
        json.dump(best_feats, f)
        
def retrieve_reduced_domain_features(domain = 'strike', refit = False):
    try:
        print('retrieve_reduced_domain_features - locating reduced features')
        if not os.path.exists('src/predictors/%s/opt.json' % (domain)) or refit:
            print('retrieve_reduced_domain_features - reduced features not found... adding')
            X, Y = form_to_domain(domain = domain)
            _feat_reduction(X, Y, domain = domain, refit = refit)
        print('retrieve_reduced_domain_features - successfully located reduced features')
        print('retrieve_reduced_domain_features - loading reduced features')
        with open('src/predictors/%s/opt.json' % (domain), 'r') as f:
            red_feats = json.load(f)    
        print('retrieve_reduced_domain_features - successfully loaded reduced features')
        return red_feats
    except Exception as e:
        print('retrieve_reduced_domain_features failed with %s' % (e))
        raise e

def form_to_domain(domain = 'strike'):
    if domain == 'all':
        df = pull_ml_training()
        df = reduce_mem_usage(df)
        df = df.sample(frac=1, random_state = 1108)
        x = df[[i for i in list(df) if i != 'winner' and i != 'date']]
        y = df['winner']
        return x, y
    else:
        df = pull_ml_training_corpus()
        df = reduce_mem_usage(df)
        df = df.sample(frac=1, random_state = 1108)
        if domain == 'strike':
            df = df[df['round_1'] == 1]
            x = df[raw_features['features']['strike']['x']]
            y = df[raw_features['features']['strike']['y']]
            return x, y
        elif domain == 'grapp':
            df = df[(df['round_2'] == 1) | (df['round_3'] == 1)]
            x = df[raw_features['features']['grapp']['x']]
            y = df[raw_features['features']['grapp']['y']]
            return x, y    

def retrieve_best_model(domain = 'strike', post_feat = False, refit = False):
    if post_feat:
        feat = 'post'
    else:
        feat = 'pre'
    if not os.path.exists('src/predictors/%s/%s_feat/model.joblib' % (domain, feat)) or refit:
        add_best_model(domain = domain, post_feat = post_feat)
    mod = load('src/predictors/%s/%s_feat/model.joblib' % (domain, feat))
    return mod

def add_best_model(domain = 'strike', post_feat = False):
    if post_feat:
        feat = 'post'
    else:
        feat = 'pre'
    mypath = 'src/training/ml/%s/%s_feat/scores' % (domain, feat)
    
    
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    
    all_scores = {'logloss': [],
     'f1': [],
     'roc': [],
     'acc': []
     }
    
    for file in onlyfiles:
        with open('%s/%s' % (mypath, file), 'r') as f:
            score = json.load(f)
        for ft in ['logloss', 'f1', 'roc', 'acc']:
            all_scores[ft].append(score[ft])
#    all_scores_df = pd.DataFrame.from_dict(all_scores)
    
    ranked_scores = {}
    for file in onlyfiles:
        with open('%s/%s' % (mypath, file), 'r') as f:
            score = json.load(f)
        idx = score['id']
        score.pop('id')
        for ft in ['logloss', 'f1', 'roc', 'acc']:
            score[ft] = percentileofscore(all_scores[ft], score[ft], 'rank')
        ranked_scores[idx] = score
    
    ranked_score_df = pd.DataFrame.from_dict(ranked_scores).T
    ranked_score_df['tot'] = ranked_score_df['logloss'] + ranked_score_df['f1'] + ranked_score_df['roc'] + ranked_score_df['acc']
    ranked_score_df.sort_values(by=['tot'], ascending = False, inplace = True)
    best_model_id = ranked_score_df.index[0]
    best_model = load('src/training/ml/%s/%s_feat/models/%s.joblib' % (domain, feat, best_model_id))
    
    dump(best_model, 'src/predictors/%s/%s_feat/model.joblib' % (domain, feat))
    return best_model

def _single_core_scorer(input_vals):
#   trainx, testx, trainy, testy, model = job
    trainx, testx, trainy, testy, model = input_vals   
    
#    test_weights = class_weight.compute_class_weight('balanced',
#                                np.unique(trainy),trainy)    
#    test_weights_dict = {i:j for i,j in zip(np.unique(trainy), test_weights)}    
        
    model.fit(trainx, trainy)        
    pred = model.predict_proba(testx)
    scores = {}
    scores['logloss'] = log_loss(testy, pred) * -1
    bin_pred = [0 if i[0] > i[1] else 1 for i in pred]
    scores['f1'] = f1_score(testy, bin_pred)
    scores['roc'] = roc_auc_score(testy, bin_pred)
    scores['acc'] = accuracy_score(testy, bin_pred)
    
    return(scores)
    
def cross_validate(x,y,est, only_scores = True, njobs = -1, verbose = False): 
#    x,y, est = X,Y, mod
    scale = StandardScaler()
    splitter = StratifiedKFold(n_splits = 10, random_state = 8686)
    all_folds = []
    for fold in splitter.split(x, y):
        all_folds.append(fold) 
    jobs = []
    for train, val in all_folds:
        scale.fit(x.iloc[train])
        x_train = pd.DataFrame(scale.transform(x.iloc[train]))
        x_train.set_index(y.iloc[train].index, inplace = True)
        x_train.rename(columns = {i:j for i,j in zip(list(x_train), list(x))}, inplace = True)
        
        x_val = pd.DataFrame(scale.transform(x.iloc[val]))
        x_val.set_index(y.iloc[val].index, inplace = True)
        x_val.rename(columns = {i:j for i,j in zip(list(x_val), list(x))}, inplace = True)       
        
        jobs.append([x_train, x_val, y.iloc[train], y.iloc[val], est])   
        
    if njobs == -1:
        scores = Parallel(n_jobs = njobs, verbose = 25)(delayed(_single_core_scorer) (i) for i in jobs)
    else:
        scores = [_single_core_scorer(i) for i in jobs]
    cv_score = {"logloss": 0, "f1": 0, "roc":0, "acc": 0}
    for score in scores:
        for k,v in score.items():
            cv_score[k] += v
    for kk,vv in cv_score.items():
        cv_score[kk] /= len(scores)    
    return cv_score
