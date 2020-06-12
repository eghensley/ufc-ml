#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:27:07 2020

@author: ehens86
"""

from joblib import load, dump
import pandas as pd

import os
from os import listdir
from os.path import isfile, join
import json
from scipy.stats import percentileofscore
from training import form_to_domain, retrieve_reduced_domain_features, form_new_ml_odds_data
from sklearn.preprocessing import StandardScaler
import numpy as np
from data import pull_bout_data
from spring.api_wrappers import saveMlScore, saveBoutMlScore, refreshBout

def retrieve_best_trained_models(domain = 'strike', refit = False):
    if not os.path.exists('predictors/%s/post_feat/trained_model.joblib' % (domain)) or refit:
        add_best_trained_model(domain = domain)
    mod = load('predictors/%s/post_feat/trained_model.joblib' % (domain))
    scale = load('predictors/%s/post_feat/trained_scale.joblib' % (domain))
    return mod, scale

def add_best_trained_model(domain = 'strike'):

    mypath = 'training/ml/%s/post_feat/scores' % (domain)
    
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
        if score['logloss'] < -.69:
            continue
        idx = score['id']
        score.pop('id')
        for ft in ['logloss', 'f1', 'roc', 'acc']:
            score[ft] = percentileofscore(all_scores[ft], score[ft], 'rank')
        ranked_scores[idx] = score
    
    ranked_score_df = pd.DataFrame.from_dict(ranked_scores).T
    ranked_score_df['tot'] = ranked_score_df['logloss'] + ranked_score_df['f1'] + ranked_score_df['roc'] + ranked_score_df['acc']
    ranked_score_df.sort_values(by=['tot'], ascending = False, inplace = True)
    best_model_id = ranked_score_df.index[0]
    best_model = load('training/ml/%s/post_feat/models/%s.joblib' % (domain, best_model_id))
    _train_model(best_model, domain = domain)
    dump(best_model, 'predictors/%s/post_feat/trained_model.joblib' % (domain))
    return best_model

#    model, domain = best_model, 'strike'
def _train_model(model, domain = 'strike'):
    scale = StandardScaler()
    X, Y = form_to_domain(domain = domain)
    red_feats = retrieve_reduced_domain_features(domain = domain)
    X = X[red_feats['features']]
    scale.fit(X)
    x_train = pd.DataFrame(scale.transform(X))
    x_train.set_index(Y.index, inplace = True)
    x_train.rename(columns = {i:j for i,j in zip(list(x_train), list(X))}, inplace = True)
    
    dump(scale, 'predictors/%s/post_feat/trained_scale.joblib' % (domain))
    
    model.fit(x_train, Y)
    
    return model
    

#    data = round_data.loc[idx]
def generate_new_ko_score(data):
    ko_model, ko_scale = retrieve_best_trained_models(domain = 'strike')
    red_feats = retrieve_reduced_domain_features(domain = 'strike')
    ko_score = ko_model.predict_proba(ko_scale.transform(np.array(data[red_feats['features']]).reshape(1,-1)))[0][1]
    return ko_score

def generate_new_sub_score(data):
    sub_model, sub_scale = retrieve_best_trained_models(domain = 'grapp')
    red_feats = retrieve_reduced_domain_features(domain = 'grapp')
    sub_score = sub_model.predict_proba(sub_scale.transform(np.array(data[red_feats['features']]).reshape(1,-1)))[0][1]
    return sub_score

def generate_new_win_score(data):
    sub_model, sub_scale = retrieve_best_trained_models(domain = 'all')
    red_feats = retrieve_reduced_domain_features(domain = 'all')
    sub_score = sub_model.predict_proba(sub_scale.transform(np.array(data[red_feats['features']]).reshape(1,-1)))[0][1]
    return sub_score

#   fight_id, bout_id = '4834ff149dc9542a', '4eff0432bd364a23'
def insert_new_ml_scores(bout_id):
    round_data = pull_bout_data(bout_id)
    for idx in round_data.index:
        if round_data.loc[idx]['gender'] != 'M':
            continue
        ko_score = generate_new_ko_score(round_data.loc[idx])
        sub_score = generate_new_sub_score(round_data.loc[idx])
        payload = {"oid": idx, "koScore": ko_score, "subScore": sub_score}
        resp = saveMlScore(payload) 
        if resp['errorMsg'] is not None:
            print(resp['errorMsg'])
            
#    bout_id = '0170db5fad04e4d0'
def insert_new_ml_prob(bout_id):
    bout_detail = refreshBout(bout_id)
    
    bout_data = form_new_ml_odds_data(bout_id)
    
    if bout_data is None:
        return
    else:
        win_prob = generate_new_win_score(bout_data)
    
        payload_1 = {"oid": bout_detail['fighterBoutXRefs'][0]['oid'], "expOdds": win_prob}
        payload_2 = {"oid": bout_detail['fighterBoutXRefs'][1]['oid'], "expOdds": 1-win_prob}
    
        resp_1 = saveBoutMlScore(payload_1) 
        if resp_1['errorMsg'] is not None:
            print(resp_1['errorMsg'])
    
        resp_2 = saveBoutMlScore(payload_2) 
        if resp_2['errorMsg'] is not None:
            print(resp_2['errorMsg'])
                        