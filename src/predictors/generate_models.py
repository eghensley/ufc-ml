#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:27:07 2020

@author: ehens86
"""

import sys, os
if __name__ == "__main__":
    sys.path.append("src")
    os.environ['ufc.flask.spring.host'] = 'http://localhost:4646'
    os.environ['ufc.flask.spring.pw'] = '1234'

    print(os.environ)
    
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
    try:
        print('retrieve_best_trained_models - locating best model')
        if not os.path.exists('src/predictors/%s/post_feat/trained_model.joblib' % (domain)) or refit:
            print('retrieve_best_trained_models - model not found... adding')
            add_best_trained_model(domain = domain)
        print('retrieve_best_trained_models - successfully located best model')
        
        print('retrieve_best_trained_models - loading best model')
        mod = load('src/predictors/%s/post_feat/trained_model.joblib' % (domain))
        print('retrieve_best_trained_models - successfully loaded best model')
        print('retrieve_best_trained_models - loading best scale')
        scale = load('src/predictors/%s/post_feat/trained_scale.joblib' % (domain))
        print('retrieve_best_trained_models - successfully loaded best scale')
        return mod, scale
    except Exception as e:
        print('retrieve_best_trained_models failed with %s' % (e))
        raise e
        
def add_best_trained_model(domain = 'strike'):

    mypath = 'src/training/ml/%s/post_feat/scores' % (domain)
    
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
    best_model = load('src/training/ml/%s/post_feat/models/%s.joblib' % (domain, best_model_id))
    _train_model(best_model, domain = domain)
    dump(best_model, 'src/predictors/%s/post_feat/trained_model.joblib' % (domain))
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
    
    dump(scale, 'src/predictors/%s/post_feat/trained_scale.joblib' % (domain))
    
    model.fit(x_train, Y)
    
    return model
    

#    data = round_data.loc[idx]
def generate_new_ko_score(data):
    try:
        print('generate_new_ko_score - retrieving best model')
        ko_model, ko_scale = retrieve_best_trained_models(domain = 'strike')
        print('generate_new_ko_score - successfully retrieved best model')
    
        print('generate_new_ko_score - retrieving domain features')
        red_feats = retrieve_reduced_domain_features(domain = 'strike')
        print('generate_new_ko_score - successfully retrieved domain features')
        
        print('generate_new_ko_score - generating new score')
        ko_score = ko_model.predict_proba(ko_scale.transform(np.array(data[red_feats['features']]).reshape(1,-1)))[0][1]
        print('generate_new_ko_score - successfully generated new score')
    
        return ko_score
    except Exception as e:
        print('generate_new_ko_score failed with %s' % (e))
        raise e


def generate_new_sub_score(data):
    try:
        print('generate_new_sub_score - retrieving best model')
        sub_model, sub_scale = retrieve_best_trained_models(domain = 'grapp')
        print('generate_new_sub_score - successfully retrieved best model')
    
        print('generate_new_sub_score - retrieving domain features')
        red_feats = retrieve_reduced_domain_features(domain = 'grapp')
        print('generate_new_sub_score - successfully retrieved domain features')
    
        print('generate_new_sub_score - generating new score')
        sub_score = sub_model.predict_proba(sub_scale.transform(np.array(data[red_feats['features']]).reshape(1,-1)))[0][1]
        print('generate_new_sub_score - successfully generated new score')
    
        return sub_score
    except Exception as e:
        print('generate_new_sub_score failed with %s' % (e))
        raise e

def generate_new_win_score(data):
    try:
        print('generate_new_win_score - retrieving best model')
        sub_model, sub_scale = retrieve_best_trained_models(domain = 'all')
        print('generate_new_win_score - successfully retrieved best model')
    
        print('generate_new_win_score - retrieving domain features')
        red_feats = retrieve_reduced_domain_features(domain = 'all')
        print('generate_new_win_score - successfully retrieved domain features')
    
        print('generate_new_win_score - generating new score')
        sub_score = sub_model.predict_proba(sub_scale.transform(np.array(data[red_feats['features']]).reshape(1,-1)))[0][1]
        print('generate_new_win_score - successfully generated new score')
    
        return sub_score
    except Exception as e:
        print('generate_new_win_score failed with %s' % (e))
        raise e

#   bout_id = '26173f6491300eaa'
def insert_new_ml_scores(bout_id):
    try:
        print('insert_new_ml_scores - pulling bout data')
        round_data = pull_bout_data(bout_id)
        print('insert_new_ml_scores - successfully pulled bout data')
        for idx in round_data.index:
            print('insert_new_ml_scores - generating ko score')
            ko_score = generate_new_ko_score(round_data.loc[idx])
            print('insert_new_ml_scores - successfully generated ko score')
    
            print('insert_new_ml_scores - generating sub score')
            sub_score = generate_new_sub_score(round_data.loc[idx])
            print('insert_new_ml_scores - successfully generated sub score')
    
            payload = {"oid": idx, "koScore": ko_score, "subScore": sub_score}
            resp = saveMlScore(payload) 
            if resp['errorMsg'] is not None:
                print(resp['errorMsg'])
    except Exception as e:
        print('insert_new_ml_scores failed with %s' % (e))
        raise e
        
#    bout_id = '054794810817c1cc'
def insert_new_ml_prob(bout_id):
    try:
        bout_detail = refreshBout(bout_id)
        print('insert_new_ml_prob - forming new bout data')
        bout_data = form_new_ml_odds_data(bout_id)
        print('insert_new_ml_prob - bout data successfully formed')
        if bout_data is None:
            return
        else:
            print('insert_new_ml_prob - generating win probability')
            win_prob = generate_new_win_score(bout_data)
            print('insert_new_ml_prob - successfully generated win probability')
            
            payload_1 = {"oid": bout_detail['fighterBoutXRefs'][0]['oid'], "expOdds": win_prob}
            payload_2 = {"oid": bout_detail['fighterBoutXRefs'][1]['oid'], "expOdds": 1-win_prob}
        
            resp_1 = saveBoutMlScore(payload_1) 
            if resp_1['errorMsg'] is not None:
                print(resp_1['errorMsg'])
                raise Exception(resp_1['errorMsg'])
        
            resp_2 = saveBoutMlScore(payload_2) 
            if resp_2['errorMsg'] is not None:
                print(resp_2['errorMsg'])
                raise Exception(resp_1['errorMsg'])

            print('insert_new_ml_prob - successfully added win probability')

    except Exception as e:
        print('insert_new_ml_prob failed with %s' % (e))
        raise e
        