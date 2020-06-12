#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:12:21 2020

@author: ehens86
"""

import sys
if __name__ == "__main__":
    sys.path.append("..")
from data import pull_ml_training_corpus
from training.training_utils import retrieve_best_model
from training.ml_model_tuning import form_to_domain, retrieve_reduced_domain_features
from utils.general import reduce_mem_usage, progress
from sklearn.preprocessing import StandardScaler
from spring.api_wrappers import saveMlScore

#    model, idx, train_x_univ, train_y_univ, test_x, feats = retrieve_best_model(post_feat = True, domain = 'strike'), idx, strike_x, strike_y, df.loc[idx], strike_features
def generate_training_score(model, idx, train_x_univ, train_y_univ, test_x, feats):
    train_x = train_x_univ.loc[train_x_univ.index != idx][feats]

    scale = StandardScaler()
    scale.fit(train_x)
    train_x = scale.transform(train_x)
    
    train_y = train_y_univ.loc[train_y_univ.index != idx]
    
    model.fit(train_x, train_y)

    scaled_test_x = scale.transform(test_x[feats].ravel().reshape(1,-1))
    score = model.predict_proba(scaled_test_x)[0][1]

    return score
  
def fill_ml_training_scores():
    print('Populating DB rounds with scores from best available models')
    df = pull_ml_training_corpus()
    df = reduce_mem_usage(df)
    
    strike_features = retrieve_reduced_domain_features(domain = 'strike')['features']
    strike_x, strike_y = form_to_domain(domain = 'strike')
    strike_x = strike_x[strike_features]
    
    grapp_features = retrieve_reduced_domain_features(domain = 'grapp')['features']
    grapp_x, grapp_y = form_to_domain(domain = 'grapp')
    grapp_x = grapp_x[grapp_features]
    
    tot = len(df.index)
    for n, (idx) in enumerate(df.index):
        ko_score = generate_training_score(retrieve_best_model(post_feat = True, domain = 'strike'), idx, strike_x, strike_y, df.loc[idx], strike_features)
        grapp_score = generate_training_score(retrieve_best_model(post_feat = True, domain = 'grapp'), idx, grapp_x, grapp_y, df.loc[idx], grapp_features)
        payload = {"oid": idx, "koScore": ko_score, "subScore": grapp_score}
        resp = saveMlScore(payload)
        if resp['errorMsg'] is not None:
            print(resp['errorMsg'])
        progress(n+1, tot)
    




#    
#domain = 'strike'
#X, Y = form_to_domain(domain = domain)
#if not os.path.exists('training/ml/%s/features/opt.json' % (domain)):
#    _feat_reduction(X, Y, domain = domain)
#with open('training/ml/%s/features/opt.json' % (domain), 'r') as f:
#    red_feats = json.load(f)
#X = X[red_feats['features']]
#
#
#
#from data.generate import pull_year_raw_training_data, pull_bout_data
#from spring.api_wrappers import saveMlScore
#from .feature_utils import load_features
#from .generate_models import retrieve_ko_model, retrieve_sub_model
#
#def generate_training_ko_score(idx, ko_test, KO_MODEL, KO_DATA, KO_FEATURES):
#    # ko_test = test
#    train_x = KO_DATA.loc[KO_DATA.index != idx][KO_FEATURES]
#    train_y = KO_DATA.loc[KO_DATA.index != idx]['offTkoKo']
#    KO_MODEL.fit(train_x, train_y)
#    ko_score = KO_MODEL.predict_proba(np.array(ko_test[KO_FEATURES]).reshape(1,-1))[0][1]
#    return ko_score
#
#def generate_training_sub_score(idx, sub_test, SUB_MODEL, SUB_DATA, SUB_FEATURES):
#    # sub_test = test
#    train_x = SUB_DATA.loc[SUB_DATA.index != idx][SUB_FEATURES]
#    train_y = SUB_DATA.loc[SUB_DATA.index != idx]['offSubmissionSuccessful']
#    SUB_MODEL.fit(train_x, train_y)
#    sub_score = SUB_MODEL.predict_proba(np.array(sub_test[SUB_FEATURES]).reshape(1,-1))[0][1]
#    return sub_score
#
#def pop_training_ml_scores():
#    KO_FEATURES = load_features('ml_ko_features')
#    SUB_FEATURES = load_features('ml_sub_features')
#
#    KO_MODEL = LogisticRegression(class_weight='balanced', solver = 'newton-cg', C = 100, max_iter = 50000)
#    SUB_MODEL = LogisticRegression(class_weight='balanced', solver = 'newton-cg', C = 100, max_iter = 10000)
#
#    DATA = pd.DataFrame()
#    for year in range(2005, 2020):
#        year_data = pull_year_raw_training_data(year, score = 'finish', add_gender = True, add_class = True)
#        DATA = DATA.append(year_data)
#        
#    DATA = DATA[DATA['gender'] == 'M']
#    KO_DATA = DATA[DATA['round'] == 1]
#    SUB_DATA = DATA[(DATA['round'] == 2) | (DATA['round'] ==3)]
#
#    for idx in DATA.index:
#        test = DATA.loc[idx]
#        ko_score = generate_training_ko_score(idx, test, KO_MODEL, KO_DATA, KO_FEATURES)
#        sub_score = generate_training_sub_score(idx, test, SUB_MODEL, SUB_DATA, SUB_FEATURES)
#        payload = {"oid": idx, "koScore": ko_score, "subScore": sub_score}
#        resp = saveMlScore(payload)
#        if resp['errorMsg'] is not None:
#            print(resp['errorMsg'])
#        
#def generate_new_ko_score(data):
#    ko_model = retrieve_ko_model()
#    ko_score = ko_model.predict_proba(np.array(data[load_features('ml_ko_features')]).reshape(1,-1))[0][1]
#    return ko_score
#
#def generate_new_sub_score(data):
#    sub_model = retrieve_sub_model()
#    sub_score = sub_model.predict_proba(np.array(data[load_features('ml_sub_features')]).reshape(1,-1))[0][1]
#    return sub_score
#        
##fight_id, bout_id = '4834ff149dc9542a', 'eed6c9aff2234b7a'
#def insert_new_ml_scores(bout_id):
#    round_data = pull_bout_data(bout_id)
#    for idx in round_data.index:
#        ko_score = generate_new_ko_score(round_data.loc[idx])
#        sub_score = generate_new_sub_score(round_data.loc[idx])
#        payload = {"oid": idx, "koScore": ko_score, "subScore": sub_score}
#        resp = saveMlScore(payload) 
#        if resp['errorMsg'] is not None:
#            print(resp['errorMsg'])
#    
        
        
#from sklearn.linear_model import LogisticRegression
#import sys
#if __name__ == "__main__":
#    sys.path.append("..")
#import pandas as pd
#import numpy as np
#from data.generate import pull_year_raw_training_data, pull_bout_data
#from spring.api_wrappers import saveMlScore
#from .feature_utils import load_features
#from .generate_models import retrieve_ko_model, retrieve_sub_model
#
#def generate_training_ko_score(idx, ko_test, KO_MODEL, KO_DATA, KO_FEATURES):
#    # ko_test = test
#    train_x = KO_DATA.loc[KO_DATA.index != idx][KO_FEATURES]
#    train_y = KO_DATA.loc[KO_DATA.index != idx]['offTkoKo']
#    KO_MODEL.fit(train_x, train_y)
#    ko_score = KO_MODEL.predict_proba(np.array(ko_test[KO_FEATURES]).reshape(1,-1))[0][1]
#    return ko_score
#
#def generate_training_sub_score(idx, sub_test, SUB_MODEL, SUB_DATA, SUB_FEATURES):
#    # sub_test = test
#    train_x = SUB_DATA.loc[SUB_DATA.index != idx][SUB_FEATURES]
#    train_y = SUB_DATA.loc[SUB_DATA.index != idx]['offSubmissionSuccessful']
#    SUB_MODEL.fit(train_x, train_y)
#    sub_score = SUB_MODEL.predict_proba(np.array(sub_test[SUB_FEATURES]).reshape(1,-1))[0][1]
#    return sub_score
#
#def pop_training_ml_scores():
#    KO_FEATURES = load_features('ml_ko_features')
#    SUB_FEATURES = load_features('ml_sub_features')
#
#    KO_MODEL = LogisticRegression(class_weight='balanced', solver = 'newton-cg', C = 100, max_iter = 50000)
#    SUB_MODEL = LogisticRegression(class_weight='balanced', solver = 'newton-cg', C = 100, max_iter = 10000)
#
#    DATA = pd.DataFrame()
#    for year in range(2005, 2020):
#        year_data = pull_year_raw_training_data(year, score = 'finish', add_gender = True, add_class = True)
#        DATA = DATA.append(year_data)
#        
#    DATA = DATA[DATA['gender'] == 'M']
#    KO_DATA = DATA[DATA['round'] == 1]
#    SUB_DATA = DATA[(DATA['round'] == 2) | (DATA['round'] ==3)]
#
#    for idx in DATA.index:
#        test = DATA.loc[idx]
#        ko_score = generate_training_ko_score(idx, test, KO_MODEL, KO_DATA, KO_FEATURES)
#        sub_score = generate_training_sub_score(idx, test, SUB_MODEL, SUB_DATA, SUB_FEATURES)
#        payload = {"oid": idx, "koScore": ko_score, "subScore": sub_score}
#        resp = saveMlScore(payload)
#        if resp['errorMsg'] is not None:
#            print(resp['errorMsg'])
#        
#def generate_new_ko_score(data):
#    ko_model = retrieve_ko_model()
#    ko_score = ko_model.predict_proba(np.array(data[load_features('ml_ko_features')]).reshape(1,-1))[0][1]
#    return ko_score
#
#def generate_new_sub_score(data):
#    sub_model = retrieve_sub_model()
#    sub_score = sub_model.predict_proba(np.array(data[load_features('ml_sub_features')]).reshape(1,-1))[0][1]
#    return sub_score
#        
##fight_id, bout_id = '4834ff149dc9542a', 'eed6c9aff2234b7a'
#def insert_new_ml_scores(bout_id):
#    round_data = pull_bout_data(bout_id)
#    for idx in round_data.index:
#        ko_score = generate_new_ko_score(round_data.loc[idx])
#        sub_score = generate_new_sub_score(round_data.loc[idx])
#        payload = {"oid": idx, "koScore": ko_score, "subScore": sub_score}
#        resp = saveMlScore(payload) 
#        if resp['errorMsg'] is not None:
#            print(resp['errorMsg'])