#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:18:06 2020

@author: ehens86
"""

import sys, os
if __name__ == "__main__":
    sys.path.append("src")
    os.environ['ufc.flask.spring.host'] = 'http://localhost:4646'
    os.environ['ufc.flask.spring.pw'] = '1234'
    
import shap
from joblib import load
import pandas as pd
from training import retrieve_reduced_domain_features, form_new_ml_odds_data
from spring.api_wrappers import refreshBout
import numpy as np

domain = 'all'

print('retrieve_best_trained_models - loading best model')
mod = load('src/predictors/%s/post_feat/trained_model.joblib' % (domain))
print('retrieve_best_trained_models - successfully loaded best model')
print('retrieve_best_trained_models - loading best scale')
scale = load('src/predictors/%s/post_feat/trained_scale.joblib' % (domain))
print('retrieve_best_trained_models - successfully loaded best scale')
        
red_feats = retrieve_reduced_domain_features(domain = domain)

bout_id = '57939328c1214107'
bout_detail = refreshBout(bout_id)
print('insert_new_ml_prob - forming new bout data')
bout_data = form_new_ml_odds_data(bout_id)[red_feats['features']]
        
new_row = pd.DataFrame(scale.transform(np.array(bout_data[red_feats['features']]).reshape(1,-1))[0]).T
new_row.set_index(pd.Index([bout_id]), inplace= True)
new_row.rename(columns = {i:j for i,j in zip(list(new_row), list(bout_data))}, inplace = True)     
        
explainer = shap.TreeExplainer(mod)
shap_values = explainer.shap_values(new_row)

effects = {}
data = []


conf = .5


[x for _,x in sorted(zip([abs(i) for i in shap_values[1][0]], list(new_row)))]

sorted_indexes = [x for _,x in sorted(zip([abs(i) for i in shap_values[1][0]], range(len(shap_values[1][0]))))]


top_10 = sorted_indexes[-10:]

[list(new_row)[i] for i in top_10]



row = {'x': '24 other features', 'y': []}
for idx in sorted_indexes[:-10]:
    i = shap_values[1][0][idx]
    if (i != 0) :
        new_conf = conf + ((1/(1 + np.exp(-1*(i)))) - .5)
    else:
        new_conf = conf
    conf = new_conf
if (conf >= new_conf):
    row['y'].append(new_conf)
    row['y'].append(conf)
else:
    row['y'].append(conf)
    row['y'].append(new_conf)
data.append(row)

for top_idx in top_10:
    i = shap_values[1][0][top_idx]
    j = list(new_row)[top_idx]
    effects[j] = i
    
    row = {'x': j, 'y': []}
    if (i != 0) :
        new_conf = conf + ((1/(1 + np.exp(-1*(i)))) - .5)
    else:
        new_conf = conf
        
    if (conf >= new_conf):
        row['y'].append(new_conf)
        row['y'].append(conf)
    else:
        row['y'].append(conf)
        row['y'].append(new_conf)
        
    data.append(row)
    conf = new_conf



effects = {}
line_data = {'name': 'Feature Impact', 'type': 'line', 'data': []}
area_data = {'name': 'Cumulative Confidence', 'type': 'area', 'data': []}

conf = 0


label_translator = {'age_1': 'BC Age',
 'age_2': 'RC Age',
 'age_diff': 'Age DIFF',
 'def_ko_1': 'RC Chin',
 'def_ko_2': 'BC Chin',
 'def_ko_diff': 'Chin DIFF',
 'def_ko_share': 'Chin SHARE',
 'def_strike_1': 'BC Striking Def',
 'def_strike_2': 'RC Striking Def',
 'def_strike_diff': 'Striking Def DIFF',
 'def_strike_share': 'Striking Def SHARE',
 'eff_grap_1': 'BC Grappling EFF',
 'eff_grap_2': 'RC Grappling EFF',
 'eff_grap_diff': 'Grappling EFF DIFF',
 'eff_grap_share': 'Grappling EFF SHARE',
 'eff_ko_2': 'RC KO EFF',
 'eff_strike_1': 'BC Striking EFF',
 'eff_strike_2': 'RC Striking EFF',
 'eff_strike_diff': 'Striking EFF DIFF',
 'eff_strike_share': 'Striking EFF SHARE',
 'off_grapp_1': 'BC Grappling Off',
 'off_grapp_2': 'RC Grappling Off',
 'off_grapp_diff': 'Grappling Off DIFF',
 'off_grapp_share': 'Grappling Off SHARE',
 'off_ko_1': 'BC KO',
 'off_ko_diff': 'KO DIFF',
 'off_ko_share': 'KO SHARE',
 'off_strike_1': 'BC Strike Off',
 'off_strike_2': 'RC Strike Off',
 'off_strike_diff': 'Strike Off DIFF',
 'off_strike_share': 'Strike Off SHARE',
 'prev_fights_1': 'BC Num Fights',
 'prev_fights_2': 'RC Num Fights',
 'rounds': 'Rounds'
}




[x for _,x in sorted(zip([abs(i) for i in shap_values[1][0]], list(new_row)))]



