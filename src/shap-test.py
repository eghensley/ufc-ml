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
line_data = {'name': 'TEAM A', 'type': 'line', 'data': []}
area_data = {'name': 'TEAM A', 'type': 'area', 'data': []}

conf = 0


label_translator: {'age_1': 'blue corner age',
 'age_2': 'red corner age',
 'age_diff': 'blue - red age',
 'def_ko_1': 'red corner chin',
 'def_ko_2': 'blue corner chin',
 'def_ko_diff': 'blue - red chin',
 'def_ko_share': 'blue / (blue + red) chin',
 'def_strike_1': 'blue strike defense',
 'def_strike_2': 'red strike defense',
 'def_strike_diff',
 'def_strike_share',
 'eff_grap_1',
 'eff_grap_2',
 'eff_grap_diff',
 'eff_grap_share',
 'eff_ko_2',
 'eff_strike_1',
 'eff_strike_2',
 'eff_strike_diff',
 'eff_strike_share',
 'off_grapp_1',
 'off_grapp_2',
 'off_grapp_diff',
 'off_grapp_share',
 'off_ko_1',
 'off_ko_diff',
 'off_ko_share',
 'off_strike_1',
 'off_strike_2',
 'off_strike_diff',
 'off_strike_share',
 'prev_fights_1',
 'prev_fights_2',
 'rounds']




[x for _,x in sorted(zip([abs(i) for i in shap_values[1][0]], list(new_row)))]

sorted_indexes = [x for _,x in sorted(zip([abs(i) for i in shap_values[1][0]], range(len(shap_values[1][0]))))]


top_10 = sorted_indexes[-10:]

for idx in sorted_indexes[:-10]:
    i = shap_values[1][0][idx]
    if (i != 0) :
        new_conf = conf + ((1/(1 + np.exp(-1*(i)))) - .5)
    else:
        new_conf = conf
    conf = new_conf
line_data['data'].append(conf)
area_data['data'].append(conf)

for top_idx in top_10:
    i = shap_values[1][0][top_idx]
    j = list(new_row)[top_idx]
    effects[j] = i
    
    if (i != 0) :
        effect = (1/(1 + np.exp(-1*(i)))) - .5
    else:
        effect = 0

    line_data['data'].append(effect)
    conf += effect
    area_data['data'].append(conf)        
    
    if (conf >= new_conf):
        row['y'].append(new_conf)
        row['y'].append(conf)
    else:
        row['y'].append(conf)
        row['y'].append(new_conf)
        
    data.append(row)
    conf = new_conf

