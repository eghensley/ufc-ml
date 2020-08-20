#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 14:26:48 2020

@author: eric.hensleyibm.com
"""

import sys, os
if __name__ == "__main__":
    sys.path.append("src")
    os.environ['ufc.flask.spring.host'] = 'http://207.237.93.29:4646'
    os.environ['ufc.flask.spring.pw'] = '1234'
    
import shap
from joblib import load
import pandas as pd
from training import retrieve_reduced_domain_features, form_new_ml_odds_data
from spring.api_wrappers import refreshBout
import numpy as np


class model_explainer:
    def __init__(self):
    
        self.label_translator = {"age_1": "BC Age",
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


        print('retrieve_best_trained_models - loading best model')
        mod = load('src/predictors/all/post_feat/trained_model.joblib')
        print('retrieve_best_trained_models - successfully loaded best model')
        print('retrieve_best_trained_models - loading best scale')
        self.scale = load('src/predictors/all/post_feat/trained_scale.joblib')
        print('retrieve_best_trained_models - successfully loaded best scale')
        
        self.red_feats = retrieve_reduced_domain_features(domain = 'all')
                
        self.explainer = shap.TreeExplainer(mod)

    #    bout_id = '7200343d2e957cf3'
    def gen_win_pred_explainer(self, bout_id):

        print('retrieve_best_trained_models - loading best model')
        mod = load('src/predictors/all/post_feat/trained_model.joblib')
        print('retrieve_best_trained_models - successfully loaded best model')
        print('retrieve_best_trained_models - loading best scale')
        scale = load('src/predictors/all/post_feat/trained_scale.joblib')
        print('retrieve_best_trained_models - successfully loaded best scale')
                        
        explainer = shap.TreeExplainer(mod)
        
        
        
        red_feats = retrieve_reduced_domain_features(domain = 'all')
        bout_detail = refreshBout(bout_id)
        # print('insert_new_ml_prob - forming new bout data')
        bout_data = form_new_ml_odds_data(bout_id)[red_feats['features']]
                
        new_row = pd.DataFrame(scale.transform(np.array(bout_data[red_feats['features']]).reshape(1,-1))[0]).T
        new_row.set_index(pd.Index([bout_id]), inplace= True)
        new_row.rename(columns = {i:j for i,j in zip(list(new_row), list(bout_data))}, inplace = True)     
        
        shap_values = explainer.shap_values(new_row)
        
        columns, fighter_1_data, fighter_2_data = self._win_pred_graph_data(shap_values, new_row)
        
        fighter_1_data['name'] = bout_detail['fighterBoutXRefs'][0]['fighter']['fighterName']
        fighter_2_data['name'] = bout_detail['fighterBoutXRefs'][1]['fighter']['fighterName']
        
        return columns, fighter_1_data, fighter_2_data
        
        
    def _win_pred_graph_data(self, shap_values, new_row):
    
        fighter_1_data = {'name': 'Feature Impact', 'data': []}
        fighter_2_data = {'name': 'Cumulative Confidence', 'data': []}
        columns = []
        
        conf = 0
        sorted_indexes = [x for _,x in sorted(zip([abs(i) for i in shap_values[1][0]], range(len(shap_values[1][0]))))]
        
        top_10 = sorted_indexes[-10:]
        
        columns.append('24 other features')
        for idx in sorted_indexes[:-10]:
            i = shap_values[1][0][idx]
            if (i != 0) :
                new_conf = conf + ((1/(1 + np.exp(-1*(i)))) - .5)
            else:
                new_conf = conf
            conf = new_conf
        
        if (conf > 0):
            fighter_1_data['data'].append(conf)
            fighter_2_data['data'].append(0)
        else:
            fighter_1_data['data'].append(0)
            fighter_2_data['data'].append(conf)        
        # line_data['data'].append(conf)
        # area_data['data'].append(conf)
        
        for top_idx in top_10:
            i = shap_values[1][0][top_idx]
            j = self.label_translator[list(new_row)[top_idx]]
            
            if (i != 0) :
                effect = (1/(1 + np.exp(-1*(i)))) - .5
            else:
                effect = 0
        
            conf += effect
            
            if (effect >= 0):
                fighter_2_data['data'].append(effect)
                fighter_1_data['data'].append(0)
            else:
                fighter_2_data['data'].append(0)
                fighter_1_data['data'].append(effect)
                
            conf = new_conf
            columns.append(j)
        return columns, fighter_1_data, fighter_2_data
    
    
    