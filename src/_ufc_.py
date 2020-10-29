#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 12:11:19 2020

@author: eric.hensleyibm.com
"""
import argparse
import sys, os

if __name__ == "__main__":
    sys.path.append("src")
    os.environ['ufc.flask.spring.host'] = 'http://192.168.1.64:4646'#'http://68.248.220.199:4646'
    os.environ['ufc.flask.spring.pw'] = '1234'
    print(os.environ)
    
parser = argparse.ArgumentParser(description='UFC Prediction Engine')
parser.add_argument('--fport', type=int,
                    help='Flask Port')
parser.add_argument('--spw', type=str,
                    help='Spring Password')
parser.add_argument('--shost', type=str,
                    help='Spring Host')
args = parser.parse_args()

if (str(args.shost) != 'None'):
    os.environ['ufc.flask.spring.host'] = str(args.shost)
if (str(args.fport) != 'None'):
    os.environ['ufc.flask.flask.port'] = str(args.fport)
if (str(args.spw) != 'None'):
    os.environ['ufc.flask.spring.pw'] = str(args.spw)

#    os.environ['ufc.flask.spring.host'] = 'http://localhost:4646'
#    print(os.environ)

from copy import deepcopy
from db import addInfoToAllBouts
from spring import getRankings, addBoutsToFutureFight, initUpdate, futureFightUpdate, refreshBout
from predictors import insert_new_ml_prob
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import shap
from joblib import load
from training import retrieve_reduced_domain_features, form_new_ml_odds_data

import numpy as np
import pandas as pd

standard_response = {'status': 'Ok', 'errorMsg': None, 'itemsFound': 1, 'itemsCompleted': 1, 'statusCode': 200}
fail_response = {'status': 'Internal Server Error', 'errorMsg': None, 'itemsFound': 1, 'itemsCompleted': 0, 'statusCode': 400}
standard_get_response = {'status': 'Ok', 'errorMsg': None, 'itemsFound': 1, 'itemsCompleted': 1, 'statusCode': 200, 'response': None}

#   wc='HW'
def calc_rankings_for_wc(wc):
    rankings = getRankings(wc)['response']
    
    print('pulled %s rankings' % (wc))
    mm_fighters = {}
    fighters = {}
    wc_stat_univ = {}
    
    tot_rank_univ = []
    
    scales = {}
    mm_scales = {}
    
    stat_cols = ['offStrikeEloPost',
                'defStrikeEloPost',
                'offGrapplingEloPost',
                'defGrapplingEloPost',
                'powerStrikeEloPost',
                'chinStrikeEloPost',
                'subGrapplingEloPost',
                'evasGrapplingEloPost'
                ]
    
    for stat_name in stat_cols:
        wc_stat_univ[stat_name] = []
        scales[stat_name] = StandardScaler()
        mm_scales[stat_name] = MinMaxScaler()
        
    for rank in rankings:
        fighters[rank['fighterOid']] = {i:None for i in stat_cols}
        fighters[rank['fighterOid']]['name'] = rank['fighterName']
        fighters[rank['fighterOid']]['fighterOid'] = rank['fighterOid']
        fighters[rank['fighterOid']]['total'] = None
        
        mm_fighters[rank['fighterOid']] = {i:None for i in stat_cols}
        mm_fighters[rank['fighterOid']]['name'] = rank['fighterName']
        mm_fighters[rank['fighterOid']]['fighterOid'] = rank['fighterOid']
        mm_fighters[rank['fighterOid']]['total'] = None
        for stat in stat_cols:
                wc_stat_univ[stat].append(rank[stat])
    
    for s_name in stat_cols:
        scales[s_name].fit(np.array(wc_stat_univ[s_name]).reshape(-1, 1))
        mm_scales[s_name].fit(np.array(wc_stat_univ[s_name]).reshape(-1, 1))

    print('  -- cleared rank init step 1')
    for f_rank in rankings:
        for f_stat in stat_cols:       
            fighters[f_rank['fighterOid']][f_stat] = scales[f_stat].transform(np.array(f_rank[f_stat]).reshape(1, -1))[0][0]
            mm_fighters[f_rank['fighterOid']][f_stat] = mm_scales[f_stat].transform(np.array(f_rank[f_stat]).reshape(1, -1))[0][0] * 100

#            fighters[f_rank['fighterOid']][f_stat] = percentileofscore(wc_stat_univ[f_stat], f_rank[f_stat], 'rank')
    
    print('  -- cleared rank init step 2')

    for f_vals in fighters.values():
        tot_rank_univ.append(np.sum([f_vals[i] for i in ['offStrikeEloPost', 'defStrikeEloPost', 'offGrapplingEloPost', 'defGrapplingEloPost']])) # ['offStrikeEloPost', 'defStrikeEloPost', 'offGrapplingEloPost', 'defGrapplingEloPost']
    
    print('  -- cleared rank init step 3')

    tot_scale = MinMaxScaler()
    tot_scale.fit(np.array(tot_rank_univ).reshape(-1, 1))
    for f_id, f_val in fighters.items():
        f_val['total'] = tot_scale.transform(np.array(np.sum([f_val[i] for i in ['offStrikeEloPost', 'defStrikeEloPost', 'offGrapplingEloPost', 'defGrapplingEloPost']]).reshape(1, -1)))[0][0] * 100 # 
#        f_val['total'] = percentileofscore(tot_rank_univ, np.sum([f_val[i] for i in ['offStrikeEloPost', 'defStrikeEloPost', 'offGrapplingEloPost', 'defGrapplingEloPost']]), 'rank')
    
    
    print('  -- cleared rank init step 4')
    
    mm_df = pd.DataFrame.from_dict(mm_fighters).T
    f_df = pd.DataFrame.from_dict(fighters).T
    f_df.sort_values('total', ascending = False, inplace = True)
#    f_df.reset_index(inplace = True)
#    f_df = f_df.iloc[0:25]
    f_df = f_df[['name', 'fighterOid', 'total']]
    mm_df = mm_df[['chinStrikeEloPost',
         'defGrapplingEloPost',
         'defStrikeEloPost',
         'evasGrapplingEloPost',
         'offGrapplingEloPost',
         'offStrikeEloPost',
         'powerStrikeEloPost',
         'subGrapplingEloPost']]
    df = f_df.join(mm_df)
    print('  -- cleared rank init step 5')

    list(df)
    response = []
    fighter_response = {}
    for i, (row) in enumerate(df.values):
        if (i < 15):
            response.append({'name': row[0], 'fighterOid': row[1], 'total': row[2], 'defKo': row[3],  'defGrapp': row[4], 'defStrike': row[5], 'defSub': row[6], 'offGrapp': row[7], 'offStrike': row[8], 'offKo': row[9], 'offSub': row[10]})
        fighter_response[row[1]] = {'name': row[0], 'fighterOid': row[1], 'total': row[2], 'defKo': row[3],  'defGrapp': row[4], 'defStrike': row[5], 'defSub': row[6], 'offGrapp': row[7], 'offStrike': row[8], 'offKo': row[9], 'offSub': row[10]}
    print('  -- cleared rank init step 6; final.')

    return response, fighter_response

class ufc_engine:
    
    def __init__(self, pw):
        self.pw = pw
        self.weight_class_rankings = {}
        self.weight_class_fighters = {}
        self.weight_classes = [ 'FW',
                                'WSW',
                                'WFW',
                                'BW',
                                'WBW',
                                'FFW',
                                'WFFW',
                                'LW',
                                'MW',
                                'WW',
                                'LHW',
                                'HW']
        for wc in self.weight_classes:
            print('initializing %s' % (wc))
            self.weight_class_rankings[wc], self.weight_class_fighters[wc] = calc_rankings_for_wc(wc)
            print('initialized %s' % (wc))
        self.label_translator = {
                                "age_1": "BC Age",
                                'age_2': 'RC Age',
                                'age_diff': 'Age DIFF',
                                'age_share': 'Age SHARE', 
                                'def_grapp_1': 'BC Grappling Def', 
                                'def_grapp_2': 'RC Grappling Def', 
                                'def_grapp_diff': 'Grappling Def DIFF', 
                                'def_grapp_share': 'Grappling Def SHARE', 
                                'def_ko_2': 'BC Chin',
                                'def_ko_diff': 'Chin DIFF',
                                'def_ko_share': 'Chin SHARE', 
                                'def_strike_1': 'BC Striking Def',
                                'def_strike_2': 'RC Striking Def',
                                'def_strike_diff': 'Striking Def DIFF',
                                'def_strike_share': 'Striking Def SHARE',
                                'def_sub_1': 'BC Submission Def', 
                                'def_sub_2': 'RC Submission Def', 
                                'def_sub_diff': 'Submission Def DIFF', 
                                'def_sub_share': 'Submission Def SHARE', 
                                'eff_grap_1': 'BC Grappling EFF',
                                'eff_grap_2': 'RC Grappling EFF',
                                'eff_grap_diff': 'Grappling EFF DIFF',
                                'eff_grap_share': 'Grappling EFF SHARE',
                                'eff_ko_1': 'BC KO EFF', 
                                'eff_ko_2': 'RC KO EFF', 
                                'eff_ko_diff': 'KO EFF DIFF', 
                                'eff_ko_share': 'KO EFF SHARE', 
                                'eff_strike_1': 'BC Striking EFF',
                                'eff_strike_2': 'RC Striking EFF',
                                'eff_strike_diff': 'Striking EFF DIFF',
                                'eff_strike_share': 'Striking EFF SHARE',
                                'eff_sub_1': 'BC Submission EFF', 
                                'eff_sub_2': 'RC Submission EFF', 
                                'eff_sub_diff': 'Submission EFF DIFF', 
                                'eff_sub_share': 'Submission EFF SHARE', 
                                'off_grapp_1': 'BC Grappling Off',
                                'off_grapp_2': 'RC Grappling Off',
                                'off_grapp_diff': 'Grappling Off DIFF',
                                'off_grapp_share': 'Grappling Off SHARE',
                                'off_ko_1': 'BC KO',
                                'off_ko_2': 'RC KO',
                                'off_ko_diff': 'KO DIFF',
                                'off_ko_share': 'KO SHARE',
                                'off_strike_1': 'BC Strike Off',
                                'off_strike_2': 'RC Strike Off',
                                'off_strike_diff': 'Strike Off DIFF',
                                'off_strike_share': 'Strike Off SHARE',
                                'off_sub_1': 'BC Submission Off', 
                                'off_sub_2': 'RC Submission Off', 
                                'off_sub_diff': 'Submission Off DIFF', 
                                'off_sub_share': 'Submission Off SHARE', 
                                'prev_fights_1': 'BC Num Fights',
                                'prev_fights_2': 'RC Num Fights',
                                'rounds': 'Rounds'
#             'def_ko_1': 'RC Chin',
#             'def_ko_2': 'BC Chin',
#             'def_ko_diff': 'Chin DIFF',
#             'def_ko_share': 'Chin SHARE',
#             'def_strike_1': 'BC Striking Def',
#             'def_strike_2': 'RC Striking Def',
#             'def_strike_diff': 'Striking Def DIFF',
#             'def_strike_share': 'Striking Def SHARE',
#             'eff_grap_1': 'BC Grappling EFF',
#             'eff_grap_2': 'RC Grappling EFF',
#             'eff_grap_diff': 'Grappling EFF DIFF',
#             'eff_grap_share': 'Grappling EFF SHARE',
#             'eff_ko_2': 'RC KO EFF',
#             'eff_strike_1': 'BC Striking EFF',
#             'eff_strike_2': 'RC Striking EFF',
#             'eff_strike_diff': 'Striking EFF DIFF',
#             'eff_strike_share': 'Striking EFF SHARE',
#             'off_grapp_1': 'BC Grappling Off',
#             'off_grapp_2': 'RC Grappling Off',
#             'off_grapp_diff': 'Grappling Off DIFF',
#             'off_grapp_share': 'Grappling Off SHARE',
#             'off_ko_1': 'BC KO',
#             'off_ko_diff': 'KO DIFF',
#             'off_ko_share': 'KO SHARE',
#             'off_strike_1': 'BC Strike Off',
#             'off_strike_2': 'RC Strike Off',
#             'off_strike_diff': 'Strike Off DIFF',
#             'off_strike_share': 'Strike Off SHARE',       
        }


        print('retrieve_best_trained_models - loading best model')
        mod = load('src/predictors/all/post_feat/trained_model.joblib')
        print('retrieve_best_trained_models - successfully loaded best model')
        print('retrieve_best_trained_models - loading best scale')
        self.scale = load('src/predictors/all/post_feat/trained_scale.joblib')
        print('retrieve_best_trained_models - successfully loaded best scale')
        
        self.red_feats = retrieve_reduced_domain_features(domain = 'all')
                
        self.explainer = shap.TreeExplainer(mod)
        
        
    def authenticate(self, headers):
        if 'Password' in headers:
            if str(headers.get('password')) == str(self.pw):
                return True
            else:
                print('Password does not match')
                return False
        else:
            print('Password missing from headers')
            return False
    
    def populate_past_fight(self, fight_id):
        try:
            addInfoToAllBouts(fight_id)
            return standard_response
        except Exception as e:
            print('Request failed with %s' % (e))
            resp = deepcopy(fail_response)
            resp['errorMsg'] = e
            return resp

    def populate_future_fight(self, fight_id):
        try:
            addBoutsToFutureFight(fight_id)
            return standard_response
        except Exception as e:
            print('Request failed with %s' % (e))
            resp = deepcopy(fail_response)
            resp['errorMsg'] = e
            return resp 
    
    def get_ranking_for_wc(self, wc):
        try:
            if (wc.upper() in self.weight_classes):
                resp = deepcopy(standard_get_response)
                resp['response'] = self.weight_class_rankings[wc.upper()]
                return resp
            else:
                resp = deepcopy(fail_response)
                resp['errorMsg'] = '%s is not a supported weight class' % (wc)
                return resp 
        except Exception as e:
            print('Request failed with %s' % (e))
            resp = deepcopy(fail_response)
            resp['errorMsg'] = e
            return resp 
        
    def get_top_wc_ranks(self):
        try:
            resp = deepcopy(standard_get_response)
            resp['response'] = []
            for wc in self.weight_classes:
                resp['response'].append({'wc': wc.upper(), 'top': self.weight_class_rankings[wc.upper()][0]})
            return resp
        except Exception as e:
            print('Request failed with %s' % (e))
            resp = deepcopy(fail_response)
            resp['errorMsg'] = e
            return resp         
        
        
        
        
    def addMlProb(self, boutId):
        try:
            insert_new_ml_prob(boutId)
            return standard_response
        except Exception as e:
            print('Request failed with %s' % (e))
            resp = deepcopy(fail_response)
            resp['errorMsg'] = e
            return resp         
            
    def popFutureBouts(self):
        try:
            futureFightUpdate()
            initUpdate()
            return standard_response
        except Exception as e:
            print('Request failed with %s' % (e))
            resp = deepcopy(fail_response)
            resp['errorMsg'] = e
            return resp       

    def get_ranking_for_wc_fighter(self, wc, fighterOid):
        try:
            if wc.upper() not in self.weight_class_fighters.keys():
                resp = deepcopy(fail_response)
                resp['errorMsg'] = '%s is not a supported weight class' % (wc)
                return resp 
            elif fighterOid not in self.weight_class_fighters[wc].keys():
                print(self.weight_class_fighters[wc].keys())
                print(wc)
                print(fighterOid)
                resp = deepcopy(fail_response)
                resp['errorMsg'] = '%s is not a supported fighter' % (fighterOid)
                return resp                 
            else:
                resp = deepcopy(standard_get_response)
                resp['response'] = self.weight_class_fighters[wc.upper()][fighterOid]
                return resp
        except Exception as e:
            print('Request failed with %s' % (e))
            resp = deepcopy(fail_response)
            resp['errorMsg'] = e
            return resp

    # bout_id = '3bdacc82209b33f5'
    def gen_win_pred_explainer(self, bout_id):
        try:
            bout_detail = refreshBout(bout_id)
            # print('insert_new_ml_prob - forming new bout data')
            bout_data = form_new_ml_odds_data(bout_id)[self.red_feats['features']]
                    
            new_row = pd.DataFrame(self.scale.transform(np.array(bout_data[self.red_feats['features']]).reshape(1,-1))[0]).T
            new_row.set_index(pd.Index([bout_id]), inplace= True)
            new_row.rename(columns = {i:j for i,j in zip(list(new_row), list(bout_data))}, inplace = True)     
            
            shap_values = self.explainer.shap_values(new_row)
            
            print(list(new_row))
            columns, fighter_1_data, fighter_2_data = self._win_pred_graph_data(shap_values, new_row)
            
            fighter_1_data['name'] = bout_detail['fighterBoutXRefs'][0]['fighter']['fighterName']
            fighter_2_data['name'] = bout_detail['fighterBoutXRefs'][1]['fighter']['fighterName']
            
            resp = deepcopy(standard_get_response)
            resp['response'] = {'statCols': columns, 'boutArray': [fighter_2_data, fighter_1_data]}
            
            return resp
        except Exception as e:
            print('Request failed with %s' % (e))
            resp = deepcopy(fail_response)
            resp['errorMsg'] = e
            return resp        
        
        
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
                fighter_1_data['data'].append(effect)
                fighter_2_data['data'].append(0)
            else:
                fighter_1_data['data'].append(0)
                fighter_2_data['data'].append(effect)
                
            conf = new_conf
            columns.append(j)
        
        columns.append('Total')
        if (conf >= 0):
            fighter_1_data['data'].append(conf)
            fighter_2_data['data'].append(0)  
        else:
            fighter_1_data['data'].append(0)
            fighter_2_data['data'].append(conf)              
        return columns, fighter_1_data, fighter_2_data
      
     
#from spring.config import CONFIG
#engine = ufc_engine(CONFIG['spring']['PW'])
#engine.gen_win_pred_explainer('59eea37af7efb06c')
#model = ufc_engine()
#model._update_fight_list()
#
#model.fight_list
#
#fight_details = model._get_next_fight()
#date_time_str = fight_details['fightDate']
#
#date_time_obj = datetime.strptime(date_time_str.split('T')[0], '%Y-%m-%d')
#
#date_time_obj.strftime("%A %B %d")


    


