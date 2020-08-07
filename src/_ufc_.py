#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 12:11:19 2020

@author: eric.hensleyibm.com
"""
import argparse
import os

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
from spring import getRankings, addBoutsToFutureFight
from predictors import insert_new_ml_prob
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from scipy.stats import percentileofscore
import numpy as np
import pandas as pd

standard_response = {'status': 'Ok', 'errorMsg': None, 'itemsFound': 1, 'itemsCompleted': 1, 'statusCode': 200}
fail_response = {'status': 'Internal Server Error', 'errorMsg': None, 'itemsFound': 1, 'itemsCompleted': 0, 'statusCode': 400}
standard_get_response = {'status': 'Ok', 'errorMsg': None, 'itemsFound': 1, 'itemsCompleted': 1, 'statusCode': 200, 'response': None}

#   wc='WW'
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
        fighters[rank['fighterOid']]['total'] = None
        
        mm_fighters[rank['fighterOid']] = {i:None for i in stat_cols}
        mm_fighters[rank['fighterOid']]['name'] = rank['fighterName']
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
    f_df = f_df.iloc[0:25]
    f_df = f_df[['name', 'total']]
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

    response = []
    for row in df.values:
        response.append({'name': row[0], 'total': row[1], 'defKo': row[2],  'defGrapp': row[3], 'defStrike': row[4], 'defSub': row[5], 'offGrapp': row[6], 'offStrike': row[7], 'offKo': row[8], 'offSub': row[9]})
    print('  -- cleared rank init step 6; final.')

    return response

class ufc_engine:
    
    def __init__(self, pw):
        self.pw = pw
        self.weight_class_rankings = {}
        self.weight_classes = ['WW',
                                'FW',
                                'WSW',
                                'WFW',
                                'BW',
                                'WBW',
                                'FFW',
                                'WFFW',
                                'LW',
                                'MW',
                                'LHW',
                                'HW']
        for wc in self.weight_classes:
            print('initializing %s' % (wc))
            self.weight_class_rankings[wc] = calc_rankings_for_wc(wc)
            print('initialized %s' % (wc))
        
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
        
    def addMlProb(self, boutId):
        try:
            insert_new_ml_prob(boutId)
            return standard_response
        except Exception as e:
            print('Request failed with %s' % (e))
            resp = deepcopy(fail_response)
            resp['errorMsg'] = e
            return resp         
            
        

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


    


