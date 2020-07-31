#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 12:11:19 2020

@author: eric.hensleyibm.com
"""

from copy import deepcopy
from training import tune_ml, fill_ml_training_scores, optimize_bet, predict_bet_winners
from elo import optimize_elo, populate_elo
from db import pop_future_bouts, pop_year_bouts, update_mybookie, addInfoToAllBouts
from spring import addRoundScore, getAllFightIds, getBoutsFromFight, addBoutScoreUrls, \
                         addBoutsToFight, addBoutDetails, addFightOddsUrl, refreshBout, \
                         scrapeBoutScores, addFightOdds, addFightExpectedOutcomes, \
                         addBoutsToFutureFight, getRankings, getLastEloCount, getEloCount
from predictors import insert_new_ml_scores, insert_new_ml_prob

import datetime
from scipy.stats import percentileofscore
import numpy as np
import pandas as pd

standard_response = {'status': 'Ok', 'errorMsg': None, 'itemsFound': 1, 'itemsCompleted': 1, 'statusCode': 200}
fail_response = {'status': 'Internal Server Error', 'errorMsg': None, 'itemsFound': 1, 'itemsCompleted': 0, 'statusCode': 400}
standard_get_response = {'status': 'Ok', 'errorMsg': None, 'itemsFound': 1, 'itemsCompleted': 1, 'statusCode': 200, 'response': None}

NOW = datetime.datetime.now()
D = datetime.timedelta(days = (365 * 2))
#wc='LW'
def calc_rankings_for_wc(wc):
    rankings = getRankings(wc)['response']
    
    fighters = {}
    wc_stat_univ = {}
    
    tot_rank_univ = []
    
    stat_cols = ['offStrikeEloPost',
                'defStrikeEloPost',
                'offGrapplingEloPost',
                'defGrapplingEloPost'
                ]
    
    for stat_name in stat_cols:
        wc_stat_univ[stat_name] = []
        
    for rank in rankings:
        rank_date = datetime.datetime.strptime(rank['fightDate'].split('.')[0], '%Y-%m-%dT%H:%M:%S')
        
        if (rank['fighterBoutXRef']['expOdds'] is not None and NOW - D < rank_date and getEloCount(rank['fighter']['oid']) > 5):
            fighters[rank['fighterBoutXRef']['fighter']['oid']] = {i:None for i in stat_cols}
            fighters[rank['fighterBoutXRef']['fighter']['oid']]['name'] = rank['fighterBoutXRef']['fighter']['fighterName']
            fighters[rank['fighterBoutXRef']['fighter']['oid']]['total'] = None
        for stat in stat_cols:
                wc_stat_univ[stat].append(rank['fighterBoutXRef'][stat])
                
    for f_rank in rankings:
        f_rank_date = datetime.datetime.strptime(f_rank['fightDate'].split('.')[0], '%Y-%m-%dT%H:%M:%S')

        if (f_rank['fighterBoutXRef']['expOdds'] is not None and NOW - D < f_rank_date and getEloCount(f_rank['fighter']['oid']) > 5):
            for f_stat in stat_cols:        
                fighters[f_rank['fighterBoutXRef']['fighter']['oid']][f_stat] = percentileofscore(wc_stat_univ[f_stat], f_rank['fighterBoutXRef'][f_stat], 'rank')
        
    for f_vals in fighters.values():
        tot_rank_univ.append(np.sum([f_vals[i] for i in stat_cols]))
    
    for f_id, f_val in fighters.items():
        f_val['total'] = percentileofscore(tot_rank_univ, np.sum([f_val[i] for i in stat_cols]), 'rank')
    
    f_df = pd.DataFrame.from_dict(fighters).T
    f_df.sort_values('total', ascending = False, inplace = True)
    f_df.reset_index(inplace = True)
    f_df = f_df.loc[0:19]
    
    response = []
    for row in f_df.values:
        response.append({'oid': row[0], 'name': row[3], 'total': row[6], 'defGrapp': row[1], 'defStrike': row[2], 'offGrapp': row[4], 'offStrike': row[5]})
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
            self.weight_class_rankings[wc] = calc_rankings_for_wc(wc)
        
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


    


