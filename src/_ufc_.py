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
                         scrapeBoutScores, addFightOdds, addFightExpectedOutcomes, addBoutsToFutureFight
from datetime import datetime

standard_response = {'status': 'Ok', 'errorMsg': None, 'itemsFound': 1, 'itemsCompleted': 1, 'statusCode': 200}
fail_response = {'status': 'Internal Server Error', 'errorMsg': None, 'itemsFound': 1, 'itemsCompleted': 0, 'statusCode': 400}



class ufc_engine:
    
    def __init__(self, pw):
        self.pw = pw
        
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


    


