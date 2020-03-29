#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 17:38:53 2020

@author: eric.hensleyibm.com
"""

import matplotlib.pyplot as plt
import requests 
import pandas as pd
import math
import numpy as np
import uuid
import random
from parse_fights import getBoutsFromFight
HOST = '207.237.93.29'
PORT = '4646'
TRAINING_YEAR = '2019'
TEST_YEAR = '2020'

GET_FIGHTS_BY_YEAR = "http://%s:%s/ufc/rest/fight/year/%s"




def getTrainingFights(year):
    r = requests.get(url = GET_FIGHTS_BY_YEAR % (HOST, PORT, year))
    response = r.json()         
    return response['response']

def getFighterRank(fighter_elos, fighterId):
    if fighterId not in fighter_elos.keys():
        fighter_elos[fighterId] = 1000
    return fighter_elos[fighterId]


def calcK(mov, elo_dif, mov_adj, pov_pwr, denom_const, denom_mult):
    return 20 * (math.pow(mov * mov_adj, pov_pwr)/(denom_const + denom_mult*(elo_dif)))
     
def calcProb(fighter_rank, opponent_rank, rank_damper):
    return 1/((math.pow(10,(fighter_rank - opponent_rank)/400)) + 1)

def calMoV(winner_score, loser_score, winner_damper, total_damper):
    return (winner_score - winner_damper) / (winner_score + loser_score - total_damper)

def adjElo(fighter_elos, fighterId, fighter_outcome, fighter_win_prob, k):
    fighter_elos[fighterId] = k * (fighter_outcome - fighter_win_prob) + fighter_elos[fighterId] 

def logError(error_log, fighter_outcome, fighter_win_prob, k):
    error_log['errors'].append(abs(k * (fighter_outcome - fighter_win_prob))**2)
    
def calElos(fighter_elos, grouped, rank_damper, mov_adj, pov_pwr, denom_const, denom_mult, winner_damper, total_damper, error_log):
#    for n in range(n_iter):
    for i, df in grouped:
        calc_res_acc = True
        if (df.iloc[0]['fighterId'] not in fighter_elos.keys() or df.iloc[1]['fighterId'] not in fighter_elos.keys()):
            calc_res_acc = False
        fighter_a_rank = getFighterRank(fighter_elos, df.iloc[0]['fighterId'])
        fighter_b_rank = getFighterRank(fighter_elos, df.iloc[1]['fighterId'])
    
    
        fighter_a_win_prob = calcProb(fighter_a_rank, fighter_b_rank, rank_damper)
        fighter_b_win_prob = calcProb(fighter_b_rank, fighter_a_rank, rank_damper)
    
    #    fighter_a_result = (df.iloc[0]['score'] - 15) / (df.iloc[0]['score'] + df.iloc[1]['score'] - 30)
    
        if (df.iloc[0]['score'] > df.iloc[1]['score']):
            mov = calMoV(df.iloc[0]['score'], df.iloc[1]['score'] , winner_damper, total_damper)
            elo_dif = fighter_a_rank - fighter_b_rank
            fighter_a_outcome = 1
            fighter_b_outcome = 0
            if calc_res_acc:
                if (fighter_a_win_prob > fighter_b_win_prob):
                    error_log['result'].append(1)
                elif (fighter_a_win_prob < fighter_b_win_prob):
                    error_log['result'].append(0)
                else:
                    error_log['result'].append(.5)
        else:
            mov = calMoV(df.iloc[1]['score'], df.iloc[0]['score'] , winner_damper, total_damper)
            elo_dif = fighter_b_rank - fighter_a_rank
            fighter_a_outcome = 0
            fighter_b_outcome = 1
            if calc_res_acc:
                if (fighter_a_win_prob > fighter_b_win_prob):
                    error_log['result'].append(0)
                elif (fighter_a_win_prob < fighter_b_win_prob):
                    error_log['result'].append(1)
                else:
                    error_log['result'].append(.5)
            
        k = calcK(mov, elo_dif, mov_adj, pov_pwr, denom_const, denom_mult)
        logError(error_log, fighter_a_outcome, fighter_a_win_prob, k)
        adjElo(fighter_elos, df.iloc[0]['fighterId'], fighter_a_outcome, fighter_a_win_prob, k)
        adjElo(fighter_elos, df.iloc[1]['fighterId'], fighter_b_outcome, fighter_b_win_prob, k)
            
def pullDataSet(set_list):
    n = 0
    training_dict = {}
    fight_id_list = getTrainingFights(set_list)
    for fight_id in fight_id_list:
        fight_details = getBoutsFromFight(fight_id)
        fight_date = fight_details['fightDate']
        for bout in fight_details['bouts']:
            bout_oid = bout['oid']
            for xref in bout['fighterBoutXRefs']:
                fighter_id = xref['fighter']['fighterId']
                for rnd in xref['boutDetails']:
                    round_data = {}
                    round_data['fightOid'] = fight_details['oid']
                    round_data['fightDate'] = fight_date
                    round_data['boutOid'] = bout_oid
                    round_data['fighterId'] =  fighter_id
                    round_data['round'] = rnd['round']
                    round_data['score'] = rnd['score']
                    training_dict[n] = round_data
                    n += 1
                    
    training_set = pd.DataFrame.from_dict(training_dict).T
    training_set.sort_values(["fightDate", "boutOid", "round"], inplace = True)
    grouped = training_set.groupby(['boutOid', "round"])
    return grouped

def eloIter(training_set, test_set, rank_damper, mov_adj, pov_pwr, denom_const, denom_mult, winner_damper, total_damper):
    error_log = {'errors':[], 'result':[]}
    validation_log = {'errors':[], 'result':[]}
    elos = {}
    calElos(elos, training_set, rank_damper, mov_adj, pov_pwr, denom_const, denom_mult, winner_damper, total_damper, error_log)
    print("Training average error: %s" % (np.mean(error_log['errors'])))
    print("Training prediction accuracy: %s" % (np.mean(error_log['result'])))
    calElos(elos, test_set, rank_damper, mov_adj, pov_pwr, denom_const, denom_mult, winner_damper, total_damper, validation_log)
    print("Validation average error: %s" % (np.mean(validation_log['errors'])))
    print("Validation prediction accuracy: %s" % (np.mean(validation_log['result'])))
    
    result = {'rank_damper': rank_damper, 'mov_adj': mov_adj, 'pov_pwr': pov_pwr, 'denom_const': denom_const, 'denom_mult': denom_mult, 'winner_damper': winner_damper, 'total_damper': total_damper, 'train_error': np.mean(error_log['errors']), 'train_acc': np.mean(error_log['result']), 'val_error': np.mean(validation_log['errors']), 'val_acc': np.mean(validation_log['result']) }
    results[str(uuid.uuid4())] = result
      
training_set = pullDataSet(TRAINING_YEAR)   
test_set = pullDataSet(TEST_YEAR)
results = {} 

for i in range(1000):
    rank_damper = random.randrange(200, 601)#400
    mov_adj = random.randrange(6,15) #10
    pov_pwr = random.randrange(5,11)/10 # .8
    denom_const = random.randrange(0,41)/10 #2
    denom_mult = random.randrange(1,101)/100 # .006
    winner_damper = random.randrange(0, 22)#15
    total_damper = random.randrange(0,43)#30 
    try:
        eloIter(training_set, test_set, rank_damper, mov_adj, pov_pwr, denom_const, denom_mult, winner_damper, total_damper)
    except:
        pass

    #
#    fighter_elos[df.iloc[0]['fighterId']] = k * (fighter_a_outcome - fighter_a_win_prob) + fighter_elos[df.iloc[0]['fighterId']] 
#    fighter_elos[df.iloc[1]['fighterId']] = k * (fighter_b_outcome - fighter_b_win_prob) + fighter_elos[df.iloc[1]['fighterId']] 

#plt.scatter([i for i in range(len(error_log))], error_log, alpha=0.5)
#plt.show()
   
    
    
    
    
