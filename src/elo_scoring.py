#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:48:14 2020

@author: ehens86
"""

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    
import numpy as np
from spring.api_wrappers import getAllBouts, refreshBout, getLastElo, updateElo, clearElo
import math
import json
import uuid 

def simplify_round_stats(round_stat):
    simp_round_stat = {'tkoKo': round_stat['tkoKo'], 'submissionSuccessful': round_stat['submissionSuccessful'], 'koScore': round_stat['koScore'], 'submissionScore': round_stat['submissionScore']}
    if simp_round_stat['koScore'] is None or simp_round_stat['submissionScore'] is None:
        if DEBUG:
            print("Null values in round stats")
        raise ValueError("Null values in round stats")
    return simp_round_stat
    
#bout_info, index = bout_info, 0
def prep_fighter(bout_info, index):
    fighter = bout_info['fighterBoutXRefs'][index]
    if fighter['offStrikeEloPost'] is not None or fighter['defStrikeEloPost'] is not None or fighter['offGrapplingEloPost'] is not None or fighter['defGrapplingEloPost'] is not None:
        if DEBUG:
            print("Elo scores for bout %s and fighter %s already saved" % (fighter['fighter']['oid'], bout_info['fightOid']))
        raise ValueError("Elo scores for bout %s and fighter %s already saved" % (fighter['fighter']['oid'], bout_info['fightOid']))
    fighter_elo = getLastElo(fighter['fighter']['oid'], bout_info['fightOid'])
    if fighter_elo['oid'] is None:
        if DEBUG:
            print("Initializing new elo baseline for fighter %s" % fighter['fighter']['oid'])      
        fighter_elo['oid'] =  fighter['oid']
    else:
        fighter_elo['oid'] =  fighter['oid']
        fighter_elo['offStrikeEloPre'] = fighter_elo['offStrikeEloPost']
        fighter_elo['defStrikeEloPre'] = fighter_elo['defStrikeEloPost'] 
        fighter_elo['offGrapplingEloPre'] = fighter_elo['offGrapplingEloPost'] 
        fighter_elo['defGrapplingEloPre'] = fighter_elo['defGrapplingEloPost'] 
        fighter_elo['powerStrikeEloPre'] = fighter_elo['powerStrikeEloPost'] 
        fighter_elo['chinStrikeEloPre'] = fighter_elo['chinStrikeEloPost'] 
        fighter_elo['subGrapplingEloPre'] = fighter_elo['subGrapplingEloPost'] 
        fighter_elo['evasGrapplingEloPre'] = fighter_elo['evasGrapplingEloPost'] 

        fighter_elo['offStrikeEloPost'] = None
        fighter_elo['defStrikeEloPost'] = None
        fighter_elo['offGrapplingEloPost'] = None
        fighter_elo['defGrapplingEloPost'] = None
        fighter_elo['powerStrikeEloPost'] = None
        fighter_elo['chinStrikeEloPost'] = None
        fighter_elo['subGrapplingEloPost'] = None
        fighter_elo['evasGrapplingEloPost']= None 
    return fighter, fighter_elo    
    
def prep_round_stats(bout_info, fighter1, fighter2):
    round_dict = {}
    for rnd in range(bout_info['finishRounds']):
        round_dict[rnd+1] = {fighter1['fighter']['oid']: {}, fighter2['fighter']['oid']: {}}
        
    for fighter1Round in fighter1['boutDetails']:
        round_dict[fighter1Round['round']][fighter1['fighter']['oid']] = simplify_round_stats(fighter1Round)
    for fighter2Round in fighter2['boutDetails']:
        round_dict[fighter2Round['round']][fighter2['fighter']['oid']] = simplify_round_stats(fighter2Round)
    return round_dict

def calc_prob_damper(x, offense, defense):
  return 1 / (1 + math.exp(10*((offense+(1-defense))/2-x)))

def calc_sub_ko_odds(x, offense, defense):
  return x**(1+offense-defense)

def adj_score(off_fighter, def_fighter, act_val, pre_post, est_feat, damper, iter_scores):
#    off_fighter, def_fighter, act_val, pre_post, est_feat, damper, iter_scores = fighter_2_elo, fighter_1_elo, vals[fighter_2['fighter']['oid']]['koScore'], pre_post, 'Strike', STRIKE_DAMPER, iter_scores
    valid = True
    prob_damper = calc_prob_damper(act_val, off_fighter['off%sElo%s' % (est_feat, pre_post)], def_fighter['def%sElo%s' % (est_feat, pre_post)])
    iter_scores['off%sElo' % (est_feat)].append(act_val - off_fighter['off%sElo%s' % (est_feat, pre_post)])
    new_off_score = off_fighter['off%sElo%s' % (est_feat, pre_post)] ** (1/ (1 + (abs(act_val - off_fighter['off%sElo%s' % (est_feat, pre_post)]) * (prob_damper - .5) * damper)))
    if (new_off_score < 0):
        print('off%sEloPost would be negative (%.2f)' % (est_feat, new_off_score))
        valid = False
    if (new_off_score > 1):
        print('off%sEloPost would be greater than 1 (%.2f)' % (est_feat, new_off_score))
        valid = False        

    iter_scores['def%sElo' % (est_feat)].append(def_fighter['def%sElo%s' % (est_feat, pre_post)] - (1-act_val))
    new_def_score = def_fighter['def%sElo%s' % (est_feat, pre_post)] ** (1/ (1- (abs(def_fighter['def%sElo%s' % (est_feat, pre_post)] - (1-act_val)) * (prob_damper - .5) * damper)))
    if (new_def_score < 0):
        print('def%sEloPost would be negative (%.2f)' % (est_feat, new_def_score))
        valid = False
    if (new_def_score > 1):
        print('def%sEloPost would be greater than 1 (%.2f)' % (est_feat, new_def_score))
        valid = False        
            
    if valid == False:
        raise ValueError("Elo scores must be between 0-1")
    off_fighter['off%sEloPost' % (est_feat)] = new_off_score
    def_fighter['def%sEloPost' % (est_feat)] = new_def_score
    
def adj_finish(off_fighter, def_fighter, act_score, act_val, pre_post, off_est_feat, def_est_feat, damper, iter_scores):
    valid = True
    finish_odds = calc_sub_ko_odds(act_score, off_fighter['%sElo%s' % (off_est_feat, pre_post)], def_fighter['%sElo%s' % (def_est_feat, pre_post)])

    iter_scores['%sElo' % (off_est_feat)].append(off_fighter['%sElo%s' % (off_est_feat, pre_post)] - finish_odds)
    iter_scores['%sElo' % (def_est_feat)].append(def_fighter['%sElo%s' % (def_est_feat, pre_post)] - finish_odds)

    if act_val == 1:
        new_off_score = off_fighter['%sElo%s' % (off_est_feat, pre_post)] ** (1/(1+ ((1-finish_odds) * (abs(off_fighter['%sElo%s' % (off_est_feat, pre_post)] - finish_odds) * damper))))
        new_def_score = def_fighter['%sElo%s' % (def_est_feat, pre_post)] ** (1/(1 - (finish_odds) * (abs(def_fighter['%sElo%s' % (def_est_feat, pre_post)] - finish_odds) * damper )))
    else:
        new_off_score = off_fighter['%sElo%s' % (off_est_feat,pre_post)] ** (1/(1 - ((1-finish_odds) * (abs(off_fighter['%sElo%s' % (off_est_feat, pre_post)] - finish_odds) * damper))))
        new_def_score = def_fighter['%sElo%s' % (def_est_feat, pre_post)] ** (1/(1 + (finish_odds) * (abs(def_fighter['%sElo%s' % (def_est_feat, pre_post)] - finish_odds) * damper )))

    if (new_off_score < 0):
        print('%sEloPost would be negative (%.2f)' % (off_est_feat, new_off_score))
        valid = False
    if (new_off_score > 1):
        print('%sEloPost would be greater than 1 (%.2f)' % (off_est_feat, new_off_score))
        valid = False        

    if (new_def_score < 0):
        print('%sEloPost would be negative (%.2f)' % (def_est_feat, new_def_score))
        valid = False
    if (new_def_score > 1):
        print('%sEloPost would be greater than 1 (%.2f)' % (def_est_feat, new_def_score))
        valid = False        
            
    if valid == False:
        raise ValueError("Elo scores must be between 0-1")
       
    off_fighter['%sEloPost' % (off_est_feat)] = new_off_score
    def_fighter['%sEloPost' % (def_est_feat)] = new_def_score

def elo_iter(strike_damper, sub_damper, ko_finish_damper, sub_finish_damper):
#    strike_damper, sub_damper, ko_finish_damper, sub_finish_damper = srike_d, sub_d, ko_d, sub_fin_d
    print("Beginning ELO parameter search with values:")
    print("  strikeDamper: %s" % strike_damper)
    print("  subDamper: %s" % sub_damper)
    print("  koFinishDamper: %s" % ko_finish_damper)
    print("  subFinishDamper: %s" % sub_finish_damper)

    clearElo()
    
    iter_scores = {'offStrikeElo':[], 'defStrikeElo':[], 'offGrapplingElo':[], 'defGrapplingElo':[], 'powerStrikeElo':[], 'chinStrikeElo':[], 'subGrapplingElo':[], 'evasGrapplingElo':[]}
    for bout_oid in BOUTS:
        bout_info = refreshBout(bout_oid)
        if bout_info['gender'] != 'MALE':
            if DEBUG:
                print("Skipping bout %s.. not male" % (bout_info['oid']))
            continue
        try:
            fighter_1, fighter_1_elo = prep_fighter(bout_info, 0)
            fighter_2, fighter_2_elo = prep_fighter(bout_info, 1)
        except ValueError:
            continue
        
        try:
            round_stats = prep_round_stats(bout_info, fighter_1, fighter_2)    
        except:
            if DEBUG:
                print("Error pulling round data for bout %s" % (bout_info['oid']))
            continue    
        for rnd, vals in round_stats.items():
            if rnd == 1:
                pre_post = 'Pre'
            else:
                pre_post = 'Post'
                
            adj_score(fighter_1_elo, fighter_2_elo, vals[fighter_1['fighter']['oid']]['koScore'], pre_post, 'Strike', strike_damper, iter_scores)
            adj_score(fighter_2_elo, fighter_1_elo, vals[fighter_2['fighter']['oid']]['koScore'], pre_post, 'Strike', strike_damper, iter_scores)
            
            adj_score(fighter_1_elo, fighter_2_elo, vals[fighter_1['fighter']['oid']]['submissionScore'], pre_post, 'Grappling', sub_damper, iter_scores)
            adj_score(fighter_2_elo, fighter_1_elo, vals[fighter_2['fighter']['oid']]['submissionScore'], pre_post, 'Grappling', sub_damper, iter_scores)
                
            adj_finish(fighter_1_elo, fighter_2_elo, vals[fighter_1['fighter']['oid']]['koScore'], vals[fighter_1['fighter']['oid']]['tkoKo'], pre_post, 'powerStrike', 'chinStrike', ko_finish_damper, iter_scores)
            adj_finish(fighter_2_elo, fighter_1_elo, vals[fighter_2['fighter']['oid']]['koScore'], vals[fighter_2['fighter']['oid']]['tkoKo'], pre_post, 'powerStrike', 'chinStrike', ko_finish_damper, iter_scores)
            
            adj_finish(fighter_1_elo, fighter_2_elo, vals[fighter_1['fighter']['oid']]['submissionScore'], vals[fighter_1['fighter']['oid']]['submissionSuccessful'], pre_post, 'subGrappling', 'evasGrappling', sub_finish_damper, iter_scores)
            adj_finish(fighter_2_elo, fighter_1_elo, vals[fighter_2['fighter']['oid']]['submissionScore'], vals[fighter_2['fighter']['oid']]['submissionSuccessful'], pre_post, 'subGrappling', 'evasGrappling', sub_finish_damper, iter_scores)
    
        fighter_1_update = updateElo(fighter_1_elo)
        if fighter_1_update['errorMsg'] is not None:
            print(fighter_1_update['errorMsg'])
        fighter_2_update = updateElo(fighter_2_elo)
        if fighter_2_update['errorMsg'] is not None:
            print(fighter_2_update['errorMsg'])
    iter_scores['strikeDamper'] = strike_damper
    iter_scores['subDamper'] = sub_damper
    iter_scores['koFinishDamper'] = ko_finish_damper
    iter_scores['subFinishDamper'] = sub_finish_damper
    
    with open('data/elo_scores/%s.json' % (uuid.uuid1()), 'w') as json_file:
        json.dump(iter_scores, json_file)
        
if __name__ == '__main__':
    global DEBUG
    DEBUG = False
    BOUTS = getAllBouts()
    for srike_d in np.linspace(.25, 2.25, 5):
        for sub_d in np.linspace(.25, 2.25, 5):
            for ko_d in np.linspace(.25, 2.25, 5):
                for sub_fin_d in np.linspace(.25, 2.25, 5):
                    elo_iter(srike_d, sub_d, ko_d, sub_fin_d)
#    STRIKE_DAMPER = .75
#    SUB_DAMPER = .75
#    KO_FINISH_DAMPER = .75
#    SUB_FINISH_DAMPER = .75