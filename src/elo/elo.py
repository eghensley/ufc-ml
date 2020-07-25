#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:09:36 2020

@author: ehens86
"""

if __name__ == "__main__":
    import sys
    sys.path.append("..")

import os
from spring.api_wrappers import getAllBouts, refreshBout, getLastElo, updateElo, clearElo, getYearBouts
import math
import json
import pandas as pd
import uuid
import optuna
from elo.elo_eval import gen_config
from scipy.stats import truncnorm
import random
import numpy as np

def calc_prob_damper(x, offense, defense):
  return 1 / (1 + math.exp(10*((offense+(1-defense))/2-x)))

def calc_sub_ko_odds(x, offense, defense):
  return x**(1+offense-defense)

from spring.api_wrappers import getLastEloCount    
from datetime import datetime

from utils.general import calcWinnings, progress
from joblib import Parallel, delayed

from sklearn.metrics import log_loss, f1_score, accuracy_score, roc_auc_score


def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)

class elo_model:
    def __init__(self, target = 'mse', cache = True, prefit = False, alll = False, strike = True, grapp = True, ko = True, sub = True, sim = True, debug = False, sd = .025, n_sims = 100, fight_threshold = 1, strike_damper = .5, grappling_damper = .5, ko_damper = .5, sub_damper = .5, default_off_strike = .15, default_def_strike = .85, default_off_grappling = .15, default_def_grappling = .85, default_off_ko = .5, default_def_ko = .5, default_off_sub = .5, default_def_sub = .5):
        self.iter_scores = {'offStrikeElo':[], 'defStrikeElo':[], 'offGrapplingElo':[], 'defGrapplingElo':[], 'powerStrikeElo':[], 'chinStrikeElo':[], 'subGrapplingElo':[], 'evasGrapplingElo':[]}

        self.dampers = {'Strike': strike_damper, 'Grappling': grappling_damper, 'powerStrike': ko_damper, 'subGrappling': sub_damper}
        
        self.debug = debug
        
        self.defaults = {}
        self.defaults['offStrike'] = default_off_strike
        self.defaults['defStrike'] = default_def_strike
        self.defaults['offGrappling'] = default_off_grappling
        self.defaults['defGrappling'] = default_def_grappling
        self.defaults['powerStrike'] = default_off_ko
        self.defaults['chinStrike'] = default_def_ko
        self.defaults['subGrappling'] = default_off_sub
        self.defaults['evasGrappling'] = default_def_sub
        self.cache = cache
        if cache:
            self.fighter_db = {}
            self.bout_db = {}
        else:
            self.fighter_dc = None
            self.bout_db = None
            
        self.bout_info = None
        self.fighter_info = {}
        self.fighters = []
        self.round_dict = None
        self.bout_oid = None
        
        self.prefit = prefit
        self.target = target
        
        self.sd = sd
        self.n_sims = n_sims
        self.fight_threshold = fight_threshold
        self.raw_output = {}
        self.eval_data = {}
        
        self.strike = strike
        self.grapp = grapp
        self.ko = ko
        self.sub = sub
        self.sim = sim
        self.alll  = alll
        
        self.prefix = None
        if self.alll:
            self.prefix = 'all'
        elif self.strike:
            self.prefix = 'strike'
        elif self.grapp:
            self.prefix = 'grapp'
        elif self.ko:
            self.prefix = 'ko'
        elif self.sub:
            self.prefix = 'sub'
        self.params = {'defaults': self.defaults, 'dampers': self.dampers, 'sd': self.sd, 'n_sims': self.n_sims, 'fight_threshold': self.fight_threshold}
        
    def reset_bout(self):
        self.bout_info = None
        self.fighter_info = {}
        self.fighters = []
        self.round_dict = None
        self.eval_data = {}

    def get_truncated_normal(self, mean=0, low=0, upp=1):
        return truncnorm(
            (low - mean) / self.sd, (upp - mean) / self.sd, loc=mean, scale=self.sd) 
    
    def form_elo_bout_input(self):    
        self.eval_data = {}
        self.eval_data['schedRounds'] = self.bout_info['schedRounds']
        for fbx in self.bout_info['fighterBoutXRefs']:
            fighter_data = {}
            if fbx['mlOdds'] is None:
                continue
            fighter_data['mlOdds'] = fbx['mlOdds']
            if fbx['offStrikeEloPre'] is None:
                continue
            fighter_data['offStrikeEloPre'] = fbx['offStrikeEloPre']
            if fbx['defStrikeEloPre'] is None:
                continue
            fighter_data['defStrikeEloPre'] = fbx['defStrikeEloPre']
            if fbx['offGrapplingEloPre'] is None:
                continue
            fighter_data['offGrapplingEloPre'] = fbx['offGrapplingEloPre']
            if fbx['defGrapplingEloPre'] is None:
                continue
            fighter_data['defGrapplingEloPre'] = fbx['defGrapplingEloPre']
            if fbx['powerStrikeEloPre'] is None:
                continue
            fighter_data['powerStrikeEloPre'] = fbx['powerStrikeEloPre']
            if fbx['chinStrikeEloPre'] is None:
                continue
            fighter_data['chinStrikeEloPre'] = fbx['chinStrikeEloPre']
            if fbx['subGrapplingEloPre'] is None:
                continue
            fighter_data['subGrapplingEloPre'] = fbx['subGrapplingEloPre']
            if fbx['evasGrapplingEloPre'] is None:
                continue
            fighter_data['evasGrapplingEloPre'] = fbx['evasGrapplingEloPre']
            
            fighter_data['prev_fights'] = getLastEloCount(fbx['fighter']['oid'], self.bout_info['fightOid'])
            fighter_data['age'] = days_between(fbx['fighter']['dob'].split('T')[0], self.bout_info['fightDate'].split('T')[0])/365
            
            if fbx['outcome'] is None:
                continue
            if fbx['outcome'] not in ['W', 'L']:
                continue
            fighter_data['outcome'] = fbx['outcome']
            
            self.eval_data[fbx['fighter']['oid']] = fighter_data
            
    def _round_sim(self, fighters, fight):
        finish_ko_1 = False
        finish_ko_2 = False
        finish_sub_1 = False
        finish_sub_2 = False
        finish_1 = False
        finish_2 = False
        finish_1_val = 0
        finish_2_val = 0
        
        mc_vals = {i:{} for i in fighters}
        for fighter in fighters:
            for val in [ 'offStrikeEloPre', 'defStrikeEloPre',
                        'offGrapplingEloPre', 'defGrapplingEloPre',
                        'powerStrikeEloPre', 'chinStrikeEloPre',
                        'subGrapplingEloPre', 'evasGrapplingEloPre']:
                mc_vals[fighter][val] = self.get_truncated_normal(mean = fight[fighter][val]).rvs(1)[0]
        
        strike_1 = (mc_vals[fighters[0]]['offStrikeEloPre'] + (1 - mc_vals[fighters[1]]['defStrikeEloPre'])) / 2
        strike_2 = (mc_vals[fighters[1]]['offStrikeEloPre'] + (1 - mc_vals[fighters[0]]['defStrikeEloPre'])) / 2
     
        grapp_1 = (mc_vals[fighters[0]]['offGrapplingEloPre'] + (1 - mc_vals[fighters[1]]['defGrapplingEloPre'])) / 2
        grapp_2 = (mc_vals[fighters[1]]['offGrapplingEloPre'] + (1 - mc_vals[fighters[0]]['defGrapplingEloPre'])) / 2
        
        ko_1 = (mc_vals[fighters[0]]['powerStrikeEloPre'] + (1 - mc_vals[fighters[1]]['chinStrikeEloPre'])) / 2
        ko_2 = (mc_vals[fighters[1]]['powerStrikeEloPre'] + (1 - mc_vals[fighters[0]]['chinStrikeEloPre'])) / 2
     
        sub_1 = (mc_vals[fighters[0]]['subGrapplingEloPre'] + (1 - mc_vals[fighters[1]]['evasGrapplingEloPre'])) / 2
        sub_2 = (mc_vals[fighters[1]]['subGrapplingEloPre'] + (1 - mc_vals[fighters[0]]['evasGrapplingEloPre'])) / 2
        
        
        finish_ko_1_rand = random.uniform(0, 1)
        finish_ko_2_rand = random.uniform(0, 1)
        finish_sub_1_rand = random.uniform(0, 1)
        finish_sub_2_rand = random.uniform(0, 1)
        
        if finish_ko_1_rand > 1 - (ko_1 * strike_1):
            finish_ko_1 = True
            finish_1 = True
            finish_1_val = ko_1 * strike_1
        if finish_sub_1_rand > 1 - (sub_1 * grapp_1):
            if finish_1:
                if sub_1 * grapp_1 > finish_1_val:
                    finish_1_val = sub_1 * grapp_1
                    finish_ko_1 = False
            finish_sub_1 = True
            finish_2 = True
            
        if finish_ko_2_rand > 1 - (ko_2 * strike_2):
            finish_ko_2 = True
            finish_2 = True
            finish_2_val = ko_2 * strike_2
        if finish_sub_2_rand > 1 - (sub_2 * grapp_2):
            if finish_2:
                if sub_2 * grapp_2 > finish_2_val:
                    finish_2_val = sub_2 * grapp_2
                    finish_ko_2 = False
            finish_sub_2 = True
            finish_2 = True
    
        if finish_1 and finish_2:
            if finish_1_val > finish_2_val:
                finish_2 = False
                finish_sub_2 = False
                finish_ko_2 = False
            else:
                finish_1 = False
                finish_sub_2 = False
                finish_ko_2 = False
        
        round_sim_result = {fighters[0]: {'score': strike_1 + grapp_1, 'sub': finish_sub_1, 'ko': finish_ko_1, 'finish': finish_1},
                fighters[1]: {'score': strike_2 + grapp_2, 'sub': finish_sub_2, 'ko': finish_ko_2, 'finish': finish_2},
                }
        return round_sim_result

    def _fight_sim(self, input_vals):
        fighters, fight = input_vals
        fight_result = {i:{} for i in fighters}
        for fighter in fighters:
            for val in ['score', 'sub', 'ko', 'finish']:
                fight_result[fighter][val] = 0
            fight_result[fighter]['result'] = None
        winner_found = False
        for rnd in range(fight['schedRounds']):
            if winner_found:
                continue
            round_sim_res = self._round_sim(fighters, fight)
            if round_sim_res[fighters[0]]['finish']:
                winner_found = True
                fight_result[fighters[0]]['finish'] = 1
                if round_sim_res[fighters[0]]['sub']:
                    fight_result[fighters[0]]['sub'] = 1
                if round_sim_res[fighters[0]]['ko']:
                    fight_result[fighters[0]]['ko'] = 1 
                fight_result[fighters[0]]['result'] = 'W'
                fight_result[fighters[1]]['result'] = 'L'
                
    
            if round_sim_res[fighters[1]]['finish']:
                winner_found = True
                fight_result[fighters[1]]['finish'] = 1
                if round_sim_res[fighters[1]]['sub']:
                    fight_result[fighters[1]]['sub'] = 1
                if round_sim_res[fighters[1]]['ko']:
                    fight_result[fighters[1]]['ko'] = 1    
                fight_result[fighters[1]]['result'] = 'W'
                fight_result[fighters[0]]['result'] = 'L'
            
            if winner_found:
                break
            
            fight_result[fighters[0]]['score'] += round_sim_res[fighters[0]]['score']
            fight_result[fighters[1]]['score'] += round_sim_res[fighters[1]]['score']
        
        if not winner_found:
            if fight_result[fighters[0]]['score'] > fight_result[fighters[1]]['score']:
                fight_result[fighters[0]]['result'] = 'W'
                fight_result[fighters[1]]['result'] = 'L'
            else:
                fight_result[fighters[1]]['result'] = 'W'
                fight_result[fighters[0]]['result'] = 'L'     
                
        fight_result['finish_rounds'] = rnd + 1
        return fight_result
    
    def _single_fight_sim(self):
    #    n_sims, fighters, fight = 5000, fighters, fight
        
        sim_results = {i:{} for i in self.fighters}
        for fighter in self.fighters:
            for val in ['sub', 'ko', 'finish', 'win']:
                sim_results[fighter][val] = 0
        sim_results['rounds'] = []     
        
        jobs = [(self.fighters, self.eval_data) for i in range(self.n_sims)]
        batch_fight_results = Parallel(n_jobs = -1, verbose = 0)(delayed(self._fight_sim) (i) for i in jobs)
        for fight_result in batch_fight_results:
            for fighter in self.fighters:
                for val in ['sub', 'ko', 'finish']:
                    sim_results[fighter][val] += fight_result[fighter][val]
                if fight_result[fighter]['result'] == 'L':
                    sim_results[fighter]['win'] += 1
            sim_results['rounds'].append(fight_result['finish_rounds'])
        sim_results['rounds'] = np.mean(sim_results['rounds'])
        
        prediction = {i:{} for i in self.fighters}
        prediction[self.fighters[0]]['prob'] = sim_results[self.fighters[0]]['win']/self.n_sims
        prediction[self.fighters[1]]['prob'] = sim_results[self.fighters[1]]['win']/self.n_sims
        
        fighter_1_result = None
        fighter_2_result = None
        bet_outcome = None
        bet_dec_diff = None
    
        if self.eval_data[self.fighters[0]]['outcome'] == 'W':
            fighter_1_result = 1
        else:
            fighter_1_result = 0
                
        if self.eval_data[self.fighters[1]]['outcome'] == 'W':
            fighter_2_result = 1
        else:
            fighter_2_result = 0
                                
        if prediction[self.fighters[0]]['prob'] * 100 > self.eval_data[self.fighters[0]]['mlOdds']:
            bet_dec_diff = prediction[self.fighters[0]]['prob'] * 100 - self.eval_data[self.fighters[0]]['mlOdds']
            if self.eval_data[self.fighters[0]]['outcome'] == 'W':
                bet_outcome = calcWinnings(1, self.eval_data[self.fighters[0]]['mlOdds'])
            else:
                bet_outcome = -1
        elif prediction[self.fighters[1]]['prob'] * 100 > self.eval_data[self.fighters[1]]['mlOdds']:
            bet_dec_diff = prediction[self.fighters[1]]['prob'] * 100 - self.eval_data[self.fighters[1]]['mlOdds']
            if self.eval_data[self.fighters[1]]['outcome'] == 'W':
                bet_outcome = calcWinnings(1, self.eval_data[self.fighters[1]]['mlOdds'])
            else:
                bet_outcome = -1           
        else:
            bet_outcome = 0
            bet_dec_diff = 0
            
        bout_result = {'fighter_1_win_prob': prediction[self.fighters[0]]['prob'],
                       'fighter_2_win_prob': prediction[self.fighters[1]]['prob'],
                       'fighter_1_ml_odds': self.eval_data[self.fighters[0]]['mlOdds'],
                       'fighter_2_ml_odds': self.eval_data[self.fighters[1]]['mlOdds'],
                       'fighter_1_result': fighter_1_result,                  
                       'fighter_2_result': fighter_2_result,
                       'rounds': sim_results['rounds'],
                       'bet_outcome': bet_outcome,
                       'bet_dec_diff': bet_dec_diff
                       }
        self.raw_output[self.bout_oid] = bout_result
        
    def eval_bout(self):
        
        if self.sim:
            self.form_elo_bout_input()
            if len(self.eval_data) == 3:
                self._single_fight_sim()
                if len(self.fighters) != 2:
                    return
                if self.eval_data[self.fighters[0]]['mlOdds'] is None or self.eval_data[self.fighters[1]]['mlOdds'] is None:
                    return
                if self.eval_data[self.fighters[0]]['prev_fights'] < self.fight_threshold or self.eval_data[self.fighters[1]]['prev_fights'] < self.fight_threshold:
                    return
                if self.eval_data[self.fighters[0]]['prev_fights'] < self.fight_threshold or self.eval_data[self.fighters[1]]['prev_fights'] < self.fight_threshold:
                    return
                self._single_fight_sim()
            
    def simplify_round_stats(self, round_stat):
        simp_round_stat = {'tkoKo': round_stat['tkoKo'], 'submissionSuccessful': round_stat['submissionSuccessful'], 'koScore': round_stat['koScore'], 'submissionScore': round_stat['submissionScore']}
        if simp_round_stat['koScore'] is None or simp_round_stat['submissionScore'] is None:
            if self.debug:
                print("Null values in round stats")
            raise ValueError("Null values in round stats")
        return simp_round_stat

    def prep_round_stats(self):
        round_dict = {}
        for rnd in range(self.bout_info['finishRounds']):
            round_dict[rnd+1] = {i : {} for i in self.fighters}
            
        for fighter_id in self.fighters:
            for fighter_round in self.fighter_info[fighter_id]['stats']['boutDetails']:
                round_dict[fighter_round['round']][fighter_id] = self.simplify_round_stats(fighter_round)
        self.round_dict = round_dict

    def adj_finish(self, vals, pre_post):
#    off_fighter, def_fighter, act_score, act_val, pre_post, off_est_feat, def_est_feat, damper, iter_scores = fighter_1_elo, fighter_2_elo, vals[fighter_1['fighter']['oid']]['submissionScore'], vals[fighter_1['fighter']['oid']]['submissionSuccessful'], pre_post, 'subGrappling', 'evasGrappling', sub_finish_damper, iter_scores

        for off_est_feat, def_est_feat, act_score_feat, act_val_feat, proc in [('powerStrike', 'chinStrike', 'koScore', 'tkoKo', self.ko), ('subGrappling', 'evasGrappling', 'submissionScore', 'submissionSuccessful', self.sub)]:
            if proc:
                for fighter_id in self.fighters:      
                    off_id = fighter_id
                    def_id = [i for i in self.fighters if i != fighter_id][0]
                    
                    off_fighter = self.fighter_info[off_id]['elo']
                    def_fighter = self.fighter_info[def_id]['elo']
                    
                    act_score = vals[off_id][act_score_feat]
                    act_val = vals[off_id][act_val_feat]
        
                    valid = True
                    finish_odds = calc_sub_ko_odds(act_score, off_fighter['%sElo%s' % (off_est_feat, pre_post)], def_fighter['%sElo%s' % (def_est_feat, pre_post)])
                
                    self.iter_scores['%sElo' % (off_est_feat)].append(off_fighter['%sElo%s' % (off_est_feat, pre_post)] - finish_odds)
                    self.iter_scores['%sElo' % (def_est_feat)].append(def_fighter['%sElo%s' % (def_est_feat, pre_post)] - finish_odds)
                
                    if act_val == 1:
                        new_off_score = off_fighter['%sElo%s' % (off_est_feat, pre_post)] ** (1/(1+ ((1-finish_odds) * (abs(off_fighter['%sElo%s' % (off_est_feat, pre_post)] - finish_odds) * self.dampers[off_est_feat]))))
                        new_def_score = def_fighter['%sElo%s' % (def_est_feat, pre_post)] ** (1/(1 - (finish_odds) * (abs(def_fighter['%sElo%s' % (def_est_feat, pre_post)] - finish_odds) * self.dampers[off_est_feat] )))
                    else:
                        new_off_score = off_fighter['%sElo%s' % (off_est_feat,pre_post)] ** (1/(1 - ((1-finish_odds) * (abs(off_fighter['%sElo%s' % (off_est_feat, pre_post)] - finish_odds) * self.dampers[off_est_feat]))))
                        new_def_score = def_fighter['%sElo%s' % (def_est_feat, pre_post)] ** (1/(1 + (finish_odds) * (abs(def_fighter['%sElo%s' % (def_est_feat, pre_post)] - finish_odds) * self.dampers[off_est_feat] )))
                
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
                    self.fighter_info[off_id]['elo'] = off_fighter
                    def_fighter['%sEloPost' % (def_est_feat)] = new_def_score
                    self.fighter_info[def_id]['elo'] = def_fighter
    
    def adj_score(self, vals, pre_post):
        for score, est_feat, proc in [('koScore', 'Strike', self.strike), ('submissionScore', 'Grappling', self.grapp)]:
            if proc:
                for fighter_id in self.fighters:
                    off_id = fighter_id
                    def_id = [i for i in self.fighters if i != fighter_id][0]
                    
                    off_fighter = self.fighter_info[off_id]['elo']
                    def_fighter = self.fighter_info[def_id]['elo']
                    
                    act_val = vals[off_id][score]
                
                    valid = True
                    prob_damper = calc_prob_damper(act_val, off_fighter['off%sElo%s' % (est_feat, pre_post)], def_fighter['def%sElo%s' % (est_feat, pre_post)])
                    self.iter_scores['off%sElo' % (est_feat)].append(act_val - off_fighter['off%sElo%s' % (est_feat, pre_post)])
                    new_off_score = off_fighter['off%sElo%s' % (est_feat, pre_post)] ** (1/ (1 + (abs(act_val - off_fighter['off%sElo%s' % (est_feat, pre_post)]) * (prob_damper - .5) * self.dampers[est_feat])))
                    if (new_off_score < 0):
                        print('off%sEloPost would be negative (%.2f)' % (est_feat, new_off_score))
                        valid = False
                    if (new_off_score > 1):
                        print('off%sEloPost would be greater than 1 (%.2f)' % (est_feat, new_off_score))
                        valid = False        
                
                    self.iter_scores['def%sElo' % (est_feat)].append(def_fighter['def%sElo%s' % (est_feat, pre_post)] - (1-act_val))
                    new_def_score = def_fighter['def%sElo%s' % (est_feat, pre_post)] ** (1/ (1- (abs(def_fighter['def%sElo%s' % (est_feat, pre_post)] - (1-act_val)) * (prob_damper - .5) * self.dampers[est_feat])))
                    if (new_def_score < 0):
                        print('def%sEloPost would be negative (%.2f)' % (est_feat, new_def_score))
                        valid = False
                    if (new_def_score > 1):
                        print('def%sEloPost would be greater than 1 (%.2f)' % (est_feat, new_def_score))
                        valid = False        
                            
                    if valid == False:
                        raise ValueError("Elo scores must be between 0-1")
                    off_fighter['off%sEloPost' % (est_feat)] = new_off_score
                    self.fighter_info[off_id]['elo'] = off_fighter
                    def_fighter['def%sEloPost' % (est_feat)] = new_def_score
                    self.fighter_info[def_id]['elo'] = def_fighter
    
    #bout_info, index = bout_info, 0
    def prep_fighter(self, index):
        fighter = self.bout_info['fighterBoutXRefs'][index]
        fighter_elo = {}
        if not self.cache and not self.prefit and (fighter['offStrikeEloPost'] is not None or fighter['defStrikeEloPost'] is not None or fighter['offGrapplingEloPost'] is not None or fighter['defGrapplingEloPost'] is not None):
            if self.debug:
                print("Elo scores for bout %s and fighter %s already saved" % (self.bout_info['fightOid'], fighter['fighter']['oid']))
            raise ValueError("Elo scores for bout %s and fighter %s already saved" % (self.bout_info['fightOid'], fighter['fighter']['oid']))
        if self.cache:
            if fighter['fighter']['oid'] in self.fighter_db.keys() and len(self.fighter_db[fighter['fighter']['oid']]) != 0:
                fighter_prev_elo = self.fighter_db[fighter['fighter']['oid']][-1]
            else:
                self.fighter_db[fighter['fighter']['oid']] = []
                fighter_prev_elo = {'oid': None}
        else:
            req_fetch = False
            fighter_elo['oid'] = fighter['oid']
            for val in [ 'offStrikeEloPre', 'defStrikeEloPre',
                        'offGrapplingEloPre', 'defGrapplingEloPre',
                        'powerStrikeEloPre', 'chinStrikeEloPre',
                        'subGrapplingEloPre', 'evasGrapplingEloPre']:
                if fighter[val] is None:
                    req_fetch = True
                    break
                else:
                    fighter_elo[val] = fighter[val]
            if req_fetch:
                fighter_prev_elo = getLastElo(fighter['fighter']['oid'], self.bout_info['fightOid'])
        if fighter_prev_elo['oid'] is None:
            if self.debug:
                print("Initializing new elo baseline for fighter %s" % fighter['fighter']['oid'])      
            fighter_elo['oid'] =  fighter['oid']
            for elo_stat in self.defaults.keys():
                fighter_elo['%sEloPre'%(elo_stat)] = self.defaults[elo_stat]
                fighter_elo['%sEloPost'%(elo_stat)] = None
                self.bout_info['fighterBoutXRefs'][index]['%sEloPre'%(elo_stat)] = fighter_elo['%sEloPre'%(elo_stat)]            
        else:
            fighter_elo['oid'] =  fighter['oid']
            for elo_stat in self.defaults.keys():
                fighter_elo['%sEloPre'%(elo_stat)] = fighter_prev_elo['%sEloPost'%(elo_stat)]
                self.bout_info['fighterBoutXRefs'][index]['%sEloPre'%(elo_stat)] = fighter_elo['%sEloPre'%(elo_stat)]            
                fighter_elo['%sEloPost'%(elo_stat)] = None
        
        self.fighters.append(fighter['fighter']['oid'])
        self.fighter_info[fighter['fighter']['oid']] = {'stats': fighter, 'elo': fighter_elo}     
            

    def update_bout_result(self):
        if self.debug:
            print("Saving values for bout %s" % (self.bout_info['oid']))           
        if self.cache:
#            if self.fighters[0] not in db.keys():
            self.fighter_db[self.fighters[0]].append(self.fighter_info[self.fighters[0]]['elo'])
            self.fighter_db[self.fighters[1]].append(self.fighter_info[self.fighters[1]]['elo'])
            
            
            self.bout_db[self.bout_oid] = {self.fighters[0]: self.fighter_info[self.fighters[0]]['elo'], 
                                        self.fighters[1]: self.fighter_info[self.fighters[1]]['elo']
                                        }
        else:
            fighter_1_update = updateElo(self.fighter_info[self.fighters[0]]['elo'])
#            if self.debug:
#                print(self.fighter_info[self.fighters[0]]['elo'])
#                print(fighter_1_update)
            if fighter_1_update['errorMsg'] is not None:
                print(fighter_1_update['errorMsg'])
            fighter_2_update = updateElo(self.fighter_info[self.fighters[1]]['elo'])
            if fighter_2_update['errorMsg'] is not None:
                print(fighter_2_update['errorMsg'])
            
    def proc_bout(self):
        self.reset_bout()
        self.bout_info = refreshBout(self.bout_oid)
#        if self.bout_info['gender'] != 'MALE':
#            if self.debug:
#                print("Skipping bout %s.. not male" % (self.bout_info['oid']))
#            return
        try:
            self.prep_fighter(0)
            self.prep_fighter(1)
        except ValueError:
            if self.debug:
                print("Error pulling fighter data for bout %s" % (self.bout_info['oid']))            
            return
        
        try:
            self.prep_round_stats()    
        except:
            if self.debug:
                print("Error pulling round data for bout %s" % (self.bout_info['oid']))
            return                
            
        self.eval_bout()
        if not self.prefit:
            for rnd, vals in self.round_dict.items():
                if rnd == 1:
                    pre_post = 'Pre'
                else:
                    pre_post = 'Post'
                    
                self.adj_score(vals, pre_post)                                                                                                        
                self.adj_finish(vals, pre_post)
            
            self.update_bout_result()

 
    def train(self, bouts):
        self.model_id = str(uuid.uuid4())
        if self.cache:
            self.fighter_db = {}
            self.bout_db = {}
        else:
            self.fighter_dc = None
            self.bout_db = None
        tot = len(bouts)
        for n,(bout) in enumerate(bouts):
            self.bout_oid = bout
            self.proc_bout()
            progress(n+1, tot)

    def score(self):
        
        if self.sim:
            df = pd.DataFrame.from_dict(self.raw_output).T
            testy = df['fighter_2_result'].values
            bin_pred = [1 if i > .5 else 0 for i in df['fighter_2_win_prob'].values]
            
            self.scores = {}
            self.scores['logloss'] = log_loss(testy, df[['fighter_1_win_prob', 'fighter_2_win_prob']].values) * -1
            self.scores['f1'] = f1_score(testy, bin_pred)
            self.scores['roc'] = roc_auc_score(testy, bin_pred)
            self.scores['acc'] = accuracy_score(testy, bin_pred)
            self.scores['cash'] = df['bet_outcome'].sum()
    
            with open('./elo/scores/%s/%s.json' % (self.prefix, self.model_id), 'w') as f:
                json.dump(self.scores, f)
            with open('./elo/models/%s/%s.json' % (self.prefix, self.model_id), 'w') as f:
                json.dump(self.params, f)
        else:
#            print(self.iter_scores)
            with open('./elo/models/%s/%s.json' % (self.prefix, self.model_id), 'w') as f:
                json.dump(self.params, f)
            with open('./elo/scores/%s/%s.json' % (self.prefix, self.model_id), 'w') as f:
                json.dump(self.iter_scores, f)
            gross_scores = []
            for k,v in self.iter_scores.items():
                if len(v) != 0:
                    mean = np.mean([(i)**2 for i in self.iter_scores[k]])
                    gross_scores.append(mean)
            self.scores = {'mse': np.mean(gross_scores) * -1}
        return self.scores[self.target]
        
def _opt_strike(trial):
    params = {
        'strike_damper': trial.suggest_loguniform('strike_damper', .4, 1),
        'default_off_strike': trial.suggest_loguniform('default_off_strike', 1e-2, .2),
        'default_def_strike': trial.suggest_loguniform('default_def_strike', .65, 1)
    }
    
#    params['elo_n_sims'] = 500
    BOUTS = getAllBouts()
    elo_mod = elo_model(cache = True,
                            debug = False,
                            strike_damper = params['strike_damper'],
                            default_off_strike = params['default_off_strike'], 
                            default_def_strike = params['default_def_strike'],
                            strike = True, 
                            grapp = False,
                            ko = False, 
                            sub = False, 
                            sim = False
                            )
    elo_mod.train(BOUTS)
    return elo_mod.score()     

def _opt_grapp(trial):
    params = {
        'grapp_damper': trial.suggest_loguniform('grapp_damper', .4, 1),
        'default_off_grapp': trial.suggest_loguniform('default_off_grapp', 1e-2, .2),
        'default_def_grapp': trial.suggest_loguniform('default_def_grapp', .65, 1)
    }
    
#    params['elo_n_sims'] = 500
    BOUTS = getAllBouts()
    elo_mod = elo_model(cache = True,
                            debug = False,
                            grappling_damper = params['grapp_damper'],
                            default_off_grappling = params['default_off_grapp'], 
                            default_def_grappling = params['default_def_grapp'],
                            strike = False, 
                            grapp = True,
                            ko = False, 
                            sub = False, 
                            sim = False
                            )
    elo_mod.train(BOUTS)
    return elo_mod.score()    

def _opt_ko(trial):
    params = {
        'ko_finish_damper': trial.suggest_loguniform('ko_finish_damper', .4, 1),
        'default_off_ko': trial.suggest_loguniform('default_off_ko', 1e-1, .5),
        'default_def_ko': trial.suggest_loguniform('default_def_ko', 1e-2, .2),
    }
    
#    params['elo_n_sims'] = 500
    BOUTS = getAllBouts()
    elo_mod = elo_model(cache = True,
                            debug = False,
                            ko_damper = params['ko_finish_damper'],
                            default_off_ko = params['default_off_ko'], 
                            default_def_ko = params['default_def_ko'], 
                            strike = False, 
                            grapp = False,
                            ko = True, 
                            sub = False, 
                            sim = False
                            )
    elo_mod.train(BOUTS)
    return elo_mod.score()  

def _opt_sub(trial):
    params = {
        'sub_finish_damper': trial.suggest_loguniform('sub_finish_damper', .4, 1),
        'default_off_sub': trial.suggest_loguniform('default_off_sub', 1e-1, .5),
        'default_def_sub': trial.suggest_loguniform('default_def_sub', 1e-2, .2),
    }
    
#    params['elo_n_sims'] = 500
    BOUTS = getAllBouts()
    elo_mod = elo_model(cache = True,
                            debug = False,
                            sub_damper = params['sub_finish_damper'],
                            default_off_sub = params['default_off_sub'], 
                            default_def_sub = params['default_def_sub'], 
                            strike = False, 
                            grapp = False,
                            ko = False, 
                            sub = True, 
                            sim = False
                            )
    elo_mod.train(BOUTS)
    return elo_mod.score()  
            

#    return elo_mod.score()           
#            
#    -0.3964006233940943 == -0.1531207563539216
#        else:
#            elo_mod.iter_scores[k] = np.mean(elo_mod.iter_scores[k])
#    scores = elo_mod.iter_scores
#    return elo_mod.scores['cash']

def _opt_elo(trial):
    params = {
        'strike_damper': trial.suggest_loguniform('strike_damper', .4, 1),
        'default_off_strike': trial.suggest_loguniform('default_off_strike', 1e-2, .2),
        'default_def_strike': trial.suggest_loguniform('default_def_strike', .65, 1),
        
        'grapp_damper': trial.suggest_loguniform('grapp_damper', .4, 1),
        'default_off_grapp': trial.suggest_loguniform('default_off_grapp', 1e-2, .2),
        'default_def_grapp': trial.suggest_loguniform('default_def_grapp', .65, 1),
        
        'ko_finish_damper': trial.suggest_loguniform('ko_finish_damper', .4, 1),
        'default_off_ko': trial.suggest_loguniform('default_off_ko', 1e-1, .5),
        'default_def_ko': trial.suggest_loguniform('default_def_ko', 1e-2, .2),

        'sub_finish_damper': trial.suggest_loguniform('sub_finish_damper', .4, 1),
        'default_off_sub': trial.suggest_loguniform('default_off_sub', 1e-1, .5),
        'default_def_sub': trial.suggest_loguniform('default_def_sub', 1e-2, .2),

        'elo_fight_treshold': trial.suggest_int('elo_fight_treshold', 2, 10),
        'elo_sd': trial.suggest_loguniform('elo_sd', 1e-2, 1e-1)
    }
    
    params['elo_n_sims'] = 500
    BOUTS = getAllBouts()
    elo_mod = elo_model(cache = True,
                            debug = False,
                            sd = params['elo_sd'], 
                            n_sims = params['elo_n_sims'],
                            fight_threshold = params['elo_fight_treshold'],
                            strike_damper = params['strike_damper'],
                            grappling_damper = params['grapp_damper'], 
                            ko_damper = params['ko_finish_damper'], 
                            sub_damper = params['sub_finish_damper'], 
                            default_off_strike = params['default_off_strike'], 
                            default_def_strike = params['default_def_strike'], 
                            default_off_grappling = params['default_off_grapp'], 
                            default_def_grappling = params['default_def_grapp'], 
                            default_off_ko = params['default_off_ko'], 
                            default_def_ko = params['default_def_ko'], 
                            default_off_sub = params['default_off_sub'], 
                            default_def_sub = params['default_def_sub'])    
    elo_mod.train(BOUTS)
    elo_mod.score()
    return elo_mod.scores['cash']
    
def populate_elo(bouts = None, refit = False):
    if not os.path.exists("predictors/elo/elo-config.json"):
        gen_config()
    with open('predictors/elo/elo-config.json', 'r') as f:
        params = json.load(f)
    
    if refit:
        clearElo()
        BOUTS = getAllBouts()
    else:
        BOUTS = bouts
        
    elo_mod = elo_model(cache = False,
                            debug = True,
#                            sd = params['elo_sd'], 
#                            n_sims = params['elo_n_sims'],
#                            fight_threshold = params['elo_fight_treshold'],
                            strike_damper = params['strike_damper'],
                            grappling_damper = params['grapp_damper'], 
                            ko_damper = params['ko_finish_damper'], 
                            sub_damper = params['sub_finish_damper'], 
                            default_off_strike = params['default_off_strike'], 
                            default_def_strike = params['default_def_strike'], 
                            default_off_grappling = params['default_off_grapp'], 
                            default_def_grappling = params['default_def_grapp'], 
                            default_off_ko = params['default_off_ko'], 
                            default_def_ko = params['default_def_ko'], 
                            default_off_sub = params['default_off_sub'], 
                            default_def_sub = params['default_def_sub'],
                            sim = False
                            )    
    elo_mod.train(BOUTS)    
   
#def _opt_monte_carlo(trial):
#    potential_params = {}
#    monte_carlo_params = {}
#    potential_params['elo_sd'] = np.linspace(.005, .1, 10)
#    potential_params['elo_n_sims'] = [1]
#    potential_params['elo_fight_treshold'] = [i for i in range(2,10)]    
#
#    for k,v in potential_params.items():
#        monte_carlo_params[k] = random.choice(v)
#    
#    if not os.path.exists("elo/elo-config.json"):
#        gen_config()
#    with open('elo/elo-config.json', 'r') as f:
#        elo_params = json.load(f)    
#    
#    BOUTS = getAllBouts()
#    elo_mod = elo_model(cache = False,
#                            debug = False,
#                            sd = monte_carlo_params['elo_sd'], 
#                            n_sims = 2,
#                            fight_threshold = monte_carlo_params['elo_fight_treshold'],
#                            strike_damper = elo_params['strike_damper'],
#                            grappling_damper = elo_params['grapp_damper'], 
#                            ko_damper = elo_params['ko_finish_damper'], 
#                            sub_damper = elo_params['sub_finish_damper'], 
#                            default_off_strike = elo_params['default_off_strike'], 
#                            default_def_strike = elo_params['default_def_strike'], 
#                            default_off_grappling = elo_params['default_off_grapp'], 
#                            default_def_grappling = elo_params['default_def_grapp'], 
#                            default_off_ko = elo_params['default_off_ko'], 
#                            default_def_ko = elo_params['default_def_ko'], 
#                            default_off_sub = elo_params['default_off_sub'], 
#                            default_def_sub = elo_params['default_def_sub'],
#                            sim = True,
#                            prefit = True, 
#                            alll = True,
#                            target = 'cash'
#                            )    
#    elo_mod.train(BOUTS)       
#    return elo_mod.score()

def _opt_monte_carlo(trial):
    monte_carlo_params = {
            'elo_sd': trial.suggest_loguniform('elo_sd', .001, .05),
            'elo_fight_treshold': trial.suggest_int('elo_fight_treshold', 4, 8)
            } 
    
    if not os.path.exists("predictors/elo/elo-config.json"):
        gen_config()
    with open('predictors/elo/elo-config.json', 'r') as f:
        elo_params = json.load(f)    
    
    BOUTS = getYearBouts()
    elo_mod = elo_model(cache = False,
                            debug = False,
                            sd = monte_carlo_params['elo_sd'], 
                            n_sims = 250,
                            fight_threshold = monte_carlo_params['elo_fight_treshold'],
                            strike_damper = elo_params['strike_damper'],
                            grappling_damper = elo_params['grapp_damper'], 
                            ko_damper = elo_params['ko_finish_damper'], 
                            sub_damper = elo_params['sub_finish_damper'], 
                            default_off_strike = elo_params['default_off_strike'], 
                            default_def_strike = elo_params['default_def_strike'], 
                            default_off_grappling = elo_params['default_off_grapp'], 
                            default_def_grappling = elo_params['default_def_grapp'], 
                            default_off_ko = elo_params['default_off_ko'], 
                            default_def_ko = elo_params['default_def_ko'], 
                            default_off_sub = elo_params['default_off_sub'], 
                            default_def_sub = elo_params['default_def_sub'],
                            sim = True,
                            prefit = True, 
                            alll = True,
                            target = 'logloss'
                            )    
    elo_mod.train(BOUTS)       
    return elo_mod.score()
    
def optimize_elo(domain, trials = 300):
    study = optuna.create_study(direction='maximize')
    if domain.upper() == 'STRIKE':
        study.optimize(_opt_strike, n_trials=trials)      
    elif domain.upper() == 'GRAPP':
        study.optimize(_opt_grapp, n_trials = trials)
    elif domain.upper() == 'KO':
        study.optimize(_opt_ko, n_trials = trials)
    elif domain.upper() == 'SUB':
        study.optimize(_opt_sub, n_trials = trials)
    elif domain.upper() == 'ALL':
        study.optimize(_opt_monte_carlo, n_trials = trials)

        
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    
    
def _opt_test():
    params = {}
    potential_params = {
        'ko_finish_damper': np.linspace(.05, .95, 10),
        'default_off_strike': np.linspace(.05, .95, 10),
        'default_def_strike': np.linspace(.05, .95, 10)
    }
    
    for k,v in potential_params.items():
        params[k] = random.choice(v)
    params['elo_n_sims'] = 500
    BOUTS = getAllBouts()
    elo_mod = elo_model(cache = True,
                            debug = True,
                            strike_damper = params['ko_finish_damper'],
                            default_off_strike = params['default_off_strike'], 
                            default_def_strike = params['default_def_strike'],
                            strike = True, 
                            grapp = False,
                            ko = False, 
                            sub = False, 
                            sim = False
                            )
    elo_mod.train(BOUTS)
    
    elo_mod.fighter_db
    elo_mod.fighters
    
    elo_mod.reset_bout()
    elo_mod.bout_info = refreshBout(elo_mod.bout_oid)
    if elo_mod.bout_info['gender'] != 'MALE':
        if elo_mod.debug:
            print("Skipping bout %s.. not male" % (elo_mod.bout_info['oid']))
        
    try:
        prep_fighter_test(elo_mod, 0)
        prep_fighter_test(elo_mod, 1)
    except ValueError:
        if elo_mod.debug:
            print("Error pulling fighter data for bout %s" % (elo_mod.bout_info['oid']))            
        
    
    try:
        elo_mod.prep_round_stats()    
    except:
        if elo_mod.debug:
            print("Error pulling round data for bout %s" % (elo_mod.bout_info['oid']))
                        
        
    elo_mod.eval_bout()
    if not elo_mod.prefit:
        for rnd, vals in elo_mod.round_dict.items():
            if rnd == 1:
                pre_post = 'Pre'
            else:
                pre_post = 'Post'
                
            elo_mod.adj_score(vals, pre_post)                                                                                                        
            elo_mod.adj_finish(vals, pre_post)
        
        elo_mod.update_bout_result()
    
    print(elo_mod.score())
    scores = elo_mod.iter_scores
    
#    elo_mod, index = elo_mod, 0
def prep_fighter_test(elo_mod, index):
    fighter = elo_mod.bout_info['fighterBoutXRefs'][index]
    if not elo_mod.cache and not elo_mod.prefit and (fighter['offStrikeEloPost'] is not None or fighter['defStrikeEloPost'] is not None or fighter['offGrapplingEloPost'] is not None or fighter['defGrapplingEloPost'] is not None):
        if elo_mod.debug:
            print("Elo scores for bout %s and fighter %s already saved" % (elo_mod.bout_info['fightOid'], fighter['fighter']['oid']))
        raise ValueError("Elo scores for bout %s and fighter %s already saved" % (elo_mod.bout_info['fightOid'], fighter['fighter']['oid']))
    if elo_mod.cache:
        fighter_elo = {}
        if fighter['fighter']['oid'] in elo_mod.fighter_db.keys() and len(elo_mod.fighter_db[fighter['fighter']['oid']]) != 0:
            fighter_prev_elo = elo_mod.fighter_db[fighter['fighter']['oid']][-1]
        else:
            elo_mod.fighter_db[fighter['fighter']['oid']] = []
            fighter_prev_elo = {'oid': None}
    else:
        req_fetch = False
        fighter_elo = {}
        fighter_elo['oid'] = fighter['oid']
        for val in [ 'offStrikeEloPre', 'defStrikeEloPre',
                    'offGrapplingEloPre', 'defGrapplingEloPre',
                    'powerStrikeEloPre', 'chinStrikeEloPre',
                    'subGrapplingEloPre', 'evasGrapplingEloPre']:
            if fighter[val] is None:
                req_fetch = True
                break
            else:
                fighter_elo[val] = fighter[val]
        if req_fetch:
            fighter_prev_elo = getLastElo(fighter['fighter']['oid'], elo_mod.bout_info['fightOid'])
            fighter_elo = {}
    if fighter_prev_elo['oid'] is None:
        if elo_mod.debug:
            print("Initializing new elo baseline for fighter %s" % fighter['fighter']['oid'])      
        fighter_elo['oid'] =  fighter['oid']
        for elo_stat in elo_mod.defaults.keys():
            fighter_elo['%sEloPre'%(elo_stat)] = elo_mod.defaults[elo_stat]
            fighter_elo['%sEloPost'%(elo_stat)] = None
            elo_mod.bout_info['fighterBoutXRefs'][index]['%sEloPre'%(elo_stat)] = fighter_elo['%sEloPre'%(elo_stat)]            
    else:
        fighter_elo['oid'] =  fighter['oid']
        for elo_stat in elo_mod.defaults.keys():
            fighter_elo['%sEloPre'%(elo_stat)] = fighter_prev_elo['%sEloPost'%(elo_stat)]
            elo_mod.bout_info['fighterBoutXRefs'][index]['%sEloPre'%(elo_stat)] = fighter_elo['%sEloPre'%(elo_stat)]            
            fighter_elo['%sEloPost'%(elo_stat)] = None
    
    elo_mod.fighters.append(fighter['fighter']['oid'])
    elo_mod.fighter_info[fighter['fighter']['oid']] = {'stats': fighter, 'elo': fighter_elo}     