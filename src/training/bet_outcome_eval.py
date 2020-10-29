#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 01:12:08 2020

@author: eric.hensleyibm.com
"""

import sys, os
if __name__ == "__main__":
    sys.path.append("src")
    os.environ['ufc.flask.spring.host'] = 'http://192.168.1.64:4646'#'http://68.248.220.199:4646'
    os.environ['ufc.flask.spring.pw'] = '1234'

    print(os.environ)
    
import numpy as np
np.random.seed(1108)
from utils.general import calcWinnings
from spring.api_wrappers import getTrainingFights, refreshBout, getBoutsFromFight, getLastEloCount
import json
import uuid
import optuna
from os import listdir
from os.path import isfile, join
import pandas as pd
from scipy.stats import percentileofscore

def evalActualWinner(bout):
    if (bout['fighterBoutXRefs'][0]['outcome'] == 'W'):
        return bout['fighterBoutXRefs'][0]['fighter']['fighterId']
    else:
        return bout['fighterBoutXRefs'][1]['fighter']['fighterId']
    
class bet_eval:
    def __init__(self, debug = False, bet_female = False, bet_ceiling = None, standard_wager = 1, start_bank = 0, bet_intercept = 1, conf_diff_lin = 0, conf_diff_quad = 0, num_fight_lin = 0, num_fight_quad = 0, diff_ceiling = .5, diff_floor = .05, prev_fight_floor = 5, prev_fight_ceiling = 99):
        self.bank = start_bank
        self.standard_wager = standard_wager

        self.fight_id_list = []
        self.bank_log = []
        self.diff_ceiling = diff_ceiling
        self.diff_floor = diff_floor
        self.prev_fight_floor = prev_fight_floor
        self.prev_fight_ceiling = prev_fight_ceiling
        self.bet_intercept = bet_intercept
        self.conf_diff_lin = conf_diff_lin
        self.conf_diff_quad = conf_diff_quad
        self.num_fight_lin = num_fight_lin
        self.num_fight_quad = num_fight_quad
        self.debug = debug
        self.bet_ceiling = bet_ceiling
        self.bet_female = bet_female
        
        self.results = []
        self.model_id = str(uuid.uuid4())
        self.params = {'model_id': self.model_id,
                        'diff_ceiling': self.diff_ceiling,
                       'diff_floor': self.diff_floor,
                       'prev_fight_floor': self.prev_fight_floor,
                       'prev_fight_ceiling': self.prev_fight_ceiling,
                       'bet_intercept': self.bet_intercept,
                       'conf_diff_lin' : self.conf_diff_lin,
                       'conf_diff_quad': self.conf_diff_quad,
                       'num_fight_lin': self.num_fight_lin,
                       'num_fight_quad': self.num_fight_quad,
                       'standard_wager': self.standard_wager,
                       'bet_ceiling': self.bet_ceiling,
                       'bet_female': self.bet_female
                       }
        
        self._skip = False
        self._bout_data = {}
        self.score = None
        self._predictions = {}
        self._bout_info = None

    def _wager_funct(self, conf_diff, f1_num, f2_num):
        
#        print(conf_diff)
#        print(f1_num)
#        print(f2_num)
#
#        conf_diff_lin_comp = self.conf_diff_lin * conf_diff
#        conf_diff_quad_comp = self.conf_diff_quad * (conf_diff**2)
#        f1_num_fight_lin_comp = self.num_fight_lin * f1_num * 2
#        f2_num_fight_lin_comp = self.num_fight_lin * f2_num * 2
#        f1_num_fight_quad_comp = (f1_num**2) * self.num_fight_quad * 2
#        f2_num_fight_quad_comp = (f2_num**2) * self.num_fight_quad * 2
#        
#        print(conf_diff_lin_comp)
#        print(conf_diff_quad_comp)
#        print(f1_num_fight_lin_comp)
#        print(f2_num_fight_lin_comp)
#        print(f1_num_fight_quad_comp)
#        print(f2_num_fight_quad_comp)
#        print(self.bet_intercept + conf_diff_lin_comp + conf_diff_quad_comp + f1_num_fight_lin_comp + f2_num_fight_lin_comp + f1_num_fight_quad_comp + f2_num_fight_quad_comp)
        bet_mult = self.bet_intercept + (self.conf_diff_lin * conf_diff) + (self.conf_diff_quad * (conf_diff**2)) + (self.num_fight_lin * f1_num) + (self.num_fight_quad * (f1_num**2)) + (self.num_fight_lin * f2_num) + (self.num_fight_quad * (f2_num**2)) + (f1_num * self.num_fight_lin) + ((f1_num**2) * self.num_fight_quad) + (f2_num * self.num_fight_lin) + ((f2_num**2) * self.num_fight_quad)
        to_wager = bet_mult * self.standard_wager
#        print(to_wager)
#        print("%s + (%s * %s) + (%s * (%s**2)) + (%s * %s) + (%s * (%s**2)) + (%s * %s) + (%s * (%s**2)))" % (self.bet_intercept, self.conf_diff_lin, conf_diff, self.conf_diff_quad, conf_diff, self.num_fight_lin, f1_num, self.num_fight_quad, f1_num, self.num_fight_lin, f2_num, self.num_fight_quad, f2_num))
#        print("$%s" % (to_wager))
        if to_wager < 0:
            to_wager = 0
        if self.bet_ceiling is not None and to_wager > self.bet_ceiling:
            to_wager = self.bet_ceiling
        return to_wager

    def _reset_bout(self):
        self._skip = False
        self._bout_data = {}
        self._bout_info = None
        
    def _score_pred(self, f, fbx):
        if fbx['outcome'] == 'W':
            self._bout_data['%s_outcome' % (f)] = 1
        elif fbx['outcome'] == 'L':
            self._bout_data['%s_outcome' % (f)] = 0
        elif fbx['outcome'] == 'D' or fbx['outcome'] == 'NC':
            if self.debug:
                print('OUTCOME - skipping bout as outcome = %s' % (fbx['outcome']))
            self._skip = True
            return
        else:
            if self.debug:
                print('OUTCOME - skipping bout as outcome = %s' % (fbx['outcome']))
            
    def _predict(self, f, fbx):
        if fbx['expOdds'] is None:
            self._skip = True
#            print(fbx)
#            print(' missing expected odds ')
            return
        self._bout_data['%s_prev_fights' % (f)] = getLastEloCount(fbx['fighter']['oid'], self._bout_info['fightOid'])
        
        if self._bout_data['%s_prev_fights' % (f)] < self.prev_fight_floor or self._bout_data['%s_prev_fights' % (f)] > self.prev_fight_ceiling:
            if self.debug:
                print('PREV FIGHT - skipping bout as %.2f fails (%.2f, %.2f) -- %s' % (self._bout_data['%s_prev_fights' % (f)], self.prev_fight_floor, self.prev_fight_ceiling, fbx['fighter']['fighterName']))
            self._skip = True
            return
        
        self._bout_data['%s_ml_odds' % (f)] = float(fbx['mlOdds'])
        self._bout_data['%s_exp_prob' % (f)] = fbx['expOdds'] * 100
#        print(self._bout_data)


        self._bout_data['%s_odds_diff' % (f)] = self._bout_data['%s_exp_prob' % (f)] - self._bout_data['%s_ml_odds' % (f)]
        
        if self.score and self._bout_data['%s_odds_diff' % (f)] > 0 and (self._bout_data['%s_odds_diff' % (f)] < self.diff_floor * 100 or self._bout_data['%s_odds_diff' % (f)] > self.diff_ceiling * 100):
            if self.debug:
                print('ODDS DIFF - skipping bout as %.2f fails (%.2f, %.2f)' % (self._bout_data['%s_odds_diff' % (f)], self.diff_floor * 100, self.diff_ceiling * 100))
            self._skip = True
            return    
                
                
    def _proc_bout(self, bout_id):
        self._reset_bout()
        self._bout_info = refreshBout(bout_id)
        if self.debug:
            print('%s VS %s' % (self._bout_info['fighterBoutXRefs'][0]['fighter']['fighterName'], self._bout_info['fighterBoutXRefs'][1]['fighter']['fighterName']))
        if not self.bet_female and self._bout_info['gender'] == 'MALE':
            for f, (fbx) in enumerate(self._bout_info['fighterBoutXRefs']):
                self._predict(f, fbx)
                if self.score:
                    self._score_pred(f, fbx)
            if self._skip:
                if self.debug:
                    print('skipping bout due to filter constraints')
                return

            if self.score:
                bet_result = 0
            else:
                bout_preds = {}
                
            if self._bout_data['0_odds_diff'] > 0:
                if self.score:
                    if self._bout_data['0_outcome'] == 1:
                        bet_result = calcWinnings(self._wager_funct(self._bout_data['0_odds_diff'], self._bout_data['0_prev_fights'], self._bout_data['1_prev_fights']), self._bout_data['0_ml_odds'])
                    else:
                        bet_result = -1 * self._wager_funct(self._bout_data['0_odds_diff'], self._bout_data['0_prev_fights'], self._bout_data['1_prev_fights'])
                else:
                    bout_preds['sug_wager'] = self._wager_funct(self._bout_data['0_odds_diff'], self._bout_data['0_prev_fights'], self._bout_data['1_prev_fights'])
                    bout_preds['pred_winner_oid'] = self._bout_info['fighterBoutXRefs'][0]['fighter']['oid']
                    bout_preds['pred_winner_name'] = self._bout_info['fighterBoutXRefs'][0]['fighter']['fighterName']
                    bout_preds['bout_name'] = '%s vs %s' % (self._bout_info['fighterBoutXRefs'][0]['fighter']['fighterName'], self._bout_info['fighterBoutXRefs'][1]['fighter']['fighterName'])
                    bout_preds['ml_odds'] = self._bout_data['0_ml_odds']
                    bout_preds['exp_prob'] = self._bout_data['0_exp_prob']
                    bout_preds['odds_diff'] = self._bout_data['0_odds_diff']
                    bout_preds['exp_outcome'] = ((bout_preds['exp_prob']/100) * calcWinnings(bout_preds['sug_wager'], bout_preds['ml_odds'])) - ((1-(bout_preds['exp_prob']/100)) * bout_preds['sug_wager'])
                    bout_preds['exp_roi'] = bout_preds['exp_outcome'] / bout_preds['sug_wager']
                    if self._bout_data['0_odds_diff'] < self.diff_floor * 100 or self._bout_data['0_odds_diff'] > self.diff_ceiling * 100:
                        bout_preds['BET'] = 'No'
                    else:
                        bout_preds['BET'] = 'Yes'
                        
            elif self._bout_data['1_odds_diff'] > 0:
                if self.score:
                    if self._bout_data['1_outcome'] == 1:
                        bet_result = calcWinnings(self._wager_funct(self._bout_data['1_odds_diff'], self._bout_data['0_prev_fights'], self._bout_data['1_prev_fights']), self._bout_data['1_ml_odds'])
                    else:
                        bet_result = -1 * self._wager_funct(self._bout_data['1_odds_diff'], self._bout_data['0_prev_fights'], self._bout_data['1_prev_fights'])              
                else:
                    bout_preds['sug_wager'] = self._wager_funct(self._bout_data['1_odds_diff'], self._bout_data['0_prev_fights'], self._bout_data['1_prev_fights'])
                    bout_preds['pred_winner_oid'] = self._bout_info['fighterBoutXRefs'][1]['fighter']['oid']
                    bout_preds['pred_winner_name'] = self._bout_info['fighterBoutXRefs'][1]['fighter']['fighterName']
                    bout_preds['bout_name'] = '%s vs %s' % (self._bout_info['fighterBoutXRefs'][0]['fighter']['fighterName'], self._bout_info['fighterBoutXRefs'][1]['fighter']['fighterName'])
                    bout_preds['ml_odds'] = self._bout_data['1_ml_odds']
                    bout_preds['exp_prob'] = self._bout_data['1_exp_prob']
                    bout_preds['odds_diff'] = self._bout_data['1_odds_diff']
                    bout_preds['exp_outcome'] = ((bout_preds['exp_prob']/100) * calcWinnings(bout_preds['sug_wager'], bout_preds['ml_odds'])) - ((1-(bout_preds['exp_prob']/100)) * bout_preds['sug_wager'])
                    bout_preds['exp_roi'] = bout_preds['exp_outcome'] / bout_preds['sug_wager']
                    if self._bout_data['1_odds_diff'] < self.diff_floor * 100 or self._bout_data['1_odds_diff'] > self.diff_ceiling * 100:
                        bout_preds['BET'] = 'No'
                    else:
                        bout_preds['BET'] = 'Yes'
            else:
                if not self.score:
                    if self._bout_data['0_odds_diff'] > self._bout_data['1_odds_diff']:
                        bout_preds['sug_wager'] = self._wager_funct(self._bout_data['0_odds_diff'], self._bout_data['0_prev_fights'], self._bout_data['1_prev_fights'])
                        bout_preds['pred_winner_oid'] = self._bout_info['fighterBoutXRefs'][0]['fighter']['oid']
                        bout_preds['pred_winner_name'] = self._bout_info['fighterBoutXRefs'][0]['fighter']['fighterName']
                        bout_preds['bout_name'] = '%s vs %s' % (self._bout_info['fighterBoutXRefs'][0]['fighter']['fighterName'], self._bout_info['fighterBoutXRefs'][1]['fighter']['fighterName'])
                        bout_preds['ml_odds'] = self._bout_data['0_ml_odds']
                        bout_preds['exp_prob'] = self._bout_data['0_exp_prob']
                        bout_preds['odds_diff'] = self._bout_data['0_odds_diff']
                        bout_preds['exp_outcome'] = ((bout_preds['exp_prob']/100) * calcWinnings(bout_preds['sug_wager'], bout_preds['ml_odds'])) - ((1-(bout_preds['exp_prob']/100)) * bout_preds['sug_wager'])
                        if self._bout_data['0_odds_diff'] < self.diff_floor * 100 or self._bout_data['0_odds_diff'] > self.diff_ceiling * 100:
                            bout_preds['BET'] = 'No'
                        else:
                            bout_preds['BET'] = 'Yes'      
                    else:
                        bout_preds['sug_wager'] = self._wager_funct(self._bout_data['1_odds_diff'], self._bout_data['0_prev_fights'], self._bout_data['1_prev_fights'])
                        bout_preds['pred_winner_oid'] = self._bout_info['fighterBoutXRefs'][1]['fighter']['oid']
                        bout_preds['pred_winner_name'] = self._bout_info['fighterBoutXRefs'][1]['fighter']['fighterName']
                        bout_preds['bout_name'] = '%s vs %s' % (self._bout_info['fighterBoutXRefs'][0]['fighter']['fighterName'], self._bout_info['fighterBoutXRefs'][1]['fighter']['fighterName'])
                        bout_preds['ml_odds'] = self._bout_data['1_ml_odds']
                        bout_preds['exp_prob'] = self._bout_data['1_exp_prob']
                        bout_preds['odds_diff'] = self._bout_data['1_odds_diff']
                        bout_preds['exp_outcome'] = ((bout_preds['exp_prob']/100) * calcWinnings(bout_preds['sug_wager'], bout_preds['ml_odds'])) - ((1-(bout_preds['exp_prob']/100)) * bout_preds['sug_wager'])
                        if self._bout_data['1_odds_diff'] < self.diff_floor * 100 or self._bout_data['1_odds_diff'] > self.diff_ceiling * 100:
                            bout_preds['BET'] = 'No'
                        else:
                            bout_preds['BET'] = 'Yes'                        
            
            if self.score:
                self.results.append(bet_result)
            else:
                self._predictions[bout_id] = bout_preds
        else:
            if self.debug:
                print('skipping due to female fight')
                
    def _proc_fight(self, fight_id):
        self._predictions = {}
        fight_details = getBoutsFromFight(fight_id)
        for bout_detail in fight_details['bouts']:  
            self._proc_bout(bout_detail['boutId'])
                    
    def _eval(self):
        for fight_id in self.fight_id_list:
            self._proc_fight(fight_id)
                    
    def evaluate(self, full_score = False, fight_list = None, year = 2020, validate = False, save_results = True):
        self.score = True
        if fight_list is None:
            self.fight_id_list = getTrainingFights(year)
        else:
            self.fight_id_list = fight_list
            
        self._eval()
        if self.debug:
            print('FINAL: %s bouts bet on' % (len(self.results)))
        score = {'model_id': self.model_id, 'average': float(np.mean(self.results)), 'gross': float(np.sum(self.results))}
        
        if self.debug:
            print(score)
        if save_results:
            if validate:
                with open('src/training/bet/validation/scores/%s.json' % (self.model_id), 'w') as f:
                    json.dump(score, f)
                with open('src/training/bet/validation/models/%s.json' % (self.model_id), 'w') as p:
                    json.dump(self.params, p)                 
            else:
                with open('src/training/bet/scores/%s.json' % (self.model_id), 'w') as f:
                    json.dump(score, f)
                with open('src/training/bet/models/%s.json' % (self.model_id), 'w') as p:
                    json.dump(self.params, p)   
        if full_score:
            return score
        else:
            return score['gross']
        
    def predict(self, fight_id):
        self.score = False
        fight_details = getBoutsFromFight(fight_id)
        for bout_detail in fight_details['bouts']:  
            if self.debug:
                print()
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print()
            self._proc_bout(bout_detail['boutId'])      
        return self._predictions
        
#fights = getTrainingFights(2020)
#fights =   ['0c1773639c795466',
# '44aa652b181bcf68',
# 'b26d3e3746fb4024',
# '5df17b3620145578',
# '33b2f68ef95252e0',
# 'fc9a9559a05f2704',
# '0b5b6876c2a4723f',
# '53278852bcd91e11',
# '898337ef520fe4d3',
# '5f8e00c27b7e7410',
# 'dfb965c9824425db',
# '14b9e0f2679a2205',
# '4c12aa7ca246e7a4',
# '1e13936d708bcff7',
# '2eab7a6c8b0ed8cc',
# 'c32eab6c2119e989',
# 'dbd198f780286aca',
# '18f5669a92e99d92',
# 'ddbd0d6259ce57cc',
# 'dde70a112e053a6c',
# '3746e21bb508391a',
# '7a82635ffa9b59fe',
# 'bda04c573563cc2e',
# 'af1e5c64b8663aa0',
# 'e29cf523ebd155c5',
# '542db012217ecb83',
# '9c37681096c6f3a9',
# '831b937811804dad',
# 'e69c5ce12f4e762b',
# '805ad1801eb26abb',
# 'fd4578cac86d75ca',
# 'd4f364dd076bb0e2',
# 'c3c38c86f5ab9b5c']
#val_fights = [
#"c3c38c86f5ab9b5c",
#"d4f364dd076bb0e2",
#"fd4578cac86d75ca",
#"805ad1801eb26abb",
#"e69c5ce12f4e762b"
#        ]
#fights = [i for i in fights if i not in val_fights]
#
#import random
#random.shuffle(fights)

#test_fights = fights[:14]
#['e29cf523ebd155c5',
# 'fc9a9559a05f2704',
# '53278852bcd91e11',
# 'b26d3e3746fb4024',
# '33b2f68ef95252e0',
# 'c32eab6c2119e989',
# '5df17b3620145578',
# '542db012217ecb83',
# '14b9e0f2679a2205',
# 'bda04c573563cc2e',
# '898337ef520fe4d3',
# 'dfb965c9824425db',
# 'ddbd0d6259ce57cc',
# 'dde70a112e053a6c']
#train_fights = fights[14:]
#['4c12aa7ca246e7a4',
# '0c1773639c795466',
# '18f5669a92e99d92',
# '44aa652b181bcf68',
# '1e13936d708bcff7',
# '831b937811804dad',
# '5f8e00c27b7e7410',
# '3746e21bb508391a',
# 'af1e5c64b8663aa0',
# '0b5b6876c2a4723f',
# '2eab7a6c8b0ed8cc',
# 'dbd198f780286aca',
# '9c37681096c6f3a9',
# '7a82635ffa9b59fe']


def _opt_betting(trial):
    param = {
    'bet_female': trial.suggest_categorical('bet_female', [True, False]),
    'prev_fight_floor': trial.suggest_int('prev_fight_floor', 3, 7),
    'diff_floor': trial.suggest_loguniform('diff_floor', 1e-4, 1e-1),
#    'prev_fight_ceiling': trial.suggest_int('prev_fight_ceiling', 1, 15),
    'diff_ceiling': trial.suggest_loguniform('diff_ceiling', 1e-1, 0.3),
    'bet_intercept': trial.suggest_uniform('bet_intercept', -0.5, 0.5),
    'conf_diff_lin': trial.suggest_uniform('conf_diff_lin', -0.25, 0.5),
    'num_fight_lin': trial.suggest_uniform('num_fight_lin', -0.25, 0.5),
    'conf_diff_quad': trial.suggest_uniform('conf_diff_quad', -0.25, 0.5),
    'num_fight_quad': trial.suggest_uniform('num_fight_quad', -0.25, 0.5),
    'bet_ceiling': trial.suggest_int('bet_ceiling', 10, 200),
    }
    
    better = bet_eval(debug = False,
                      conf_diff_lin = param['conf_diff_lin'],
                      conf_diff_quad = param['conf_diff_quad'],
                      num_fight_lin = param['num_fight_lin'],
                      num_fight_quad = param['num_fight_quad'],
                      bet_intercept = param['bet_intercept'],
#                      prev_fight_ceiling = param['prev_fight_ceiling'],
                      prev_fight_floor = param['prev_fight_floor'],
                      diff_ceiling = param['diff_ceiling'],
                      diff_floor = param['diff_floor'],
                      bet_ceiling = param['bet_ceiling'],
                      bet_female = param['bet_female']
                      )
    
    return better.evaluate(fight_list = ['4c12aa7ca246e7a4',
                                         '0c1773639c795466',
                                         '18f5669a92e99d92',
                                         '44aa652b181bcf68',
                                         '1e13936d708bcff7',
                                         '831b937811804dad',
                                         '5f8e00c27b7e7410',
                                         '3746e21bb508391a',
                                         'af1e5c64b8663aa0',
                                         '0b5b6876c2a4723f',
                                         '2eab7a6c8b0ed8cc',
                                         'dbd198f780286aca',
                                         '9c37681096c6f3a9',
                                         '7a82635ffa9b59fe']
                            )    
    
def optimize_bet(clf = 'light', domain = 'strike', trials = 5000):
    study = optuna.create_study(direction='maximize')
    study.optimize(_opt_betting, n_trials=trials)      
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    
def _opt_betting_test():
    param = {
    'conf_diff_lin': 0,
    'conf_diff_quad': 0,
    'num_fight_lin': 0,
    'num_fight_quad': 0,
    'bet_intercept': 1,
    'prev_fight_ceiling': 15,
    'prev_fight_floor': 1,
    'diff_ceiling': .7,
    'diff_floor': .05
    }
    
    better = bet_eval(debug = True,
                      conf_diff_lin = param['conf_diff_lin'],
                      conf_diff_quad = param['conf_diff_quad'],
                      num_fight_lin = param['num_fight_lin'],
                      num_fight_quad = param['num_fight_quad'],
                      bet_intercept = param['bet_intercept'],
                      prev_fight_ceiling = param['prev_fight_ceiling'],
                      prev_fight_floor = param['prev_fight_floor'],
                      diff_ceiling = param['diff_ceiling'],
                      diff_floor = param['diff_floor']
                      )
    better.evaluate()  

def gen_score_report():
#    domain = 'strike'
    all_scores = {'average':[], 'gross':[]}
        
    onlyfiles = [f for f in listdir("src/training/bet/scores") if isfile(join("src/training/bet/scores", f))]
    for file in onlyfiles:
        with open('src/training/bet/scores/%s'% (file)) as f:
            data = json.load(f)
        all_scores['average'].append(data['average'])
        all_scores['gross'].append(data['gross'])
                
    ranked_scores = {}
    for file in onlyfiles:
        with open('src/training/bet/scores/%s'% (file)) as f:
            data = json.load(f)
        idx = data['model_id']
        score = {}
        score['average'] = data['average']
        score['gross'] = data['gross']
        score['average_rank'] = percentileofscore(all_scores['average'], data['average'], 'rank')
        score['gross_rank'] = percentileofscore(all_scores['gross'], data['gross'], 'rank')   
        score['tot_rank'] = score['average_rank'] + score['gross_rank']
        ranked_scores[idx] = score
    
    ranked_score_df = pd.DataFrame.from_dict(ranked_scores).T
    ranked_score_df['tot'] = ranked_score_df['average_rank'] + ranked_score_df['gross_rank']
    ranked_score_df.sort_values(by=['tot'], ascending = False, inplace = True)
    ranked_score_df.dropna(inplace = True)
    
    validated_scores = {}
    for idx in ranked_score_df.index:
        if ranked_score_df.loc[idx]['gross'] > 0:
            
            val_score = {}
            
            with open('src/training/bet/models/%s.json'% (idx)) as f:
                param = json.load(f)
                
            better = bet_eval(debug = True,
                      conf_diff_lin = param['conf_diff_lin'],
                      conf_diff_quad = param['conf_diff_quad'],
                      num_fight_lin = param['num_fight_lin'],
                      num_fight_quad = param['num_fight_quad'],
                      bet_intercept = param['bet_intercept'],
                      prev_fight_ceiling = param['prev_fight_ceiling'],
                      prev_fight_floor = param['prev_fight_floor'],
                      diff_ceiling = param['diff_ceiling'],
                      diff_floor = param['diff_floor'],
                      bet_ceiling = param['bet_ceiling'],
                      bet_female = param['bet_female']
                              )
            res = better.evaluate(full_score = True, 
                                  fight_list = ['e29cf523ebd155c5',
                                                 'fc9a9559a05f2704',
                                                 '53278852bcd91e11',
                                                 'b26d3e3746fb4024',
                                                 '33b2f68ef95252e0',
                                                 'c32eab6c2119e989',
                                                 '5df17b3620145578',
                                                 '542db012217ecb83',
                                                 '14b9e0f2679a2205',
                                                 'bda04c573563cc2e',
                                                 '898337ef520fe4d3',
                                                 'dfb965c9824425db',
                                                 'ddbd0d6259ce57cc',
                                                 'dde70a112e053a6c'],
                                    save_results = False,
                                    validate = True
                                    )  
            
            val_score['gross'] = res['gross']
            val_score['average'] = res['average']
            
            validated_scores[idx] = val_score
            
    validated_score_df = pd.DataFrame.from_dict(validated_scores).T
    
    pos_scores_df = validated_score_df.loc[validated_score_df['gross'] > 0]
    pos_scores_ids = [i for i in pos_scores_df.index]

    pos_models = {}
    for pos_score in pos_scores_ids:
        with open('src/training/bet/models/%s.json'% (pos_score)) as f:
            mod = json.load(f)
        with open('src/training/bet/scores/%s.json'% (pos_score)) as f:
            score = json.load(f)        
        score['average_rank'] = percentileofscore(all_scores['average'], score['average'], 'rank')
        score['gross_rank'] = percentileofscore(all_scores['gross'], score['gross'], 'rank') 
        mod['average_rank'] = score['average_rank']
        mod['gross_rank'] = score['gross_rank']     
        mod['total_rank'] = mod['average_rank'] + mod['gross_rank']
        mod['average'] = score['average']
        mod['gross'] = score['gross']  
        mod['val_gross'] = pos_scores_df.loc[pos_score]['gross']
        mod['val_gross_rank'] = percentileofscore(pos_scores_df['gross'], pos_scores_df.loc[pos_score]['gross'], 'rank') 
        mod['val_average'] = pos_scores_df.loc[pos_score]['average']
        mod['val_average_rank'] = percentileofscore(pos_scores_df['average'], pos_scores_df.loc[pos_score]['average'], 'rank') 
        mod['tot_val_rank'] = mod['val_average_rank'] + mod['val_gross_rank']
        pos_models[pos_score] = mod
        
    with open('src/betting_model_params_new.json', 'w') as b:
        json.dump(pos_models, b)
    pos_models_df = pd.DataFrame.from_dict(pos_models).T
    pos_models_df.to_csv("src/betting_model_params_new.csv")
    

def val_fights():
    results = {}
    for file in ['ebab23ef-11a3-4e5a-90fa-af107661dda0',
                '4d8443ff-65af-440d-973b-a009116efdd8',
                '11e7d3a5-eebd-485b-8fb9-6b0c8cc09d9c',
                'd9a5ea77-5715-433c-945b-e060c82468f7',
                '4c71e947-2185-439f-b044-0b9c5fad1ca5',
                '90b7e07b-dd7b-42ec-b73b-9497077fb0ae',
                '10d7f9ec-d79a-479a-a397-410c6276b11e',
                '7dc50b87-84af-462f-8939-69d88417be7b',
                '170a4c24-1bee-48f6-91cf-dbbe82711c67',
                'f733b0d5-d516-478d-82da-eacec7dbc8ab',
                '9dcee953-84ef-415f-80a8-ed8e736cda7d',
                'fa2c06ef-0e1d-4b5b-9260-e3da50cfafa0',
                'f692aef2-fd4f-4074-87b2-010b1cff767a',
                '12de8fc5-d3df-4e43-8824-53b3ea4bd908',
                '981a1214-15d1-4214-b180-2711f97c66d3',
                '87424177-05cd-4427-b60e-1a5461bebf22',
                'abde0e96-4ca2-401e-9095-046a328fffbc',
                '5f353ef0-289e-449b-a635-e29c6e026e3d',
                'a0f2a2ae-b5ac-4420-a487-eb03805a6bd2',
                'a77925c9-6ee6-41bb-b357-b3e5992c11bb',
                '67e2b251-4878-4c8d-bd19-e9f6780b657c',
                '82087aa8-8db1-4b05-b3b5-7359ff821e68',
                '0cb9f260-32af-403c-9f57-1b8adaa3ff3a'
                ]:
        with open('src/training/bet/models/%s.json' % (file), 'r') as r:
            param = json.load(r)    
        bettor = bet_eval(debug = True,
                          conf_diff_lin = param['conf_diff_lin'],
                          conf_diff_quad = param['conf_diff_quad'],
                          num_fight_lin = param['num_fight_lin'],
                          num_fight_quad = param['num_fight_quad'],
                          bet_intercept = param['bet_intercept'],
                          prev_fight_ceiling = param['prev_fight_ceiling'],
                          prev_fight_floor = param['prev_fight_floor'],
                          diff_ceiling = param['diff_ceiling'],
                          diff_floor = param['diff_floor'],
                          bet_ceiling = param['bet_ceiling'],
                          bet_female = param['bet_female']
                          )      
        res = bettor.evaluate(full_score = True, 
                              fight_list = [#'c3c38c86f5ab9b5c',
                                             'd4f364dd076bb0e2',
                                             'fd4578cac86d75ca',
                                             '805ad1801eb26abb',
                                             'e69c5ce12f4e762b'],
                                save_results = False,
                                validate = True
                                )    
        results[file] = res
    with open('src/new_validation_results.json', 'w') as b:
        json.dump(results, b)    


def validate_new_fights():
    with open('src/predictors/bet/bettor_config_new.json', 'r') as r:
        param = json.load(r)    
    bettor = bet_eval(debug = True,
                      conf_diff_lin = param['conf_diff_lin'],
                      conf_diff_quad = param['conf_diff_quad'],
                      num_fight_lin = param['num_fight_lin'],
                      num_fight_quad = param['num_fight_quad'],
                      bet_intercept = param['bet_intercept'],
                      prev_fight_ceiling = param['prev_fight_ceiling'],
                      prev_fight_floor = param['prev_fight_floor'],
                      diff_ceiling = param['diff_ceiling'],
                      diff_floor = param['diff_floor'],
                      bet_ceiling = param['bet_ceiling'],
                      bet_female = param['bet_female']
                      )      
    res = bettor.evaluate(full_score = True, 
                          fight_list = [
#                                          "dde70a112e053a6c",
#                                        "ddbd0d6259ce57cc",
#                                        "18f5669a92e99d92",
                                        "dbd198f780286aca"
                                        ],
                            save_results = False,
                            validate = True
                            )
#    FINAL: 18 bouts bet on
#    {'model_id': 'b9808a44-f9d9-4df0-bc1c-e71ebdf1c96c', 'average': -17.988492850591758, 'gross': -323.79287131065166}

#    FINAL: 19 bouts bet on
#    {'model_id': '09c03503-f97d-4828-bd91-e02178ff041c', 'average': -25.564903527386413, 'gross': -485.73316702034185}
#    res = bettor.evaluate(full_score = True, 
#                          fight_list = ["dfb965c9824425db",
#                                        "5f8e00c27b7e7410",
#                                        "898337ef520fe4d3",
#                                        "53278852bcd91e11",
#                                        "0b5b6876c2a4723f"],
#                            save_results = False,
#                            validate = True
#                            )  
#    FINAL: 18 bouts bet on
#    {'model_id': 'b940c641-91a8-4026-ab86-0d58ff0c9bf5', 'average': -4.852955869906466, 'gross': -87.35320565831638}

#   FINAL: 19 bouts bet on
#   {'model_id': 'd24d5722-460b-4e2f-ba7b-292b51550e1b', 'average': 24.518646893254747, 'gross': 465.8542909718402}
#    
#   FINAL: 20 bouts bet on
#   {'model_id': 'b1d20286-05bc-494c-9098-b4f22c7d838f', 'average': 12.336904823309654, 'gross': 246.73809646619307}

#    res = bettor.evaluate(full_score = True, 
#                          fight_list = ['fc9a9559a05f2704',
#                                        '33b2f68ef95252e0',
#                                        '5df17b3620145578',
#                                        'b26d3e3746fb4024',
#                                        '44aa652b181bcf68',
#                                        '0c1773639c795466'],
#                            save_results = False,
#                            validate = True
#                            )  
#    
#    FINAL: 14 bouts bet on
#    {'model_id': 'f9ad41a2-bf01-4889-a49b-99b895ce0080', 'average': 31.866347217970617, 'gross': 446.1288610515886}
#    FINAL: 14 bouts bet on
#    {'model_id': '88d00cf2-331b-4c6d-971f-ed3a013d002c', 'average': 31.298301890398868, 'gross': 438.17622646558414}
    val_score = {}
    val_score['gross'] = res['gross']
    val_score['average'] = res['average']
    
#    bettor._predictions
    
#    fight_id = 'dbd198f780286aca'
    
def add_best_model():
    best_model_id = 'fa2c06ef-0e1d-4b5b-9260-e3da50cfafa0'
    with open('src/training/bet/models/%s.json'% (best_model_id)) as f:
        mod = json.load(f)    
    with open('src/predictors/bet/bettor_config.json', 'w') as w:
        json.dump(mod, w)
    
#    fight_id = '3746e21bb508391a'
def predict_bet_winners(fight_id):
    with open('src/predictors/bet/bettor_config_new.json', 'r') as r:
        param = json.load(r)    
    bettor = bet_eval(debug = True,
                      conf_diff_lin = param['conf_diff_lin'],
                      conf_diff_quad = param['conf_diff_quad'],
                      num_fight_lin = param['num_fight_lin'],
                      num_fight_quad = param['num_fight_quad'],
                      bet_intercept = param['bet_intercept'],
                      prev_fight_ceiling = param['prev_fight_ceiling'],
                      prev_fight_floor = param['prev_fight_floor'],
                      diff_ceiling = param['diff_ceiling'],
                      diff_floor = param['diff_floor'],
                      bet_ceiling = param['bet_ceiling'],
                      bet_female = param['bet_female']
                      )  
    preds = bettor.predict(fight_id)
    print(preds)
    preds_df = pd.DataFrame.from_dict(preds).T
    
    