#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 01:12:08 2020

@author: eric.hensleyibm.com
"""

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    
import numpy as np
np.random.seed(1108)
from utils.general import calcWinnings
from dummy_predictor import predictWinner
from spring.api_wrappers import getTrainingFights, getBoutsFromFight

def evalActualWinner(bout):
    if (bout['fighterBoutXRefs'][0]['outcome'] == 'W'):
        return bout['fighterBoutXRefs'][0]['fighter']['fighterId']
    else:
        return bout['fighterBoutXRefs'][1]['fighter']['fighterId']
    
class bet_eval:
    def __init__(self, start_bank, bet_wager, year):
        self.bank = start_bank
        self.wager = bet_wager
        self.fight_id_list = getTrainingFights(year)
        self.bank_log = []
        
    def evaluate(self):
        for fight_id in self.fight_id_list:
            fight_details = getBoutsFromFight(fight_id)
            for bout in fight_details['bouts']:
                predicted_winner = predictWinner(bout)
                actual_winner = evalActualWinner(bout)
                if (predicted_winner == actual_winner):
                    if (bout['fighterBoutXRefs'][0]['fighter']['fighterId'] == predicted_winner):
                        self.bank += calcWinnings(self.wager, bout['fighterBoutXRefs'][0]['mlOdds'])
                    else:
                        self.bank += calcWinnings(self.wager, bout['fighterBoutXRefs'][1]['mlOdds'])                
                else:
                    self.bank -= self.wager
                self.bank_log.append(self.bank)       
        
if __name__ == "__main__":
    WAGER = 20
    BANK = 500
    YEAR = 2020
    case = bet_eval(BANK, WAGER, YEAR)
    case.evaluate()
    print("$%s" % case.bank)
    
          

    
    

