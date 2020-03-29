#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 01:12:08 2020

@author: eric.hensleyibm.com
"""

import matplotlib.pyplot as plt
import requests 
import pandas as pd
import math
import numpy as np
import uuid
import random
np.random.seed(1108)
from parse_fights import getBoutsFromFight
HOST = '207.237.93.29'
PORT = '4646'
TRAINING_YEAR = '2019'
TEST_YEAR = '2020'
WAGER = 20
BANK = 500

BANK_LOG = []
GET_FIGHTS_BY_YEAR = "http://%s:%s/ufc/rest/fight/year/%s"

def getTrainingFights(year):
    r = requests.get(url = GET_FIGHTS_BY_YEAR % (HOST, PORT, year))
    response = r.json()         
    return response['response']

def predictWinner(bout):
    return bout['fighterBoutXRefs'][np.random.choice([0,1])]['fighter']['fighterId']
    
def evalActualWinner(bout):
    if (bout['fighterBoutXRefs'][0]['outcome'] == 'W'):
        return bout['fighterBoutXRefs'][0]['fighter']['fighterId']
    else:
        return bout['fighterBoutXRefs'][1]['fighter']['fighterId']

def convImpPercToAmericanOdds(impPerc):
    if impPerc == 50:
        return 100
    elif impPerc > 50:
        return (-1*(impPerc/(100-impPerc)))*100
    else:
        return ((100 - impPerc)/impPerc)*100
          
def calcWinnings(wager, impPerc):
#    wager, odds = WAGER, bout['fighterBoutXRefs'][1]['mlOdds']
    odds = convImpPercToAmericanOdds(impPerc)
    if (odds > 0):
        return wager * (odds/100)
    else:
        return -1*(wager*100)/odds 
    

convImpPercToAmericanOdds(25)  
    
fight_id_list = getTrainingFights(TEST_YEAR)
n = 0
for fight_id in fight_id_list:
    fight_details = getBoutsFromFight(fight_id)
    for bout in fight_details['bouts']:
        predicted_winner = predictWinner(bout)
        actual_winner = evalActualWinner(bout)
#        if n == 28:
#            asadsfa
        if (predicted_winner == actual_winner):
            if (bout['fighterBoutXRefs'][0]['fighter']['fighterId'] == predicted_winner):
                BANK += calcWinnings(WAGER, bout['fighterBoutXRefs'][0]['mlOdds'])
            else:
                BANK += calcWinnings(WAGER, bout['fighterBoutXRefs'][1]['mlOdds'])                
        else:
            BANK -= WAGER
        BANK_LOG.append(BANK)
        n += 1