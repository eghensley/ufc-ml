#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:19:56 2020

@author: ehens86
"""

#import yaml
#from ._config_ import CONFIG_PATH
#CONFIG = yaml.load(open(CONFIG_PATH))

import os

CONFIG = {'spring': {'HOST': os.getenv('ufc.flask.spring.host'),
  'PW': os.getenv('ufc.flask.spring.pw'),
  'rest': {'GET_FIGHTS_BY_YEAR': '%s/ufc/rest/fight/year/%s',
   'GET_BOUT_DATA_BY_OID': '%s/ufc/rest/bout/%s/data',
   'INIT_FIGHT_URL': '%s/ufc/parse/fights',
   'INIT_FUTURE_FIGHT_URL': '%s/ufc/parse/fights/next',
   'PARSE_YEAR_JUDGE_SCORE_URL': '%s/ufc/parse/fights/judgeScores/%s',
   'ADD_BOUTS_TO_FIGHT': '%s/ufc/parse/fights/%s/bouts',
   'ADD_BOUTS_TO_FUTURE_FIGHT': '%s/ufc/parse/fights/%s/bouts/future',
   'GET_BOUTS_FROM_FIGHT': '%s/ufc/rest/fight/%s/details',
   'ADD_BOUTS_DETAILS': '%s/ufc/parse/fights/%s/bouts/%s',
   'ADD_FIGHT_ODDS_URL': '%s/ufc/parse/odds/fight/%s/fightOdds/%s',
   'ADD_BOUT_SCORE_URL': '%s/ufc/parse/fights/judgeScores/fight/%s',
   'SCRAPE_BOUT_SCORE': '%s/ufc/parse/fights/judgeScores/bout/%s',
   'GET_BOUT': '%s/ufc/rest/bout/%s/full',
   'ADD_FIGHT_EXP_OUTCOME': '%s/ufc/parse/odds/fight/%s/expOutcomes',
   'ADD_FIGHT_ODDS': '%s/ufc/parse/odds/fight/%s/odds',
   'GET_ALL_FIGHT_IDS': '%s/ufc/rest/fight/all',
   'ADD_ROUND_SCORE': '%s/ufc/scores/bout/%s/round/add',
   'ADD_ML_SCORE': '%s/ufc/scores/bout/round/ml/add',
   'GET_ALL_BOUTS': '%s/ufc/rest/bout/all',
   'GET_NEW_BOUTS': '%s/ufc/rest/bout/new',
   'GET_YEAR_BOUTS': '%s/ufc/rest/bout/year/2019',
   'GET_LAST_ELO': '%s/ufc/scores/elo/last/fighter/%s/fight/%s',
   'GET_LAST_ELO_COUNT': '%s/ufc/scores/elo/last/fighter/%s/fight/%s/count',
   'UPDATE_ELO': '%s/ufc/scores/elo/update',
   'CLEAR_ELO': '%s/ufc/scores/elo/clear',
   'ADD_BOUT_ML_SCORE': '%s/ufc/scores/bout/ml/add',
   'ADD_FUT_BOUT_SUMMARY': '%s/ufc/rest/bout/future/summary/add',
   'ADD_MY_BOOKIE_ODDS': '%s/ufc/scores/odds/myBookie/add',
   'UPDATE_RANKING': '%s/ufc/ranks/update',
   'GET_WC_RANKINGS': '%s/ufc/ranks/weightClass/%s/basic',
   'GET_ELO_COUNT': '%s/ufc/scores/elo/fighter/%s/count'}},
 'flask': {'PORT': os.getenv('ufc.flask.flask.port')}}

if (CONFIG['flask']['PORT'] is None):
    CONFIG['flask']['PORT'] = 8080
    
print(CONFIG)