#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:15:18 2020

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
GET_BOUT_DATA_BY_OID = "http://%s:%s/ufc/rest/bout/%s/data"

def getTrainingFights(year):
    r = requests.get(url = GET_FIGHTS_BY_YEAR % (HOST, PORT, year))
    response = r.json()         
    return response['response']

def getBoutData(boutOid):
    r = requests.get(url = GET_BOUT_DATA_BY_OID % (HOST, PORT, boutOid))
    response = r.json()         
    return response['response']

fight_id_list = getTrainingFights(TRAINING_YEAR)
data_dict = {}
for fight_id in fight_id_list:
    fight_details = getBoutsFromFight(fight_id)
    fight_date = fight_details['fightDate']
    for bout in fight_details['bouts']:
        bout_id = bout['boutId']
        bout_data = getBoutData(bout_id)
        for row in bout_data:
            data_dict[row['oid']] = row
            
data = pd.DataFrame.from_dict(data_dict).T
list(data)
export_data_cont = data[[
 'defBodySigStrikeAccuracy',
 'defBodySigStrikeAttemped',
 'defBodySigStrikeSuccessful',
 'defClinchSigStrikeAccuracy',
 'defClinchSigStrikeAttemped',
 'defClinchSigStrikeSuccessful',
 'defDistanceSigStrikeAccuracy',
 'defDistanceSigStrikeAttemped',
 'defDistanceSigStrikeSuccessful',
 'defGroundSigStrikeAccuracy',
 'defGroundSigStrikeAttemped',
 'defGroundSigStrikeSuccessful',
 'defHeadSigStrikeAccuracy',
 'defHeadSigStrikeAttemped',
 'defHeadSigStrikeSuccessful',
 'defKnockdowns',
 'defLegSigStrikeAccuracy',
 'defLegSigStrikeAttemped',
 'defLegSigStrikeSuccessful',
 'defPassSuccessful',
 'defReversalSuccessful',
 'defSubmissionAttempted',
 'defTakedownAttempted',
 'defTakedownSuccessful',
 'defTotStrikeAccuracy',
 'defTotStrikeAttempted',
 'defTotStrikeSuccessful',
 'offBodySigStrikeAccuracy',
 'offBodySigStrikeAttemped',
 'offBodySigStrikeSuccessful',
 'offClinchSigStrikeAccuracy',
 'offClinchSigStrikeAttemped',
 'offClinchSigStrikeSuccessful',
 'offDistanceSigStrikeAccuracy',
 'offDistanceSigStrikeAttemped',
 'offDistanceSigStrikeSuccessful',
 'offGroundSigStrikeAccuracy',
 'offGroundSigStrikeAttemped',
 'offGroundSigStrikeSuccessful',
 'offHeadSigStrikeAccuracy',
 'offHeadSigStrikeAttemped',
 'offHeadSigStrikeSuccessful',
 'offKnockdowns',
 'offLegSigStrikeAccuracy',
 'offLegSigStrikeAttemped',
 'offLegSigStrikeSuccessful',
 'offPassSuccessful',
 'offReversalSuccessful',
 'offSubmissionAttempted',
 'offTakedownAttempted',
 'offTakedownSuccessful',
 'offTotStrikeAccuracy',
 'offTotStrikeAttempted',
 'offTotStrikeSuccessful',
 'score']]

export_data_cont.to_csv("output_round_score_data_cont.csv")
data['binary_score'] = data['score'].apply(lambda x: 1 if x > 0 else 0)
export_data_bin = data[[
 'defBodySigStrikeAccuracy',
 'defBodySigStrikeAttemped',
 'defBodySigStrikeSuccessful',
 'defClinchSigStrikeAccuracy',
 'defClinchSigStrikeAttemped',
 'defClinchSigStrikeSuccessful',
 'defDistanceSigStrikeAccuracy',
 'defDistanceSigStrikeAttemped',
 'defDistanceSigStrikeSuccessful',
 'defGroundSigStrikeAccuracy',
 'defGroundSigStrikeAttemped',
 'defGroundSigStrikeSuccessful',
 'defHeadSigStrikeAccuracy',
 'defHeadSigStrikeAttemped',
 'defHeadSigStrikeSuccessful',
 'defKnockdowns',
 'defLegSigStrikeAccuracy',
 'defLegSigStrikeAttemped',
 'defLegSigStrikeSuccessful',
 'defPassSuccessful',
 'defReversalSuccessful',
 'defSubmissionAttempted',
 'defTakedownAttempted',
 'defTakedownSuccessful',
 'defTotStrikeAccuracy',
 'defTotStrikeAttempted',
 'defTotStrikeSuccessful',
 'offBodySigStrikeAccuracy',
 'offBodySigStrikeAttemped',
 'offBodySigStrikeSuccessful',
 'offClinchSigStrikeAccuracy',
 'offClinchSigStrikeAttemped',
 'offClinchSigStrikeSuccessful',
 'offDistanceSigStrikeAccuracy',
 'offDistanceSigStrikeAttemped',
 'offDistanceSigStrikeSuccessful',
 'offGroundSigStrikeAccuracy',
 'offGroundSigStrikeAttemped',
 'offGroundSigStrikeSuccessful',
 'offHeadSigStrikeAccuracy',
 'offHeadSigStrikeAttemped',
 'offHeadSigStrikeSuccessful',
 'offKnockdowns',
 'offLegSigStrikeAccuracy',
 'offLegSigStrikeAttemped',
 'offLegSigStrikeSuccessful',
 'offPassSuccessful',
 'offReversalSuccessful',
 'offSubmissionAttempted',
 'offTakedownAttempted',
 'offTakedownSuccessful',
 'offTotStrikeAccuracy',
 'offTotStrikeAttempted',
 'offTotStrikeSuccessful',
 'binary_score']]
export_data_bin.to_csv("output_round_score_data_binary.csv")
