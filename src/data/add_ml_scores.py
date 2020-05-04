#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 23:56:02 2020

@author: ehens86
"""
from sklearn.linear_model import LogisticRegression
import sys, os
if __name__ == "__main__":
    sys.path.append("..")
import pandas as pd
import numpy as np
from data.generate import pull_year_raw_training_data
from spring.api_wrappers import saveMlScore

KO_FEATURES = ['offBodySigStrikeAccuracy', 'offBodySigStrikeAttempted',
       'offBodySigStrikeSuccessful', 'offClinchSigStrikeAccuracy',
       'offClinchSigStrikeAttempted', 'offClinchSigStrikeSuccessful',
       'offDistanceSigStrikeAccuracy', 'offDistanceSigStrikeAttempted',
       'offDistanceSigStrikeSuccessful', 'offGroundSigStrikeAccuracy',
       'offGroundSigStrikeAttempted', 'offGroundSigStrikeSuccessful',
       'offHeadSigStrikeAccuracy', 'offHeadSigStrikeAttempted',
       'offHeadSigStrikeSuccessful', 'offKnockdowns',
       'offLegSigStrikeAccuracy', 'offLegSigStrikeAttempted',
       'offLegSigStrikeSuccessful', 'offTotStrikeAccuracy',
       'offTotStrikeAttempted', 'offTotStrikeSuccessful', 'weight',
       'offBodySigStrikeSuccessfulShare', 'offBodySigStrikeAttemptedShare',
       'offClinchSigStrikeSuccessfulShare', 'offClinchSigStrikeAttemptedShare',
       'offDistanceSigStrikeSuccessfulShare',
       'offDistanceSigStrikeAttemptedShare',
       'offGroundSigStrikeSuccessfulShare', 'offGroundSigStrikeAttemptedShare',
       'offHeadSigStrikeSuccessfulShare', 'offHeadSigStrikeAttemptedShare',
       'offLegSigStrikeSuccessfulShare', 'offLegSigStrikeAttemptedShare',
       'offTotStrikeSuccessfulShare', 'offTotStrikeAttemptedShare',
       'offKnockdownsShare']
#SUB_FEATURES = ['offGroundSigStrikeAccuracy', 'offGroundSigStrikeAttempted',
#       'offGroundSigStrikeSuccessful', 'offKnockdowns', 'offPassSuccessful',
#       'offReversalSuccessful', 'offReversalSuccessful',
#       'offReversalSuccessful', 'offReversalSuccessful', 'offTakedownAccuracy',
#       'offTakedownAttempted', 'offTakedownSuccessful', 'offTotStrikeAccuracy',
#       'offTotStrikeAttempted', 'offTotStrikeSuccessful', 'weight',
#       'offGroundSigStrikeSuccessfulShare', 'offGroundSigStrikeAttemptedShare',
#       'offTotStrikeSuccessfulShare', 'offTotStrikeAttemptedShare',
#       'offTakedownSuccessfulShare', 'offTakedownAttemptedShare',
#       'offKnockdownsShare', 'offPassSuccessfulShare',
#       'offReversalSuccessfulShare', 'offSubmissionAttemptedShare',
#       'offSubmissionAttempted']

SUB_FEATURES = ['offGroundSigStrikeAccuracy', 'offGroundSigStrikeAttempted',
       'offGroundSigStrikeSuccessful', 'offKnockdowns', 'offPassSuccessful',
       'offReversalSuccessful', 'offReversalSuccessful',
       'offReversalSuccessful', 'offReversalSuccessful', 'offTakedownAccuracy',
       'offTakedownAttempted', 'offTakedownSuccessful', 'offTotStrikeAccuracy',
       'offTotStrikeAttempted', 'offTotStrikeSuccessful', 'weight',
       'offGroundSigStrikeSuccessfulShare', 'offGroundSigStrikeAttemptedShare',
       'offTotStrikeSuccessfulShare', 'offTotStrikeAttemptedShare',
       'offTakedownSuccessfulShare', 'offTakedownAttemptedShare',
       'offKnockdownsShare', 'offPassSuccessfulShare',
       'offReversalSuccessfulShare', 'offSubmissionAttemptedShare',
       'offSubmissionAttempted']

KO_MODEL = LogisticRegression(class_weight='balanced', solver = 'newton-cg', C = 100, max_iter = 50000)
#SUB_MODEL = LogisticRegression(class_weight='balanced', solver = 'newton-cg', C = 1.00000000e+04)
SUB_MODEL = LogisticRegression(class_weight='balanced', solver = 'newton-cg', C = 100, max_iter = 10000)


DATA = pd.DataFrame()
for year in range(2005, 2020):
    year_data = pull_year_raw_training_data(year, score = 'finish', add_gender = True, add_class = True)
    DATA = DATA.append(year_data)
    
DATA = DATA[DATA['gender'] == 'M']
KO_DATA = DATA[DATA['round'] == 1]
SUB_DATA = DATA[(DATA['round'] == 2) | (DATA['round'] ==3)]

import warnings
warnings.filterwarnings("error")

def generate_ko_score(idx, ko_test):
    # ko_test = test
    train_x = KO_DATA.loc[KO_DATA.index != idx][KO_FEATURES]
    train_y = KO_DATA.loc[KO_DATA.index != idx]['offTkoKo']
    KO_MODEL.fit(train_x, train_y)
    ko_score = KO_MODEL.predict_proba(np.array(ko_test[KO_FEATURES]).reshape(1,-1))[0][1]
    return ko_score

def generate_sub_score(idx, sub_test):
    # sub_test = test
    train_x = SUB_DATA.loc[SUB_DATA.index != idx][SUB_FEATURES]
    train_y = SUB_DATA.loc[SUB_DATA.index != idx]['offSubmissionSuccessful']
#    mod = LogisticRegression(class_weight='balanced', solver = 'newton-cg', C = 1.00000000e+04, max_iter = 5000)
#    mod.fit(train_x, train_y)
    SUB_MODEL.fit(train_x, train_y)
#    SUB_MODEL.fit(SUB_DATA[SUB_FEATURES], SUB_DATA['offSubmissionSuccessful'])
    sub_score = SUB_MODEL.predict_proba(np.array(sub_test[SUB_FEATURES]).reshape(1,-1))[0][1]
    return sub_score
    
for idx in DATA.index:
    test = DATA.loc[idx]
    ko_score = generate_ko_score(idx, test)
    sub_score = generate_sub_score(idx, test)
    payload = {"oid": idx, "koScore": ko_score, "subScore": sub_score}
    resp = saveMlScore(payload)
    
    
    
