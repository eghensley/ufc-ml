#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:15:18 2020

@author: eric.hensleyibm.com
"""
    
import os
import numpy as np
import pandas as pd
from spring.api_wrappers import getTrainingFights, getBoutData, getBoutsFromFight

def generate_year_data(year):
    fight_id_list = getTrainingFights(year)
    data_dict = {}
    for fight_id in fight_id_list:
        fight_details = getBoutsFromFight(fight_id)
        fight_date = fight_details['fightDate']
        for bout in fight_details['bouts']:
            bout_id = bout['boutId']
            bout_data = getBoutData(bout_id)
            for row in bout_data:
                row['date'] = fight_date
                data_dict[row['oid']] = row
                
    data = pd.DataFrame.from_dict(data_dict).T
    data.set_index('oid', inplace = True)
    data.to_csv("data/raw_data/output_round_data_%s.csv" % (year))

def single_row_per_round(data):
    data['dupCol'] = data['boutOid'] + data['round'].astype(str)
    data.drop_duplicates(subset = 'dupCol', inplace = True)
    return data

def apply_gender(weightClass):
    if (weightClass in ['WW', 'LW', 'FFW', 'MW', 'HW', 'FW', 'BW', 'CW', 'LHW']):
        return "M"
    else:
        return "F"
    
def add_gender_col(data):
    data['gender']=data['weightClass'].apply(lambda x: apply_gender(x))
    return data

def add_class_col(data):
    class_dict = {'WSW': "SW", 'FW':"FW", 'WFW': "FW", 'BW': "BW", 'WBW': "BW", 'FFW': "FFW", 'WFFW': "FFW", 'LW': "LW", 'WW': "WW", 'MW': "MW", 'LHW': "LHW", 'HW': "HW", 'CW': "CW"}
    data['class'] = data['weightClass'].apply(lambda x: class_dict[x])
    return data

def conv_stats_to_rate(data):
    for strike_col in ['BodySigStrike', 'ClinchSigStrike', 'DistanceSigStrike', 'GroundSigStrike', 'HeadSigStrike', 'LegSigStrike', 'TotStrike', 'Takedown']:
        data['off'+strike_col+'Successful'] = data['off'+strike_col+'Successful'] / data['seconds']
        data['def'+strike_col+'Successful'] = data['def'+strike_col+'Successful'] / data['seconds']
        data['off'+strike_col+'Attempted'] = data['off'+strike_col+'Attempted'] / data['seconds']
        data['def'+strike_col+'Attempted'] = data['def'+strike_col+'Attempted'] / data['seconds']        
    for tech_col in ['Knockdowns', 'PassSuccessful', 'ReversalSuccessful', 'SubmissionAttempted']:
        data['off'+tech_col] = data['off'+tech_col] / data['seconds']
        data['def'+tech_col] = data['def'+tech_col] / data['seconds']             
    return data
    
def add_share_cols(data, fill_zero = False):
    for strike_col in ['BodySigStrike', 'ClinchSigStrike', 'DistanceSigStrike', 'GroundSigStrike', 'HeadSigStrike', 'LegSigStrike', 'TotStrike', 'Takedown']:
#        if fill_zero and data['off'+strike_col+'Successful'] + data['def'+strike_col+'Successful'] == 0:
#            data['off'+strike_col+'SuccessfulShare'] = 0
#        else:
        data['off'+strike_col+'SuccessfulShare'] = data['off'+strike_col+'Successful'] / (data['off'+strike_col+'Successful'] + data['def'+strike_col+'Successful'])
        data['off'+strike_col+'AttemptedShare'] = data['off'+strike_col+'Attempted'] / (data['off'+strike_col+'Attempted'] + data['def'+strike_col+'Attempted'])
        data['off'+strike_col+'SuccessfulShare'].replace([np.inf, -np.inf], 0, inplace = True)
        data['off'+strike_col+'AttemptedShare'].replace([np.inf, -np.inf], 0, inplace = True)
        data['off'+strike_col+'SuccessfulShare'].fillna(0, inplace = True)
        data['off'+strike_col+'AttemptedShare'].fillna(0, inplace = True)
    for tech_col in ['Knockdowns', 'PassSuccessful', 'ReversalSuccessful', 'SubmissionAttempted']:
        data['off'+tech_col+'Share'] = data['off'+tech_col] / (data['off'+tech_col] + data['def'+tech_col])
        data['off'+tech_col+'Share'].replace([np.inf, -np.inf], 0, inplace = True)
        data['off'+tech_col+'Share'].fillna(0, inplace = True)
    return data

def add_share_cols_new(data):
    for strike_col in ['BodySigStrike', 'ClinchSigStrike', 'DistanceSigStrike', 'GroundSigStrike', 'HeadSigStrike', 'LegSigStrike', 'TotStrike', 'Takedown']:
        data['off'+strike_col+'SuccessfulShare'] = (data['off'+strike_col+'Successful'] / (data['off'+strike_col+'Successful'] + data['def'+strike_col+'Successful']).apply(lambda x: np.nan if x == 0 else x)).replace(np.nan, 0)
        data['off'+strike_col+'AttemptedShare'] = (data['off'+strike_col+'Attempted'] / (data['off'+strike_col+'Attempted'] + data['def'+strike_col+'Attempted']).apply(lambda x: np.nan if x == 0 else x)).replace(np.nan, 0)
        data['off'+strike_col+'SuccessfulShare'].replace([np.inf, -np.inf], 0, inplace = True)
        data['off'+strike_col+'AttemptedShare'].replace([np.inf, -np.inf], 0, inplace = True)
        data['off'+strike_col+'SuccessfulShare'].fillna(0, inplace = True)
        data['off'+strike_col+'AttemptedShare'].fillna(0, inplace = True)
    for tech_col in ['Knockdowns', 'PassSuccessful', 'ReversalSuccessful', 'SubmissionAttempted']:
        data['off'+tech_col+'Share'] = (data['off'+tech_col] / (data['off'+tech_col] + data['def'+tech_col]).apply(lambda x: np.nan if x == 0 else x)).replace(np.nan, 0)
        data['off'+tech_col+'Share'].replace([np.inf, -np.inf], 0, inplace = True)
        data['off'+tech_col+'Share'].fillna(0, inplace = True)
    return data
        
def pull_year_raw_training_data(year, norm_to_rate = True, add_share_feats = True, score = 'ko', drop_finishes = False, single_row = False, add_gender = False, add_class = False):
    if not os.path.exists("data/raw_data/output_round_data_%s.csv" % (year)):
        generate_year_data(year)
    data = pd.read_csv("data/raw_data/output_round_data_%s.csv" % (year))
    data.set_index("oid", inplace = True)
    if (drop_finishes):
        data = data[data['finish'] != 1]
    if (single_row):
        data = single_row_per_round(data)
    if (add_gender):
        data = add_gender_col(data)
    if (add_class):
        data = add_class_col(data)
    if (norm_to_rate):
        data = conv_stats_to_rate(data)
    if (add_share_feats):
        data = add_share_cols(data)
    return data

def pull_ml_training_corpus():
    data = pd.DataFrame()
    for year in range(2005, 2020):
        year_data = pull_year_raw_training_data(year, score = 'finish', add_gender = True, add_class = True)
        data = data.append(year_data)
    data = data[data['gender'] == 'M']
    return data

#fight_id, bout_id = '4834ff149dc9542a', 'eed6c9aff2234b7a'
def pull_bout_data(bout_id):
    raw_data = getBoutData(bout_id)
    data_dict = {}
    for data_row in raw_data:
        data_dict[data_row['oid']] = data_row
    data = pd.DataFrame.from_dict(data_dict).T
    data = add_gender_col(data)
    data = add_class_col(data)
    data = conv_stats_to_rate(data)
    data = add_share_cols_new(data)
    return data
