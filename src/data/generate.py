#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:15:18 2020

@author: eric.hensleyibm.com
"""

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    
import os
import numpy as np
import pandas as pd
from spring.api_wrappers import getTrainingFights, getBoutData, getBoutsFromFight

def generate_raw_round_data():
    data_dict = {}
    for year in range(2012, 2020):
        fight_id_list = getTrainingFights(year)
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
    data = data[[i for i in list(data) if i != 'score']]
    data.to_csv("raw_round_data.csv")

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
    data.to_csv("output_round_data_%s.csv" % (year))
    
def generate_score_training_data():
    fight_id_list = getTrainingFights(2019)
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
    data.to_csv("output_round_score_data.csv")

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
    
def add_share_cols(data):
    for strike_col in ['BodySigStrike', 'ClinchSigStrike', 'DistanceSigStrike', 'GroundSigStrike', 'HeadSigStrike', 'LegSigStrike', 'TotStrike', 'Takedown']:
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
    
def pull_score_training_data(norm_to_rate = True, add_share_feats = True, binary = True, drop_finishes = True, single_row = False, add_gender = False, add_class = False):
    if not os.path.exists("output_round_score_data.csv"):
        generate_score_training_data()
    data = pd.read_csv("../data/output_round_score_data.csv")
    data.set_index("oid", inplace = True)
    if (drop_finishes):
        data = data[data['finish'] != 1]
    if (binary):
        data['score'] = data['score'].apply(lambda x: 1 if x > 0 else -1)
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
    data.to_csv("watson_score_classification_output.csv")
        
def pull_year_raw_training_data(year, norm_to_rate = True, add_share_feats = True, score = 'ko', drop_finishes = False, single_row = False, add_gender = False, add_class = False):
    if not os.path.exists("../data/output_round_data_%s.csv" % (year)):
        generate_year_data(year)
    data = pd.read_csv("../data/output_round_data_%s.csv" % (year))
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

def pull_raw_training_data(norm_to_rate = True, add_share_feats = True, score = 'binary', drop_finishes = True, single_row = False, add_gender = False, add_class = False):
    if not os.path.exists("raw_round_data.csv"):
        generate_raw_round_data()
    data = pd.read_csv("../data/raw_round_data.csv")
    data.set_index("oid", inplace = True)
    if (drop_finishes):
        data = data[data['finish'] != 1]
    if (score == 'binary'):
        data['score'] = data['score'].apply(lambda x: 1 if x > 0 else -1)
    elif score == 'finish':
        data['score'] = data['offFinish'] - data['defFinish']
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

def combinedRoundStats(data):
    strike_feats = []
    for strike_col in ['BodySigStrike', 'ClinchSigStrike', 'DistanceSigStrike', 'GroundSigStrike', 'HeadSigStrike', 'LegSigStrike', 'TotStrike', 'Takedown', 'Submission']:
        strike_feats.append(strike_col+'TotAccuracy')
        strike_feats.append(strike_col+'TotSuccessfulVolume')    
        strike_feats.append(strike_col+'TotAttemptedVolume')   
        
        data[strike_col+'TotAttemptedVolume'] = (data['off'+strike_col+'Attempted']+data['def'+strike_col+'Attempted'])/data['seconds']
        data[strike_col+'TotSuccessfulVolume'] = (data['off'+strike_col+'Successful']+data['def'+strike_col+'Successful'])/data['seconds']
        data[strike_col+'TotAccuracy'] = data[strike_col+'TotSuccessfulVolume']/data[strike_col+'TotAttemptedVolume']
        data[strike_col+'TotAccuracy'].fillna(0, inplace = True)
 
    for tech_feat in ['Knockdowns', 'PassSuccessful', 'ReversalSuccessful']:
        strike_feats.append("tot" + tech_feat + "Volume")
        data["tot" + tech_feat + "Volume"] = (data['off'+tech_feat]+data['def'+tech_feat])/data['seconds']
    data['SubmissionTotAccuracy'].replace([np.inf, -np.inf], 0, inplace = True)
    tot_data = data[strike_feats]
    return tot_data

