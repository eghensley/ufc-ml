#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 17:16:38 2020

@author: ehens86
"""

from spring.api_wrappers import getAllBouts, refreshBout, getLastEloCount, getLastElo
from datetime import datetime
import pandas as pd
import json
import os

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)

def generate_json_data():
    print('generating json elo data')
    bouts = getAllBouts()
    fight_data = {}
    for bout_oid in bouts:
        bout_data = format_elo_data(bout_oid)
        if bout_data is not None:
            fight_data[bout_oid] = bout_data
        
    with open("src/training/elo/elo_preds.json", "w") as file:
        json.dump(fight_data, file)
    
def generate_csv_data():
    print('generating csv elo data')
    if not os.path.exists("src/training/elo/elo_preds.json"):
        generate_json_data()
    
    with open("src/training/elo/elo_preds.json", "r") as file:
        fight_data = json.load(file)

    training_data = {}
    for bout_oid, fight in fight_data.items():
        bout_training_data = format_elo_data_2(fight)
        if bout_training_data is not None:
            training_data[bout_oid] = bout_training_data
        
    training_df = pd.DataFrame.from_dict(training_data).T
    training_df.to_csv("training/elo/elo_predictors.csv")
    
def pull_ml_training():
    if not os.path.exists('src/training/elo/elo_predictors.csv'):
        generate_csv_data()
    
    df = pd.read_csv('src/training/elo/elo_predictors.csv')
    df.set_index('Unnamed: 0', inplace = True)
#    test = df.loc['0b89c88d506da083']
    return df
    
    
def format_elo_data(bout_oid, use_prev = False):
    bout_info = refreshBout(bout_oid)
#    if bout_info['gender'] == 'FEMALE':
#        return
    bout_data = {}

    bout_data['schedRounds'] = bout_info['schedRounds']
    for fbx in bout_info['fighterBoutXRefs']:
        fighter_data = {}

        fighter_data['mlOdds'] = fbx['mlOdds']
        if use_prev:
            prev_data = getLastElo(fbx['fighter']['oid'], bout_info['fightOid'])
#            print(prev_data)
            if prev_data['offStrikeEloPost'] is None:
                continue
            fighter_data['offStrikeEloPre'] = prev_data['offStrikeEloPost']
            if prev_data['defStrikeEloPost'] is None:
                continue
            fighter_data['defStrikeEloPre'] = prev_data['defStrikeEloPost']
            if prev_data['offGrapplingEloPost'] is None:
                continue
            fighter_data['offGrapplingEloPre'] = prev_data['offGrapplingEloPost']
            if prev_data['defGrapplingEloPost'] is None:
                continue
            fighter_data['defGrapplingEloPre'] = prev_data['defGrapplingEloPost']
            if prev_data['powerStrikeEloPost'] is None:
                continue
            fighter_data['powerStrikeEloPre'] = prev_data['powerStrikeEloPost']
            if prev_data['chinStrikeEloPost'] is None:
                continue
            fighter_data['chinStrikeEloPre'] = prev_data['chinStrikeEloPost']
            if prev_data['subGrapplingEloPost'] is None:
                continue
            fighter_data['subGrapplingEloPre'] = prev_data['subGrapplingEloPost']
            if prev_data['evasGrapplingEloPost'] is None:
                continue
            fighter_data['evasGrapplingEloPre'] = prev_data['evasGrapplingEloPost']
        else:
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
        
        fighter_data['prev_fights'] = getLastEloCount(fbx['fighter']['oid'], bout_info['fightOid'])
        fighter_data['age'] = days_between(fbx['fighter']['dob'].split('T')[0], bout_info['fightDate'].split('T')[0])/365
        fighter_data['date'] = bout_info['fightDate']
        
        if use_prev:
            fighter_data['outcome'] = None
        else:
            if fbx['outcome'] is None:
                continue
            if fbx['outcome'] not in ['W', 'L']:
                continue
            fighter_data['outcome'] = fbx['outcome']
        
        bout_data[fbx['fighter']['oid']] = fighter_data
    
    return bout_data
            
def format_elo_data_2(fight):
    bout_training_data = {}
    fighters = [i for i in fight.keys() if i != 'schedRounds']
    if len(fighters) != 2:
        return
    
    eff_strike_1 = (fight[fighters[0]]['offStrikeEloPre'] + (1 - fight[fighters[1]]['defStrikeEloPre']))/2
    eff_strike_2 = (fight[fighters[1]]['offStrikeEloPre'] + (1 - fight[fighters[0]]['defStrikeEloPre']))/2
    
    eff_grap_1 = (fight[fighters[0]]['offGrapplingEloPre'] + (1 - fight[fighters[1]]['defGrapplingEloPre']))/2
    eff_grap_2 = (fight[fighters[1]]['offGrapplingEloPre'] + (1 - fight[fighters[0]]['defGrapplingEloPre']))/2
    
    eff_ko_1 = (fight[fighters[0]]['powerStrikeEloPre'] + (1 - fight[fighters[1]]['chinStrikeEloPre']))/2
    eff_ko_2 = (fight[fighters[1]]['powerStrikeEloPre'] + (1 - fight[fighters[0]]['chinStrikeEloPre']))/2
    
    eff_sub_1 = (fight[fighters[0]]['subGrapplingEloPre'] + (1 - fight[fighters[1]]['evasGrapplingEloPre']))/2
    eff_sub_2 = (fight[fighters[1]]['subGrapplingEloPre'] + (1 - fight[fighters[0]]['evasGrapplingEloPre']))/2
    
    age_1 = fight[fighters[0]]['age']
    age_2 = fight[fighters[1]]['age']
    
    prev_fights_1 = fight[fighters[0]]['prev_fights']
    prev_fights_2 = fight[fighters[1]]['prev_fights']

    bout_training_data['off_strike_1'] = fight[fighters[0]]['offStrikeEloPre']
    bout_training_data['def_strike_1'] = fight[fighters[0]]['defStrikeEloPre']
    bout_training_data['off_grapp_1'] = fight[fighters[0]]['offGrapplingEloPre']
    bout_training_data['def_grapp_1'] = fight[fighters[0]]['defGrapplingEloPre']
    bout_training_data['off_sub_1'] = fight[fighters[0]]['subGrapplingEloPre']
    bout_training_data['def_sub_1'] = fight[fighters[0]]['evasGrapplingEloPre']
    bout_training_data['off_ko_1'] = fight[fighters[0]]['powerStrikeEloPre']
    bout_training_data['def_ko_1'] = fight[fighters[0]]['chinStrikeEloPre']

    bout_training_data['off_strike_2'] = fight[fighters[1]]['offStrikeEloPre']
    bout_training_data['def_strike_2'] = fight[fighters[1]]['defStrikeEloPre']
    bout_training_data['off_grapp_2'] = fight[fighters[1]]['offGrapplingEloPre']
    bout_training_data['def_grapp_2'] = fight[fighters[1]]['defGrapplingEloPre']
    bout_training_data['off_sub_2'] = fight[fighters[1]]['subGrapplingEloPre']
    bout_training_data['def_sub_2'] = fight[fighters[1]]['evasGrapplingEloPre']
    bout_training_data['off_ko_2'] = fight[fighters[1]]['powerStrikeEloPre']
    bout_training_data['def_ko_2'] = fight[fighters[1]]['chinStrikeEloPre']
    
    
    bout_training_data['off_strike_diff'] = bout_training_data['off_strike_1'] - bout_training_data['off_strike_2']
    bout_training_data['def_strike_diff'] = bout_training_data['def_strike_1'] - bout_training_data['def_strike_2']
    bout_training_data['off_grapp_diff'] = bout_training_data['off_grapp_1'] - bout_training_data['off_grapp_2']
    bout_training_data['def_grapp_diff'] = bout_training_data['def_grapp_1'] - bout_training_data['def_grapp_2']
    bout_training_data['off_sub_diff'] = bout_training_data['off_sub_1'] - bout_training_data['off_sub_2']
    bout_training_data['def_sub_diff'] = bout_training_data['def_sub_1'] - bout_training_data['def_sub_2']
    bout_training_data['off_ko_diff'] = bout_training_data['off_ko_1'] - bout_training_data['off_ko_2']
    bout_training_data['def_ko_diff'] = bout_training_data['def_ko_1'] - bout_training_data['def_ko_2']
        
    bout_training_data['off_strike_share'] = bout_training_data['off_strike_1'] / (bout_training_data['off_strike_1'] + bout_training_data['off_strike_2'])
    bout_training_data['def_strike_share'] = bout_training_data['def_strike_1'] / (bout_training_data['def_strike_1'] + bout_training_data['def_strike_2'])
    bout_training_data['off_grapp_share'] = bout_training_data['off_grapp_1'] / (bout_training_data['off_grapp_1'] + bout_training_data['off_grapp_2'])
    bout_training_data['def_grapp_share'] = bout_training_data['def_grapp_1'] / (bout_training_data['def_grapp_1'] + bout_training_data['def_grapp_2'])
    bout_training_data['off_sub_share'] = bout_training_data['off_sub_1'] / (bout_training_data['off_sub_1'] + bout_training_data['off_sub_2'])
    bout_training_data['def_sub_share'] = bout_training_data['def_sub_1'] / (bout_training_data['def_sub_1'] + bout_training_data['def_sub_2'])
    bout_training_data['off_ko_share'] = bout_training_data['off_ko_1'] / (bout_training_data['off_ko_1'] + bout_training_data['off_ko_2'])
    bout_training_data['def_ko_share'] = bout_training_data['def_ko_1'] / (bout_training_data['def_ko_1'] + bout_training_data['def_ko_2'])
    
    bout_training_data['eff_strike_1'] = eff_strike_1
    bout_training_data['eff_strike_2'] = eff_strike_2
    bout_training_data['eff_strike_diff'] = eff_strike_1 - eff_strike_2
    bout_training_data['eff_strike_share'] = eff_strike_1 / (eff_strike_1 + eff_strike_2)
    
    bout_training_data['eff_grap_1'] = eff_grap_1
    bout_training_data['eff_grap_2'] = eff_grap_2
    bout_training_data['eff_grap_diff'] = eff_grap_1 - eff_grap_2
    bout_training_data['eff_grap_share'] = eff_grap_1 / (eff_grap_1 + eff_grap_2)

    bout_training_data['eff_ko_1'] = eff_ko_1
    bout_training_data['eff_ko_2'] = eff_ko_2
    bout_training_data['eff_ko_diff'] = eff_ko_1 - eff_ko_2
    bout_training_data['eff_ko_share'] = eff_ko_1 / (eff_ko_1 + eff_ko_2)

    bout_training_data['eff_sub_1'] = eff_sub_1
    bout_training_data['eff_sub_2'] = eff_sub_2
    bout_training_data['eff_sub_diff'] = eff_sub_1 - eff_sub_2
    bout_training_data['eff_sub_share'] = eff_sub_1 / (eff_sub_1 + eff_sub_2)

    bout_training_data['age_1'] = age_1
    bout_training_data['age_2'] = age_2
    bout_training_data['age_diff'] = age_1 - age_2
    bout_training_data['age_share'] = age_1 / (age_1 + age_2)

    bout_training_data['date'] = fight[fighters[0]]['date']
    
    if fight[fighters[0]]['outcome'] == 'W':
        bout_training_data['winner'] = 1
    else:
        bout_training_data['winner'] = 0
        
    bout_training_data['prev_fights_1'] = prev_fights_1
    bout_training_data['prev_fights_2'] = prev_fights_2
    bout_training_data['rounds'] = fight['schedRounds']
    return bout_training_data
    
#   bout_oid = 'e16f42a666e163d8'
def form_new_ml_odds_data(bout_oid):
    try:
        fight = format_elo_data(bout_oid, use_prev = True)
        bout_training_data = format_elo_data_2(fight)
        df = pd.DataFrame.from_dict({bout_oid: bout_training_data}).T
        return df
    except:
        return

