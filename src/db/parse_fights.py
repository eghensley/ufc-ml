#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 17:06:55 2019

@author: eric.hensleyibm.com
"""

import sys, os
if __name__ == "__main__":
    sys.path.append("src")
    os.environ['ufc.flask.spring.host'] = 'http://localhost:4646'
    os.environ['ufc.flask.spring.pw'] = '1234'

    print(os.environ)
    
from spring import addRoundScore, getAllFightIds, getBoutsFromFight, addBoutScoreUrls, \
                         addBoutsToFight, addBoutDetails, addFightOddsUrl, refreshBout, \
                         scrapeBoutScores, addFightOdds, addFightExpectedOutcomes, \
                         getTrainingFights
from predictors import insert_new_ml_scores, insert_new_ml_prob
from elo import populate_elo

global ADD_ODDS
global ADD_SCORES
ADD_ODDS = True
ADD_SCORES = False

def evalIfAllBoutScoreUrls(fight_details):
    if (ADD_SCORES):
        for bout_details in fight_details['bouts']:
            if bout_details['finishMethod'] in ['KO_TKO', 'SUB'] :
                continue
            if bout_details['mmaDecBoutUrl'] is None:
                return True
        return False
    else:
        return False
      
def evalIfMissingBoutScores(bout_detail):
    if (ADD_SCORES):
        if bout_detail['mmaDecBoutUrl'] is None:
            print("No bout score URL available for bout %s" % bout_detail['boutId'])
            if (bout_detail['finishMethod'] in ['KO_TKO', 'SUB']):
                print("Bout ended before decision, manual addition required")
            return False
        for x_ref in bout_detail['fighterBoutXRefs']:
            for bout_round in x_ref["boutDetails"]:
                if bout_round["score"] is None:
                    return True
        return False
    else:
        return False
    
def evalIfMissingFightOddsInfo(fight_details):
    odds_completion = {'odds': False, 'expectedOutcome': False}
    if fight_details['bestFightOddsUrl'] is None:
        return odds_completion
    for bout_detail in fight_details['bouts']:
        for x_ref in bout_detail['fighterBoutXRefs']:
            if x_ref['mlOdds'] is None:
                odds_completion['odds'] = True
            if x_ref['bfoExpectedOutcomes'] is None:
                odds_completion['expectedOutcome'] = True
    if odds_completion['odds']:
        print('Fight %s is missing odds data' % (fight_details['fightName']))
    if odds_completion['expectedOutcome']:
        print('Fight %s is missing expected outcome data' % (fight_details['fightName']))
    return odds_completion
    
#   fight_id = "7a82635ffa9b59fe"
def addInfoToAllBouts(fight_id):
    fight_details = getBoutsFromFight(fight_id)
    print("addInfoToAllBouts - Beginning full parse of %s (%s)" % (fight_details['fightName'], fight_details['fightId']))
    fight_oid = fight_details["oid"]
    if evalIfAllBoutScoreUrls(fight_details):
        print("addInfoToAllBouts - Adding bout score URLs")
        addBoutScoreUrls(fight_oid)
    if not fight_details['completed']:
        print("addInfoToAllBouts - No bouts included for fight... adding")
        addBoutsToFight(fight_id)
        print("addInfoToAllBouts - Refreshing fight info after adding bouts")
        fight_details = getBoutsFromFight(fight_id)
#    if ADD_ODDS and fight_details['bestFightOddsUrl'] is None:
#        addFightOddsUrl(fight_details)
    for bout_detail in fight_details['bouts']:
        if (bout_detail['completed']):
            if (len(bout_detail['fighterBoutXRefs']) != 2):
                print("~~~~~~ INCOMPLETE BOUT (%s) ~~~~~~~~~" % (bout_detail['oid']))
            print("addInfoToAllBouts - Bout %s already completed.. skipping detail scrape" % bout_detail['oid'])
            if (evalIfMissingBoutScores(bout_detail)):
                print("addInfoToAllBouts - Adding bout round scores")
                scrapeBoutScores(bout_detail['oid'])
        else:
            print("addInfoToAllBouts - Beginning update of bout %s" % (bout_detail['boutId']))
            bout_detail_parse_response = addBoutDetails(fight_id, bout_detail['boutId'])
            if (bout_detail_parse_response != True):
                break
            refreshed_bout_detail = refreshBout(bout_detail['boutId'])
            if (evalIfMissingBoutScores(refreshed_bout_detail)):
                print("addInfoToAllBouts - Adding bout round scores")
                scrapeBoutScores(bout_detail['oid'])
#        if bout_detail['gender'] != 'MALE':
        print("addInfoToAllBouts - Updating ML scores")
        insert_new_ml_scores(bout_detail['boutId'])
        insert_new_ml_prob(bout_detail['boutId'])
        print("addInfoToAllBouts - Updating ELO scores")
        populate_elo(bouts = [bout_detail['boutId']], refit = False)
#        populate_elo_bout(bout_detail['boutId'])
    fight_details_refreshed = getBoutsFromFight(fight_id)
    odds_completion = evalIfMissingFightOddsInfo(fight_details_refreshed)
    if odds_completion['odds']:
        addFightOdds(fight_id)
    if odds_completion['expectedOutcome']:
        addFightExpectedOutcomes(fight_id)

def addFightOddsUrls(fight_id):
    fight_details = getBoutsFromFight(fight_id)
    print("Beginning fight odds url parse of %s - %s (%s)" % (fight_details['fightName'], fight_details['fightDate'], fight_details['fightId']))
    if fight_details['bestFightOddsUrl'] is None:
        addFightOddsUrl(fight_details)   
    fight_details_refreshed = getBoutsFromFight(fight_id)
    odds_completion = evalIfMissingFightOddsInfo(fight_details_refreshed)
    if odds_completion['odds']:
        addFightOdds(fight_id)
    if odds_completion['expectedOutcome']:
        addFightExpectedOutcomes(fight_id)    
    
def addFightOddsToAllFights():
    fight_id_list = getAllFightIds()
    for fight_id in fight_id_list:
        addFightOddsUrls(fight_id)
   
def evalIfFightMissingRoundScores(fight_details):
    if fight_details['mmaDecFightUrl'] is None:
        print("Decision URL missing for %s... please add" % (fight_details["fightName"]))
        return False
    for bout_details in fight_details['bouts']:
        for xref in bout_details['fighterBoutXRefs']:
            for bout_round in xref['boutDetails']:
                if bout_round['score'] is None:
                    print("%s is missing round score data (%s)" % (fight_details["fightName"], fight_details["fightDate"]))
                    return True
    return False     

def evalIfBoutMissingRoundScores(bout_details):
    for xref in bout_details['fighterBoutXRefs']:
        for bout_round in xref['boutDetails']:
            if bout_round['score'] is None:
                print("  %s VS %s" % (bout_details['fighterBoutXRefs'][0]['fighter']['fighterName'], bout_details['fighterBoutXRefs'][1]['fighter']['fighterName']))
                return True
    return False          
        
def addFightMissingRoundScores(fight_id):
    fight_details = getBoutsFromFight(fight_id)
    if evalIfFightMissingRoundScores(fight_details):
        skipKey = input()
        if skipKey.upper() == "SKIP":
            return
        for bout_details in fight_details['bouts']:
            if evalIfBoutMissingRoundScores(bout_details):
                addBoutMissingRoundScores(bout_details)
 
def addBoutMissingRoundScores(bout_details):
    score_dict = {bout_details['fighterBoutXRefs'][0]['fighter']['fighterName']: {'id': bout_details['fighterBoutXRefs'][0]['fighter']['fighterId'], 'scores': []},
                  bout_details['fighterBoutXRefs'][1]['fighter']['fighterName']: {'id': bout_details['fighterBoutXRefs'][1]['fighter']['fighterId'], 'scores': []}} 
    
    for fighter in bout_details['fighterBoutXRefs']:
        for bout_round in fighter['boutDetails']:
            if bout_round['score'] is None:
                score_dict[fighter['fighter']['fighterName']]['scores'].append({"round":bout_round['round'], "score": None, 'oid': bout_round['oid']})
        
    for fighter_name in score_dict.keys():
        print("    %s" % (fighter_name))
        for round_dict in score_dict[fighter_name]['scores']:
            print("      Round %s" % (round_dict['round']))
            try:    
                round_dict['score'] = int(input())
            except:
                continue
    for fighter_name in score_dict.keys():
        for round_dict in score_dict[fighter_name]['scores']:
            if (round_dict['score'] is not None):
                addRoundScore(bout_details['oid'], round_dict, fighter_name)
            else:
                print("Round %s score for %s cannot be Null" % (round_dict['round'], fighter_name))
     
def addInfoToAllFights():
    fight_id_list = getAllFightIds()
    
    for fight_id in reversed(fight_id_list):
        addInfoToAllBouts(fight_id)

def pop_year_bouts():
    for year in range(2005, 2021):
        fights = getTrainingFights(year)
        for fight in fights:
            addInfoToAllBouts(fight)
        
def addScoresToAllFights():
    fight_id_list = getAllFightIds()    
    for fight_id in fight_id_list:
        addFightMissingRoundScores(fight_id)
