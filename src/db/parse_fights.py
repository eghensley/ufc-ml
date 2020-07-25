#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 17:06:55 2019

@author: eric.hensleyibm.com
"""

#import sys
#if __name__ == "__main__":
#    sys.path.append("..")
    
from spring import addRoundScore, getAllFightIds, getBoutsFromFight, addBoutScoreUrls, \
                         addBoutsToFight, addBoutDetails, addFightOddsUrl, refreshBout, \
                         scrapeBoutScores, addFightOdds, addFightExpectedOutcomes, getTrainingFights
from predictors import insert_new_ml_scores, insert_new_ml_prob
from elo import populate_elo

global ADD_ODDS
global ADD_SCORES
ADD_ODDS = True
ADD_SCORES = False

#refreshBout('0fcf42b68f2f0fa3')


#ALTER TABLE ufc2.bfo_expected_outcome DISABLE TRIGGER ALL;
#ALTER TABLE ufc2.fighter_bout_xref DISABLE TRIGGER ALL;
#
#delete from ufc2.bfo_expected_outcome beo 
#	where beo.fighter_bout_oid 
#		in (select fbx.oid from ufc2.bout b 
#			join ufc2.fighter_bout_xref fbx 
#			on b.oid = fbx.bout_oid 
#			join ufc2.fighter f 
#			on f.oid = fbx.fighter_oid 
#		where b.fight_oid = 'cd200d5f-2813-4ab0-9eb2-f9a026883d79' 
#		order by b.oid);
#		
#delete from ufc2.fighter_bout_xref fbx2 
#	where fbx2.oid 
#		in (select fbx.oid from ufc2.bout b 
#			join ufc2.fighter_bout_xref fbx 
#			on b.oid = fbx.bout_oid 
#			join ufc2.fighter f 
#			on f.oid = fbx.fighter_oid 
#		where b.fight_oid = 'cd200d5f-2813-4ab0-9eb2-f9a026883d79' 
#		order by b.oid);
#
#ALTER TABLE ufc2.bfo_expected_outcome ENABLE TRIGGER ALL;
#ALTER TABLE ufc2.fighter_bout_xref ENABLE TRIGGER ALL;
#
#delete from ufc2.bout b2 
#	where b2.oid 
#		in (select b.oid 
#			from ufc2.bout b 
#		where fight_oid = 'cd200d5f-2813-4ab0-9eb2-f9a026883d79');
#
#update ufc2.fight set f_completed = false where oid = 'cd200d5f-2813-4ab0-9eb2-f9a026883d79';


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
    
#   fight_id = "271fe91f4ba9d2c5"   31ceaf0e670c1578
def addInfoToAllBouts(fight_id):
    fight_details = getBoutsFromFight(fight_id)
    print("Beginning full parse of %s (%s)" % (fight_details['fightName'], fight_details['fightId']))
    fight_oid = fight_details["oid"]
    if evalIfAllBoutScoreUrls(fight_details):
        print("Adding bout score URLs")
        addBoutScoreUrls(fight_oid)
    if not fight_details['completed']:
        print("No bouts included for fight... adding")
        addBoutsToFight(fight_id)
        print("Refreshing fight info after adding bouts")
        fight_details = getBoutsFromFight(fight_id)
#    if ADD_ODDS and fight_details['bestFightOddsUrl'] is None:
#        addFightOddsUrl(fight_details)
    for bout_detail in fight_details['bouts']:
        if (bout_detail['completed']):
            if (len(bout_detail['fighterBoutXRefs']) != 2):
                print("~~~~~~ INCOMPLETE BOUT (%s) ~~~~~~~~~" % (bout_detail['oid']))
            print("Bout %s already completed.. skipping detail scrape" % bout_detail['oid'])
            if (evalIfMissingBoutScores(bout_detail)):
                print("Adding bout round scores")
                scrapeBoutScores(bout_detail['oid'])
        else:
            print("Beginning update of bout %s" % (bout_detail['boutId']))
            bout_detail_parse_response = addBoutDetails(fight_id, bout_detail['boutId'])
            if (bout_detail_parse_response != True):
                break
            refreshed_bout_detail = refreshBout(bout_detail['boutId'])
            if (evalIfMissingBoutScores(refreshed_bout_detail)):
                print("Adding bout round scores")
                scrapeBoutScores(bout_detail['oid'])
#        if bout_detail['gender'] == 'MALE':
        print("Updating ML scores")
        insert_new_ml_scores(bout_detail['boutId'])
        insert_new_ml_prob(bout_detail['boutId'])
        print("Updating ELO scores")
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
    
#    fight_id_list = ['dbd198f780286aca',
#'c32eab6c2119e989',
#'2eab7a6c8b0ed8cc',
#'1e13936d708bcff7',
#'4c12aa7ca246e7a4',
#'14b9e0f2679a2205',
#
#'dfb965c9824425db',
#'5f8e00c27b7e7410',
#'898337ef520fe4d3',
#'53278852bcd91e11',
#'0b5b6876c2a4723f',
#
#'fc9a9559a05f2704',
#'33b2f68ef95252e0',
#'5df17b3620145578',
#'b26d3e3746fb4024',
#'44aa652b181bcf68',
#'0c1773639c795466']
    for fight_id in reversed(fight_id_list):
        addInfoToAllBouts(fight_id)

def pop_year_bouts():
    for year in range(2005, 2021):
        fights = getTrainingFights(year)
        for fight in fights:
#            try:
                addInfoToAllBouts(fight)
#            except:
#                input("error.. proceed?")
        
def addScoresToAllFights():
    fight_id_list = getAllFightIds()    
    for fight_id in fight_id_list:
        addFightMissingRoundScores(fight_id)
        
#if __name__ == '__main__':
#    pop_year_bouts()