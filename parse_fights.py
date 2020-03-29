#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 17:06:55 2019

@author: eric.hensleyibm.com
"""

# importing the requests library 
import requests 
  
HOST = '207.237.93.29'
PORT = '4646'
ADD_ODDS = True
ADD_SCORES = False
#for year in range(2000, 2020):
#    r = requests.get(url = "http://localhost:8686/ufc/parse/fights/judgeScores/%s" % (year))
#
INIT_FIGHT_URL = "http://%s:%s/ufc/parse/fights"
PARSE_YEAR_JUDGE_SCORE_URL = "http://%s:%s/ufc/parse/fights/judgeScores/%s"
ADD_BOUTS_TO_FIGHT = "http://%s:%s/ufc/parse/fights/%s/bouts"

GET_BOUTS_FROM_FIGHT = "http://%s:%s/ufc/rest/fight/%s/details"

ADD_BOUTS_DETAILS = "http://%s:%s/ufc/parse/fights/%s/bouts/%s"
ADD_FIGHT_ODDS_URL = "http://%s:%s/ufc/parse/odds/fight/%s/fightOdds/%s"

ADD_BOUT_SCORE_URL = "http://%s:%s/ufc/parse/fights/judgeScores/fight/%s"
SCRAPE_BOUT_SCORE = "http://%s:%s/ufc/parse/fights/judgeScores/bout/%s"

GET_BOUT = "http://%s:%s/ufc/rest/bout/%s/info"

ADD_FIGHT_EXP_OUTCOME = "http://%s:%s/ufc/parse/odds/fight/%s/expOutcomes"
ADD_FIGHT_ODDS = "http://%s:%s/ufc/parse/odds/fight/%s/odds"

GET_ALL_FIGHT_IDS = "http://%s:%s/ufc/rest/fight/all"

ADD_ROUND_SCORE = "http://%s:%s/ufc/scores/bout/%s/round/add"

def initUpdate():
    r = requests.get(url = INIT_FIGHT_URL % (HOST, PORT))
    r = requests.get(url = PARSE_YEAR_JUDGE_SCORE_URL % (HOST, PORT, "2020"))

def addBoutsToFight(fight_id):
    r = requests.get(url = ADD_BOUTS_TO_FIGHT % (HOST, PORT, fight_id))
    response = r.json()
    print("Add Bouts to Fight %s completed %s with %s bouts found and %s completed" % (fight_id, response['status'], response['itemsFound'], response['itemsCompleted']))

def getBoutsFromFight(fight_id):
    r = requests.get(url = GET_BOUTS_FROM_FIGHT % (HOST, PORT, fight_id))
    response = r.json()

    if (response['status'] == 404):
        print("Get bouts from fight %s failed" % (fight_id))
    else:
        return response['response']

def addFightOddsUrl(fight_details):
    print("Please provide the bestFightOdds url for %s (%s)" % (fight_details['fightName'], fight_details['fightDate']))
    fight_url = input()
    r = requests.get(url = ADD_FIGHT_ODDS_URL % (HOST, PORT, fight_details['fightId'], fight_url.replace("https://www.bestfightodds.com/events/", "")))
    response = r.json()
    print("Add bestFightOdds (%s) to fight %s completed with %s" % (fight_url, fight_details['fightName'], response['status']))
 
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
      
def addBoutScoreUrls(fight_oid):
    r = requests.get(url = ADD_BOUT_SCORE_URL % (HOST, PORT, fight_oid))
    response = r.json()
    print("Add Bouts Score URLs to Fight %s completed with %s with %s bouts found and %s completed" % (fight_oid, response['status'], response['itemsFound'], response['itemsCompleted']))

def addBoutDetails(fight_id, bout_id):
    r = requests.get(url = ADD_BOUTS_DETAILS % (HOST, PORT, fight_id, bout_id))
    response = r.json()
    if (response['status'] != 'OK'):
        print("Add Bouts Details to Bout %s failed" % (bout_id))
        return False
    else:
        print("Add Bouts Details to Bout %s completed with %s with %s fighters found and %s completed" % (bout_id, response['status'], response['itemsFound'], response['itemsCompleted']))
        return True

def scrapeBoutScores(bout_id):
    r = requests.get(url = SCRAPE_BOUT_SCORE % (HOST, PORT, bout_id))
    response = r.json()
    print("Add round scores to Bout %s completed wtih %s with %s rounds found and %s completed" % (bout_id, response['status'], response['itemsFound'], response['itemsCompleted']))

def refreshBout(bout_id):
    r = requests.get(url = GET_BOUT % (HOST, PORT, bout_id))
    response = r.json() 
    return response['response']

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
    
def addFightExpectedOutcomes(fight_id):
    r = requests.get(url = ADD_FIGHT_EXP_OUTCOME % (HOST, PORT, fight_id))
    response = r.json() 
    print("Add bout expected outcomes to Fight %s completed wtih %s with %s rounds found and %s completed" % (fight_id, response['status'], response['itemsFound'], response['itemsCompleted']))

def addFightOdds(fight_id):
    r = requests.get(url = ADD_FIGHT_ODDS % (HOST, PORT, fight_id))
    response = r.json() 
    print("Add bout odds to Fight %s completed wtih %s with %s rounds found and %s completed" % (fight_id, response['status'], response['itemsFound'], response['itemsCompleted']))

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
    if ADD_ODDS and fight_details['bestFightOddsUrl'] is None:
        addFightOddsUrl(fight_details)
    for bout_detail in fight_details['bouts']:  
        if (bout_detail['completed']):
            if (len(bout_detail['fighterBoutXRefs']) != 2):
                print("~~~~~~ INCOMPLETE BOUT (%s) ~~~~~~~~~" % (bout_detail['oid']))
            print("Bout %s already completed.. skipping detail scrape")
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
#    if ADD_ODDS:
    fight_details_refreshed = getBoutsFromFight(fight_id)
    odds_completion = evalIfMissingFightOddsInfo(fight_details_refreshed)
    if odds_completion['odds']:
        addFightOdds(fight_id)
    if odds_completion['expectedOutcome']:
        addFightExpectedOutcomes(fight_id)
        
def getAllFightIds():
    r = requests.get(url = GET_ALL_FIGHT_IDS % (HOST, PORT))
    response = r.json()         
    return response['response']

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
                r = requests.post(url = ADD_ROUND_SCORE % (HOST, PORT, bout_details['oid']), json = round_dict)
                response = r.json() 
                if response['status'] != 'OK':
                    print("Save for %s round %s failed with.. %s" % (fighter_name, round_dict['round'], response['errorMsg']))
            else:
                print("Round %s score for %s cannot be Null" % (round_dict['round'], fighter_name))
     
def addInfoToAllFights():
    fight_id_list = getAllFightIds()
    
#    addInfoToAllBouts(fight_id_list[2])
#    fight_id = fight_id_list[0]
#    
    for fight_id in fight_id_list:
        addInfoToAllBouts(fight_id)

# Fixed        
# fix stipe vs cormier 2
# ferguson vs cerrone, round 1 make tony 27
# cannonier vs silva, round 1 make cannonier 30
# chaisson vs moras, round 2 chiasson 30    
# de la rosa vs kassem, round 1 de la rosa 30    
# maia vs good, round 1 good 15    
# hunt vs willis, round 1 willis 27
        
# TODO
def addScoresToAllFights():
    fight_id_list = getAllFightIds()    
    for fight_id in fight_id_list:
        addFightMissingRoundScores(fight_id)
        
if __name__ == '__main__':
    addInfoToAllFights()