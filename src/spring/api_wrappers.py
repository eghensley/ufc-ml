#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:22:50 2020

@author: ehens86
"""

if (__name__ == "__main__"):
    import sys
    sys.path.append("..")

from .config import CONFIG

import requests 

def clearElo():
    headers = requests.utils.default_headers()
    headers['password'] = str(CONFIG['spring']['PW'])
    r = requests.get(url = CONFIG['spring']['rest']['CLEAR_ELO'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT']), headers = headers)
    response = r.json()
    return response

def updateElo(payload):
    headers = requests.utils.default_headers()
    headers['password'] = str(CONFIG['spring']['PW'])
    while True:
        try:
            r = requests.post(url = CONFIG['spring']['rest']['UPDATE_ELO'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT']), json = payload, headers = headers)
            response = r.json()
            return response
        except requests.exceptions.RequestException as err:
            print ("OOps: Something Else",err)
            pass
        except requests.exceptions.HTTPError as errh:
            print ("Http Error:",errh)
            pass
        except requests.exceptions.ConnectionError as errc:
            print ("Error Connecting:",errc)
            pass
        except requests.exceptions.Timeout as errt:
            print ("Timeout Error:",errt)  
            pass
    
def getLastEloCount(fighterOid, fightOid, debug = False):
    while True:
        try:
            r = requests.get(url = CONFIG['spring']['rest']['GET_LAST_ELO_COUNT'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], fighterOid, fightOid))
            response = r.json() 
            if response['errorMsg'] is not None and debug:
                print(response['errorMsg'])
            return response['response']
        except requests.exceptions.RequestException as err:
            print ("OOps: Something Else",err)
            pass
        except requests.exceptions.HTTPError as errh:
            print ("Http Error:",errh)
            pass
        except requests.exceptions.ConnectionError as errc:
            print ("Error Connecting:",errc)
            pass
        except requests.exceptions.Timeout as errt:
            print ("Timeout Error:",errt)  
            pass
        
def getLastElo(fighterOid, fightOid, debug = False):
    while True:
        try:
            r = requests.get(url = CONFIG['spring']['rest']['GET_LAST_ELO'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], fighterOid, fightOid))
            response = r.json() 
            if response['errorMsg'] is not None and debug:
                print(response['errorMsg'])
            return response['response']
        except requests.exceptions.RequestException as err:
            print ("OOps: Something Else",err)
            pass
        except requests.exceptions.HTTPError as errh:
            print ("Http Error:",errh)
            pass
        except requests.exceptions.ConnectionError as errc:
            print ("Error Connecting:",errc)
            pass
        except requests.exceptions.Timeout as errt:
            print ("Timeout Error:",errt)  
            pass
        
def saveMlScore(payload):
    headers = requests.utils.default_headers()
    headers['password'] = str(CONFIG['spring']['PW'])
    r = requests.post(url = CONFIG['spring']['rest']['ADD_ML_SCORE'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT']), json = payload, headers = headers)
    response = r.json()
    return response

def saveBoutMlScore(payload):
    headers = requests.utils.default_headers()
    headers['password'] = str(CONFIG['spring']['PW'])
    r = requests.post(url = CONFIG['spring']['rest']['ADD_BOUT_ML_SCORE'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT']), json = payload, headers = headers)
    response = r.json()
    return response

def getBoutData(boutOid):
    r = requests.get(url = CONFIG['spring']['rest']['GET_BOUT_DATA_BY_OID'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], boutOid))
    response = r.json()
    if response['errorMsg'] is not None:
        print(response['errorMsg'])
    return response['response']

def addRoundScore(oid, round_dict, fighter_name):
    headers = requests.utils.default_headers()
    headers['password'] = str(CONFIG['spring']['PW'])
    r = requests.post(url = CONFIG['spring']['rest']['ADD_ROUND_SCORE'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], oid), json = round_dict, headers = headers)
    response = r.json() 
    if response['status'] != 'OK':
        print("Save for %s round %s failed with.. %s" % (fighter_name, round_dict['round'], response['errorMsg']))

def addFutBoutData(payload):
    headers = requests.utils.default_headers()
    headers['password'] = str(CONFIG['spring']['PW'])
    r = requests.post(url = CONFIG['spring']['rest']['ADD_FUT_BOUT_SUMMARY'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT']), json = payload, headers = headers)
    response = r.json() 
    return response

def addMyBookieOdds(payload):
    headers = requests.utils.default_headers()
    headers['password'] = str(CONFIG['spring']['PW'])
    r = requests.post(url = CONFIG['spring']['rest']['ADD_MY_BOOKIE_ODDS'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT']), json = payload, headers = headers)
    response = r.json() 
    return response
                    
def getTrainingFights(year):
    r = requests.get(url = CONFIG['spring']['rest']['GET_FIGHTS_BY_YEAR'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], year))
    response = r.json()         
    return response['response']

def getAllBouts():
    r = requests.get(url = CONFIG['spring']['rest']['GET_ALL_BOUTS'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT']))
    response = r.json()         
    return response['response']

def getYearBouts():
    r = requests.get(url = CONFIG['spring']['rest']['GET_YEAR_BOUTS'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT']))
    response = r.json()         
    return response['response']

def getNewBouts():
    r = requests.get(url = CONFIG['spring']['rest']['GET_NEW_BOUTS'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT']))
    response = r.json()         
    return response['response']

def addBoutsToFight(fight_id):
    headers = requests.utils.default_headers()
    headers['password'] = str(CONFIG['spring']['PW'])
    r = requests.get(url = CONFIG['spring']['rest']['ADD_BOUTS_TO_FIGHT'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], fight_id), headers = headers)
    response = r.json()
    if (response['status'] != 'OK'):
        print("Add Bouts to Fight %s failed with %s" % (fight_id, response['errorMsg']))
        return False
    else:
        print("Add Bouts to Fight %s completed %s with %s bouts found and %s completed" % (fight_id, response['status'], response['itemsFound'], response['itemsCompleted']))
        return True
    
def addBoutsToFutureFight(fight_id):
    headers = requests.utils.default_headers()
    headers['password'] = str(CONFIG['spring']['PW'])
    r = requests.get(url = CONFIG['spring']['rest']['ADD_BOUTS_TO_FUTURE_FIGHT'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], fight_id), headers = headers)
    response = r.json()
    print("Add Bouts to Fight %s completed %s with %s bouts found and %s completed" % (fight_id, response['status'], response['itemsFound'], response['itemsCompleted']))

def getBoutsFromFight(fight_id):
    r = requests.get(url = CONFIG['spring']['rest']['GET_BOUTS_FROM_FIGHT'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], fight_id))
    response = r.json()

    if (response['status'] == 404):
        print("Get bouts from fight %s failed" % (fight_id))
    else:
        return response['response']
    
def futureFightUpdate():
    headers = requests.utils.default_headers()
    headers['password'] = str(CONFIG['spring']['PW'])
    r = requests.get(url = CONFIG['spring']['rest']['INIT_FUTURE_FIGHT_URL'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT']), headers = headers)
    response = r.json()
    if (response['errorMsg'] is not None):
        print(response['errorMsg'])
    
def initUpdate():
    r = requests.get(url = CONFIG['spring']['rest']['INIT_FIGHT_URL'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT']))
    r = requests.get(url = CONFIG['spring']['rest']['PARSE_YEAR_JUDGE_SCORE_URL'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], "2020"))

def addBoutScoreUrls(fight_oid):
    r = requests.get(url = CONFIG['spring']['rest']['ADD_BOUT_SCORE_URL'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], fight_oid))
    response = r.json()
    print("Add Bouts Score URLs to Fight %s completed with %s with %s bouts found and %s completed" % (fight_oid, response['status'], response['itemsFound'], response['itemsCompleted']))

def addBoutDetails(fight_id, bout_id):
    headers = requests.utils.default_headers()
    headers['password'] = str(CONFIG['spring']['PW'])
    r = requests.get(url = CONFIG['spring']['rest']['ADD_BOUTS_DETAILS'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], fight_id, bout_id),  headers = headers)
    response = r.json()
    if (response['status'] != 'OK'):
        print("Add Bouts Details to Bout %s failed" % (bout_id))
        return False
    else:
        print("Add Bouts Details to Bout %s completed with %s with %s fighters found and %s completed" % (bout_id, response['status'], response['itemsFound'], response['itemsCompleted']))
        return True

def scrapeBoutScores(bout_id):
    headers = requests.utils.default_headers()
    headers['password'] = str(CONFIG['spring']['PW'])
    r = requests.get(url = CONFIG['spring']['rest']['SCRAPE_BOUT_SCORE'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], bout_id), headers = headers)
    response = r.json()
    print("Add round scores to Bout %s completed wtih %s with %s rounds found and %s completed" % (bout_id, response['status'], response['itemsFound'], response['itemsCompleted']))

def refreshBout(bout_id):
    while True:
        try:
            r = requests.get(url = CONFIG['spring']['rest']['GET_BOUT'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], bout_id))
            response = r.json() 
            return response['response']
        except requests.exceptions.RequestException as err:
            print ("OOps: Something Else",err)
            pass
        except requests.exceptions.HTTPError as errh:
            print ("Http Error:",errh)
            pass
        except requests.exceptions.ConnectionError as errc:
            print ("Error Connecting:",errc)
            pass
        except requests.exceptions.Timeout as errt:
            print ("Timeout Error:",errt)  
            pass

def addFightExpectedOutcomes(fight_id):
    headers = requests.utils.default_headers()
    headers['password'] = str(CONFIG['spring']['PW'])
    r = requests.get(url = CONFIG['spring']['rest']['ADD_FIGHT_EXP_OUTCOME'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], fight_id), headers = headers)
    response = r.json() 
    print("Add bout expected outcomes to Fight %s completed wtih %s with %s rounds found and %s completed" % (fight_id, response['status'], response['itemsFound'], response['itemsCompleted']))

def addFightOdds(fight_id):
    headers = requests.utils.default_headers()
    headers['password'] = str(CONFIG['spring']['PW'])
    r = requests.get(url = CONFIG['spring']['rest']['ADD_FIGHT_ODDS'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], fight_id), headers = headers)
    response = r.json() 
    print("Add bout odds to Fight %s completed wtih %s with %s rounds found and %s completed" % (fight_id, response['status'], response['itemsFound'], response['itemsCompleted']))

def getAllFightIds():
    r = requests.get(url = CONFIG['spring']['rest']['GET_ALL_FIGHT_IDS'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT']))
    response = r.json()         
    return response['response']

def addFightOddsUrl(fight_details):
    print("Please provide the bestFightOdds url for %s (%s)" % (fight_details['fightName'], fight_details['fightDate']))
    fight_url = input()
    headers = requests.utils.default_headers()
    headers['password'] = str(CONFIG['spring']['PW'])
    r = requests.get(url = CONFIG['spring']['rest']['ADD_FIGHT_ODDS_URL'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], fight_details['fightId'], fight_url.replace("https://www.bestfightodds.com/events/", "")), headers = headers)
    response = r.json()
    print("Add bestFightOdds (%s) to fight %s completed with %s" % (fight_url, fight_details['fightName'], response['status']))
 
def updateRanking(payload):
    headers = requests.utils.default_headers()
    headers['password'] = str(CONFIG['spring']['PW'])
    r = requests.post(url = CONFIG['spring']['rest']['UPDATE_RANKING'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT']), json = payload, headers = headers)
    response = r.json() 
    return response    

def getRankings(weight_class):
    r = requests.get(url = CONFIG['spring']['rest']['GET_WC_RANKINGS'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], weight_class))
    response = r.json() 
    return response    

def getEloCount(fighterOid):
    r = requests.get(url = CONFIG['spring']['rest']['GET_ELO_COUNT'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], fighterOid))
    response = r.json() 
    if response['errorMsg'] is not None:
        print(response['errorMsg'])
    return response['response']    
    