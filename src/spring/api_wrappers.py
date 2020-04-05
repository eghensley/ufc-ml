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

def getBoutData(boutOid):
    r = requests.get(url = CONFIG['spring']['rest']['GET_BOUT_DATA_BY_OID'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], boutOid))
    response = r.json()
    return response['response']

def addRoundScore(oid, round_dict, fighter_name):
    r = requests.post(url = CONFIG['spring']['rest']['ADD_ROUND_SCORE'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], oid), json = round_dict)
    response = r.json() 
    if response['status'] != 'OK':
        print("Save for %s round %s failed with.. %s" % (fighter_name, round_dict['round'], response['errorMsg']))
                    
def getTrainingFights(year):
    r = requests.get(url = CONFIG['spring']['rest']['GET_FIGHTS_BY_YEAR'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], year))
    response = r.json()         
    return response['response']

def addBoutsToFight(fight_id):
    r = requests.get(url = CONFIG['spring']['rest']['ADD_BOUTS_TO_FIGHT'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], fight_id))
    response = r.json()
    print("Add Bouts to Fight %s completed %s with %s bouts found and %s completed" % (fight_id, response['status'], response['itemsFound'], response['itemsCompleted']))

def getBoutsFromFight(fight_id):
    r = requests.get(url = CONFIG['spring']['rest']['GET_BOUTS_FROM_FIGHT'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], fight_id))
    response = r.json()

    if (response['status'] == 404):
        print("Get bouts from fight %s failed" % (fight_id))
    else:
        return response['response']
    
def initUpdate():
    r = requests.get(url = CONFIG['spring']['rest']['INIT_FIGHT_URL'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT']))
    r = requests.get(url = CONFIG['spring']['rest']['PARSE_YEAR_JUDGE_SCORE_URL'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], "2020"))

def addBoutScoreUrls(fight_oid):
    r = requests.get(url = CONFIG['spring']['rest']['ADD_BOUT_SCORE_URL'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], fight_oid))
    response = r.json()
    print("Add Bouts Score URLs to Fight %s completed with %s with %s bouts found and %s completed" % (fight_oid, response['status'], response['itemsFound'], response['itemsCompleted']))

def addBoutDetails(fight_id, bout_id):
    r = requests.get(url = CONFIG['spring']['rest']['ADD_BOUTS_DETAILS'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], fight_id, bout_id))
    response = r.json()
    if (response['status'] != 'OK'):
        print("Add Bouts Details to Bout %s failed" % (bout_id))
        return False
    else:
        print("Add Bouts Details to Bout %s completed with %s with %s fighters found and %s completed" % (bout_id, response['status'], response['itemsFound'], response['itemsCompleted']))
        return True

def scrapeBoutScores(bout_id):
    r = requests.get(url = CONFIG['spring']['rest']['SCRAPE_BOUT_SCORE'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], bout_id))
    response = r.json()
    print("Add round scores to Bout %s completed wtih %s with %s rounds found and %s completed" % (bout_id, response['status'], response['itemsFound'], response['itemsCompleted']))

def refreshBout(bout_id):
    r = requests.get(url = CONFIG['spring']['rest']['GET_BOUT'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], bout_id))
    response = r.json() 
    return response['response']

def addFightExpectedOutcomes(fight_id):
    r = requests.get(url = CONFIG['spring']['rest']['ADD_FIGHT_EXP_OUTCOME'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], fight_id))
    response = r.json() 
    print("Add bout expected outcomes to Fight %s completed wtih %s with %s rounds found and %s completed" % (fight_id, response['status'], response['itemsFound'], response['itemsCompleted']))

def addFightOdds(fight_id):
    r = requests.get(url = CONFIG['spring']['rest']['ADD_FIGHT_ODDS'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], fight_id))
    response = r.json() 
    print("Add bout odds to Fight %s completed wtih %s with %s rounds found and %s completed" % (fight_id, response['status'], response['itemsFound'], response['itemsCompleted']))

def getAllFightIds():
    r = requests.get(url = CONFIG['spring']['rest']['GET_ALL_FIGHT_IDS'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT']))
    response = r.json()         
    return response['response']

def addFightOddsUrl(fight_details):
    print("Please provide the bestFightOdds url for %s (%s)" % (fight_details['fightName'], fight_details['fightDate']))
    fight_url = input()
    r = requests.get(url = CONFIG['spring']['rest']['ADD_FIGHT_ODDS_URL'] % (CONFIG['spring']['HOST'], CONFIG['spring']['PORT'], fight_details['fightId'], fight_url.replace("https://www.bestfightodds.com/events/", "")))
    response = r.json()
    print("Add bestFightOdds (%s) to fight %s completed with %s" % (fight_url, fight_details['fightName'], response['status']))
 