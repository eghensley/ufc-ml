#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 17:52:05 2020

@author: ehens86
"""

from spring.api_wrappers import initUpdate, futureFightUpdate, getTrainingFights, addBoutsToFutureFight, getBoutsFromFight, addFutBoutData, addMyBookieOdds
from db.parse_fights import addInfoToAllBouts, addFightOddsUrl, evalIfMissingFightOddsInfo, addFightOdds
from predictors import insert_new_ml_scores, insert_new_ml_prob
from utils.general import convAmericanOddsToImpPerc

#addFightOddsUrls('53278852bcd91e11')

#delete from ufc2.fighter_bout_xref fbx2 where fbx2.oid in (select fbx.oid from ufc2.bout b join ufc2.fighter_bout_xref fbx on b.oid = fbx.bout_oid join ufc2.fighter f on f.oid = fbx.fighter_oid where b.fight_oid = 'f350febb-5ff9-4c85-875c-d39fcd853143' order by b.oid)
#delete from ufc2.bout b2 where b2.oid in (select b.oid from ufc2.bout b where fight_oid = 'f350febb-5ff9-4c85-875c-d39fcd853143')

#ALTER TABLE ufc2.bfo_expected_outcome DISABLE TRIGGER ALL;
#ALTER TABLE ufc2.fighter_bout_xref DISABLE TRIGGER ALL;
#
#delete from ufc2.bfo_expected_outcome beo where beo.fighter_bout_oid  in (select fbx.oid from ufc2.bout b join ufc2.fighter_bout_xref fbx on b.oid = fbx.bout_oid join ufc2.fighter f on f.oid = fbx.fighter_oid where b.fight_oid = '9d682860-b23d-44d3-877b-92eb5cffe97e' order by b.oid);
#delete from ufc2.fighter_bout_xref fbx2 where fbx2.oid in (select fbx.oid from ufc2.bout b join ufc2.fighter_bout_xref fbx on b.oid = fbx.bout_oid join ufc2.fighter f on f.oid = fbx.fighter_oid where b.fight_oid = '9d682860-b23d-44d3-877b-92eb5cffe97e' order by b.oid);
#
#ALTER TABLE ufc2.bfo_expected_outcome ENABLE TRIGGER ALL;
#ALTER TABLE ufc2.fighter_bout_xref ENABLE TRIGGER ALL;
#
#delete from ufc2.bout b2 where b2.oid in (select b.oid from ufc2.bout b where fight_oid = '9d682860-b23d-44d3-877b-92eb5cffe97e');
#

#addInfoToAllBouts("53278852bcd91e11")
#
#   fight_id = 'ddbd0d6259ce57cc'
def add_new_bouts(fight_id):
    addBoutsToFutureFight(fight_id)
    fight_details = getBoutsFromFight(fight_id)
    for bout in fight_details['bouts']:
        if bout['schedRounds'] is None:
            champ_bout = None
            num_rounds = None
            print("Required fields missing from %s vs %s" % (bout['fighterBoutXRefs'][0]['fighter']['fighterName'], bout['fighterBoutXRefs'][1]['fighter']['fighterName']))
            num_rounds_raw = input("Scheduled Rounds: ")
            champ_bout_raw = input("Championship Fight (Y/N): ")
            
            num_rounds = int(num_rounds_raw.strip())
            if champ_bout_raw.upper().strip() == 'Y':
                champ_bout = True
            else:
                champ_bout = False
                
            payload = {'oid': bout['oid'], 'schedRounds': num_rounds, 'champBout': champ_bout}
            resp = addFutBoutData(payload)

    if fight_details['bestFightOddsUrl'] is None:
        addFightOddsUrl(fight_details)
    fight_details_refreshed = getBoutsFromFight(fight_id)
    odds_completion = evalIfMissingFightOddsInfo(fight_details_refreshed)
    if odds_completion['odds']:
        addFightOdds(fight_id)
        update_mybookie(fight_id)
    fight_details_refresh = getBoutsFromFight(fight_id)
    for bout_refresh in fight_details_refresh['bouts']:
        insert_new_ml_prob(bout_refresh['boutId'])
#    insert_new_ml_scores(bout_id)
#    insert_new_ml_prob(bout_id)

def update_mybookie(fight_id):
    fight_details = getBoutsFromFight(fight_id)
    for bout in fight_details['bouts']:
#        if bout['oid'] != 'ecb7e38b-0315-4d0d-a734-53f6167ae6bb':
#            continue
#        bout = fight_details['bouts'][-4]
        if bout['gender'] != 'MALE':
            continue
        fighter_1_odds_raw = None
        fighter_2_odds_raw = None
        
        fighter_1_odds_int = None
        fighter_2_odds_int = None
        
        fighter_1_odds = None
        fighter_2_odds = None
        
        print('Please add MyBookie odds for %s vs %s' % (bout['fighterBoutXRefs'][0]['fighter']['fighterName'], bout['fighterBoutXRefs'][1]['fighter']['fighterName']))
        fighter_1_odds_raw = input('%s odds:   ' % (bout['fighterBoutXRefs'][0]['fighter']['fighterName']))
        fighter_2_odds_raw = input('%s odds:   ' % (bout['fighterBoutXRefs'][1]['fighter']['fighterName']))

        fighter_1_odds_int = int(fighter_1_odds_raw)
        fighter_2_odds_int = int(fighter_2_odds_raw)
                
        fighter_1_odds = convAmericanOddsToImpPerc(fighter_1_odds_int) * 100
        fighter_2_odds = convAmericanOddsToImpPerc(fighter_2_odds_int) * 100
                
        payload_1 = {'oid': bout['fighterBoutXRefs'][0]['oid'], 'mlOdds': fighter_1_odds}
        addMyBookieOdds(payload_1)
        
        payload_2 = {'oid': bout['fighterBoutXRefs'][1]['oid'], 'mlOdds': fighter_2_odds}
        addMyBookieOdds(payload_2)
        
#    year = 2020
def pop_year_bouts(year):
    fights = getTrainingFights(year)
    for fight in fights:
        addInfoToAllBouts(fight)
    
def pop_future_bouts():
    futureFightUpdate()
    initUpdate()
#    pop_year_bouts(2020)