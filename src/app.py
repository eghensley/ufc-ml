#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:50:47 2020

@author: eric.hensleyibm.com
"""

from flask import Flask, request, jsonify
from _ufc_ import ufc_engine
from spring.config import CONFIG
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

failed_login_response = {'status': 'Forbidden', 'errorMsg': 'Login Failed'}


@app.route('/ufc/api/v1.0/populate/past/<fightId>', methods=['GET'])
def addPastFightBouts(fightId):
    if not engine.authenticate(request.headers):
        return jsonify(failed_login_response), 404
    else:
        response = engine.populate_past_fight(fightId)
        return jsonify(response), response['statusCode']

@app.route('/ufc/api/v1.0/populate/future/<fightId>', methods=['GET'])
def addFutureFightBouts(fightId):
    if not engine.authenticate(request.headers):
        return jsonify(failed_login_response), 404
    else:
        response = engine.populate_future_fight(fightId)
        return jsonify(response), response['statusCode']

@app.route('/ufc/api/v1.0/rankings/<weightClass>', methods=['GET'])
def getWeightClassRanks(weightClass):
    response = engine.get_ranking_for_wc(weightClass)
    return jsonify(response), response['statusCode']

@app.route('/ufc/api/v1.0/rankings/top', methods=['GET'])
def getTopWeightClassRanks():
    response = engine.get_top_wc_ranks()
    return jsonify(response), response['statusCode']

@app.route('/ufc/api/v1.0/rankings/<weightClass>/fighter/<fighterOid>', methods=['GET'])
def getWeightClassFighterRank(weightClass, fighterOid):
    response = engine.get_ranking_for_wc_fighter(weightClass, fighterOid)
    return jsonify(response), response['statusCode']

@app.route('/ufc/api/v1.0/populate/ml/<boutId>', methods=['GET'])
def addMlOddsToBout(boutId):
    if not engine.authenticate(request.headers):
        return jsonify(failed_login_response), 404
    else:
        response = engine.addMlProb(boutId)
        return jsonify(response), response['statusCode']
    
@app.route('/ufc/api/v1.0/populate/future', methods=['GET'])
def initFuture():
    if not engine.authenticate(request.headers):
        return jsonify(failed_login_response), 404
    else:
        response = engine.popFutureBouts()
        return jsonify(response), response['statusCode']
    
@app.route('/ufc/api/v1.0/explain/bout/<boutId>', methods=['GET'])
def getBoutExplainer(boutId):
    response = engine.gen_win_pred_explainer(boutId)
    return jsonify(response), response['statusCode']
    
if __name__ == '__main__':
    print('initializing')
    engine = ufc_engine(CONFIG['spring']['PW'])
    print('app initialized')
    app.run(port=int(CONFIG['flask']['PORT']), host='0.0.0.0')
