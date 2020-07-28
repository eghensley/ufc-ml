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
    
if __name__ == '__main__':
    engine = ufc_engine(CONFIG['spring']['PW'])
    app.run(port=CONFIG['flask']['PORT'])
