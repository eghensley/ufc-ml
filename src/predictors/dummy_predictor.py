#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:55:58 2020

@author: ehens86
"""

import numpy as np

def predictWinner(bout):
    return bout['fighterBoutXRefs'][np.random.choice([0,1])]['fighter']['fighterId']