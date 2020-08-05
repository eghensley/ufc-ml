#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:11:07 2020

@author: ehens86
"""

import json

with open("./src/common/raw_feature_universe.json", "r") as f:
    raw_features = json.load(f)
    print(raw_features)