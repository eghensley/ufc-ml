#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:50:44 2020

@author: ehens86
"""

def convImpPercToAmericanOdds(impPerc):
    if impPerc == 50:
        return 100
    elif impPerc > 50:
        return (-1*(impPerc/(100-impPerc)))*100
    else:
        return ((100 - impPerc)/impPerc)*100
    
def calcWinnings(wager, impPerc):
#    wager, odds = WAGER, bout['fighterBoutXRefs'][1]['mlOdds']
    odds = convImpPercToAmericanOdds(impPerc)
    if (odds > 0):
        return wager * (odds/100)
    else:
        return -1*(wager*100)/odds 
    
