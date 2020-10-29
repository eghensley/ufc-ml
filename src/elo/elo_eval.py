#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:48:14 2020

@author: ehens86
"""

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    
import numpy as np
import json
from os import listdir
from os.path import isfile, join
import pandas as pd
from scipy.stats import percentileofscore

def trend(vals):
    x = np.arange(0,len(vals))
    y=np.array(vals)
    z = np.polyfit(x,y,1)
    print("{0}x + {1}".format(*z))
    return(z[0])
    
def gen_score_report(domain):
#    domain = 'strike'
    all_scores = {}
        
    onlyfiles = [f for f in listdir("src/elo/scores/%s" % (domain)) if isfile(join("src/elo/scores/%s" % (domain), f))]
    for file in onlyfiles:
        with open('src/elo/scores/%s/%s'% (domain, file)) as f:
            data = json.load(f)
        for feat in [i for i in data.keys() if i.find('Elo') > -1]:
            if len(data[feat]) == 0:
                data.pop(feat)
                continue
            
#            if '%s-trend' % (feat) not in all_scores.keys():
#                all_scores['%s-trend' % (feat)] = []
            if '%s-mse' % (feat) not in all_scores.keys():
                all_scores['%s-mse' % (feat)] = []                
            if '%s-std' % (feat) not in all_scores.keys():
                all_scores['%s-std' % (feat)] = []                      
                
            data[feat] = [i*10 for i in data[feat]]
#            all_scores['%s-trend'%(feat)].append(trend(data[feat]) * -10000)
            
            mean = np.mean([(i)**2 for i in data[feat]])
            std = sum([((x - mean) ** 2) for x in data[feat]]) / len(data[feat]) ** 2
            data.pop(feat)
            all_scores['%s-mse' % (feat)].append(mean * -1)
            all_scores['%s-std' % (feat)].append(std * -1)
        
    ranked_scores = {}
    for file in onlyfiles:
        with open('src/elo/scores/%s/%s'% (domain, file)) as f:
            data = json.load(f)
        
        idx = file.replace('.json', '')
        score = {}
#        score['id'] = idx
        for feat in [i for i in data.keys() if i.find('Elo') > -1]:
            if len(data[feat]) == 0:
                data.pop(feat)
                continue
            data[feat] = [i*10 for i in data[feat]]
#            tnd = trend(data[feat]) * -10000
            mean = np.mean([(i)**2 for i in data[feat]]) * -1
            std = (sum([((x - mean) ** 2) for x in data[feat]]) / len(data[feat]) ** 2) * -1

#            score['%s-trend' % (feat)] = percentileofscore(all_scores['%s-trend' % (feat)], tnd, 'rank')    
            score['%s-std' % (feat)] = percentileofscore(all_scores['%s-std' % (feat)], std, 'rank') / 2  
            score['%s-mse' % (feat)] = percentileofscore(all_scores['%s-mse' % (feat)], mean, 'rank')   

        ranked_scores[idx] = score
    
    ranked_score_df = pd.DataFrame.from_dict(ranked_scores).T
    ranked_score_df['tot'] = ranked_score_df.sum(axis = 1)
    ranked_score_df.sort_values(by=['tot'], ascending = False, inplace = True)

    best_model_id = ranked_score_df.index[0]
    
    return best_model_id
    
def gen_config():
    models = {}
    for dom in ['strike', 'grapp', 'ko', 'sub']:
        models[dom] = gen_score_report(dom)
       
    params = {}
    for ft in models.keys():
        with open('src/elo/models/%s/%s.json' % (ft, models[ft])) as f:
            param = json.load(f)
        if ft == 'strike':
            params['default_off_strike'] = param['defaults']['offStrike']
            params['default_def_strike'] = param['defaults']['defStrike']
            params['strike_damper'] = param['dampers']['Strike']
        elif ft == 'grapp':
            params['default_off_grapp'] = param['defaults']['offGrappling']
            params['default_def_grapp'] = param['defaults']['defGrappling']
            params['grapp_damper'] = param['dampers']['Grappling']
        elif ft == 'ko':
            params['default_off_ko'] = param['defaults']['powerStrike']
            params['default_def_ko'] = param['defaults']['chinStrike']
            params['ko_finish_damper'] = param['dampers']['powerStrike']
        elif ft == 'sub':
            params['default_off_sub'] = param['defaults']['subGrappling']
            params['sub_finish_damper'] = param['defaults']['evasGrappling']
            params['default_def_sub'] = param['dampers']['subGrappling']
    
    with open('src/predictors/elo/elo-config.json', 'w') as r:
        json.dump(params, r)

