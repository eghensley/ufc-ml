#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 15:23:51 2020

@author: ehens86
"""
#def warn(*args, **kwargs):
#    pass
#import warnings
#warnings.warn = warn
import time
import warnings
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns; sns.set(style="ticks", color_codes=True)
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from itertools import cycle, islice
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegressionCV

random_state = 170

import sys, os
if __name__ == "__main__":
    sys.path.append("..")
    
from data.generate import pull_year_raw_training_data


data = pd.DataFrame()
for year in range(2005, 2020):
    year_data = pull_year_raw_training_data(year, score = 'finish', add_gender = True, add_class = True)
    data = data.append(year_data)
    
list(data)

data = data[['offBodySigStrikeAccuracy', 'offBodySigStrikeAttempted',
 'offBodySigStrikeSuccessful', 'offClinchSigStrikeAccuracy',
 'offClinchSigStrikeAttempted', 'offClinchSigStrikeSuccessful',
 'offDistanceSigStrikeAccuracy', 'offDistanceSigStrikeAttempted',
 'offDistanceSigStrikeSuccessful', 'offGroundSigStrikeAccuracy',
 'offGroundSigStrikeAttempted', 'offGroundSigStrikeSuccessful',
 'offHeadSigStrikeAccuracy', 'offHeadSigStrikeAttempted',
 'offHeadSigStrikeSuccessful', 'offKnockdowns',
 'offLegSigStrikeAccuracy', 'offLegSigStrikeAttempted',
 'offLegSigStrikeSuccessful', 'offTotStrikeAccuracy',
 'offTotStrikeAttempted', 'offTotStrikeSuccessful',
 'weight', 'gender', 'offBodySigStrikeSuccessfulShare',
 'offBodySigStrikeAttemptedShare', 'offClinchSigStrikeSuccessfulShare',
 'offClinchSigStrikeAttemptedShare', 'offDistanceSigStrikeSuccessfulShare',
 'offDistanceSigStrikeAttemptedShare', 'offGroundSigStrikeSuccessfulShare',
 'offGroundSigStrikeAttemptedShare', 'offHeadSigStrikeSuccessfulShare',
 'offHeadSigStrikeAttemptedShare', 'offLegSigStrikeSuccessfulShare',
 'offLegSigStrikeAttemptedShare', 'offTotStrikeSuccessfulShare',
 'offTotStrikeAttemptedShare', 'offKnockdownsShare','round',  'offTkoKo'
 ]]
data.to_csv("finish_2005-2019.csv")

columns = ['offBodySigStrikeAccuracy', 'offBodySigStrikeAttempted',
 'offBodySigStrikeSuccessful', 'offClinchSigStrikeAccuracy',
 'offClinchSigStrikeAttempted', 'offClinchSigStrikeSuccessful',
 'offDistanceSigStrikeAccuracy', 'offDistanceSigStrikeAttempted',
 'offDistanceSigStrikeSuccessful', 'offGroundSigStrikeAccuracy',
 'offGroundSigStrikeAttempted', 'offGroundSigStrikeSuccessful',
 'offHeadSigStrikeAccuracy', 'offHeadSigStrikeAttempted',
 'offHeadSigStrikeSuccessful', 'offKnockdowns',
 'offLegSigStrikeAccuracy', 'offLegSigStrikeAttempted',
 'offLegSigStrikeSuccessful', 'offTotStrikeAccuracy',
 'offTotStrikeAttempted', 'offTotStrikeSuccessful',
 'weight', 'offBodySigStrikeSuccessfulShare',
 'offBodySigStrikeAttemptedShare', 'offClinchSigStrikeSuccessfulShare',
 'offClinchSigStrikeAttemptedShare', 'offDistanceSigStrikeSuccessfulShare',
 'offDistanceSigStrikeAttemptedShare', 'offGroundSigStrikeSuccessfulShare',
 'offGroundSigStrikeAttemptedShare', 'offHeadSigStrikeSuccessfulShare',
 'offHeadSigStrikeAttemptedShare', 'offLegSigStrikeSuccessfulShare',
 'offLegSigStrikeAttemptedShare', 'offTotStrikeSuccessfulShare',
 'offTotStrikeAttemptedShare', 'offKnockdownsShare',
 'offTkoKo' 
 ]

data = data[data['gender'] == 'M']
data = data[data['round'] == 1]
data = data[columns]

#data.to_csv("finish_2005-2019.csv")

X = data[[i for i in columns if i != 'offTkoKo']]
y = data['offTkoKo']



#model = LogisticRegressionCV(cv=5, random_state=0, class_weight=None, solver = 'liblinear', scoring = 'neg_log_loss')
#model.fit(X, y)
#model.get_params()
#param_scores = model.scores_[1]
#param_scores.mean(axis = 0)
#param_values = model.Cs_
#
#tuned_model = LogisticRegression(class_weight=None, solver = 'liblinear', C = 1.66810054e+02)
#rfecv_tuned_log_loss = RFECV(estimator=tuned_model, step=1, cv=StratifiedKFold(6),
#                      scoring='neg_log_loss')
#rfecv_tuned_log_loss.fit(X, y)       
#print("Optimal number of features : %d (%.4f)" % (rfecv_tuned_log_loss.n_features_, rfecv_tuned_log_loss.grid_scores_[rfecv_tuned_log_loss.n_features_-1])) 
#print('Best features :', X.columns[rfecv_tuned_log_loss.support_])
#
#plt.figure()
#plt.xlabel("Number of features selected")
#plt.ylabel("Cross validation score (nb of correct classifications)")
#plt.plot(range(1, len(rfecv_tuned_log_loss.grid_scores_) + 1), rfecv_tuned_log_loss.grid_scores_)
#plt.show()
#
#rfecv_tuned_f1 = RFECV(estimator=tuned_model, step=1, cv=StratifiedKFold(6),
#                      scoring='f1')
#rfecv_tuned_f1.fit(X, y)     
#print("Optimal number of features : %d (%.4f)" % (rfecv_tuned_f1.n_features_, rfecv_tuned_f1.grid_scores_[rfecv_tuned_f1.n_features_-1]))   
#plt.figure()
#plt.xlabel("Number of features selected")
#plt.ylabel("Cross validation score (nb of correct classifications)")
#plt.plot(range(1, len(rfecv_tuned_f1.grid_scores_) + 1), rfecv_tuned_f1.grid_scores_)
#plt.show()
#
#rfecv_tuned_roc = RFECV(estimator=tuned_model, step=1, cv=StratifiedKFold(6),
#                      scoring='roc_auc')
#rfecv_tuned_roc.fit(X, y)      
#print("Optimal number of features : %d (%.4f)" % (rfecv_tuned_roc.n_features_, rfecv_tuned_roc.grid_scores_[rfecv_tuned_roc.n_features_-1])) 
#plt.figure()
#plt.xlabel("Number of features selected")
#plt.ylabel("Cross validation score (nb of correct classifications)")
#plt.plot(range(1, len(rfecv_tuned_roc.grid_scores_) + 1), rfecv_tuned_roc.grid_scores_)
#plt.show()



solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag']#, 'saga']
weights = ['balanced']

print("~~~~~~~~~~~ logloss ~~~~~~~~~~~")
scores = {}
n = 0
for solver in solvers:
    for weight in weights:
        print("Solver = %s, weight = %s" % (solver, weight))
        n += 1
        # Create the RFE object and compute a cross-validated score.
        reg = LogisticRegressionCV(cv=6, random_state=0, class_weight=weight, solver = solver, max_iter = 5000)
        #reg = LogisticRegression(class_weight=None, solver = 'saga')
        # The "accuracy" scoring is proportional to the number of correct
        # classifications
        rfecv_tuned_log_loss = RFECV(estimator=reg, step=1, cv=StratifiedKFold(6),
                      scoring='neg_log_loss')
        rfecv_tuned_log_loss.fit(X, y)
        print("Optimal number of features : %d (%.4f)" % (rfecv_tuned_log_loss.n_features_, rfecv_tuned_log_loss.grid_scores_[rfecv_tuned_log_loss.n_features_-1]))
        print('Best features :', X.columns[rfecv_tuned_log_loss.support_])   
        scores[n] = {'solver': solver, 'weight': weight, 'score': rfecv_tuned_log_loss.grid_scores_[rfecv_tuned_log_loss.n_features_-1]}
        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv_tuned_log_loss.grid_scores_) + 1), rfecv_tuned_log_loss.grid_scores_)
        plt.show()
        
#scores = {1: {'solver': 'newton-cg',
#  'weight': 'balanced',
#  'score': -0.2345554691624514},
# 2: {'solver': 'newton-cg', 'weight': None, 'score': -0.12406901995518554},
# 3: {'solver': 'lbfgs', 'weight': 'balanced', 'score': -0.23823102939442678},
# 4: {'solver': 'lbfgs', 'weight': None, 'score': -0.12379590506611238},
# 5: {'solver': 'liblinear',
#  'weight': 'balanced',
#  'score': -0.23349377798314977},
# 6: {'solver': 'liblinear', 'weight': None, 'score': -0.12434342203609168},
# 7: {'solver': 'sag', 'weight': 'balanced', 'score': -0.24046762631765836},
# 8: {'solver': 'sag', 'weight': None, 'score': -0.12492563959291743},
# 9: {'solver': 'saga', 'weight': 'balanced', 'score': -0.24063265874439413},
# 10: {'solver': 'saga', 'weight': None, 'score': -0.1251681304230677}}
scores_df = pd.DataFrame.from_dict(scores).T

print("~~~~~~~~~~~ rocauc ~~~~~~~~~~~")
roc_scores = {}
n = 0
for solver in solvers:
    for weight in weights:
        print("Solver = %s, weight = %s" % (solver, weight))
        n += 1
        # Create the RFE object and compute a cross-validated score.
        reg = LogisticRegressionCV(cv=6, random_state=0, class_weight=weight, solver = solver, max_iter = 5000)
        #reg = LogisticRegression(class_weight=None, solver = 'saga')
        # The "accuracy" scoring is proportional to the number of correct
        # classifications
        rfecv_tuned_roc = RFECV(estimator=reg, step=1, cv=StratifiedKFold(6),
                      scoring='roc_auc')
        rfecv_tuned_roc.fit(X, y)
        print("Optimal number of features : %d (%.4f)" % (rfecv_tuned_roc.n_features_, rfecv_tuned_roc.grid_scores_[rfecv_tuned_roc.n_features_-1]))
        print('Best features :', X.columns[rfecv_tuned_roc.support_])   
        roc_scores[n] = {'solver': solver, 'weight': weight, 'score': rfecv_tuned_roc.grid_scores_[rfecv_tuned_roc.n_features_-1]}
        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv_tuned_roc.grid_scores_) + 1), rfecv_tuned_roc.grid_scores_)
        plt.show()     
    
#roc_scores={1: {'solver': 'newton-cg', 'weight': 'balanced', 'score': 0.9681854669385586},
# 2: {'solver': 'newton-cg', 'weight': None, 'score': 0.9682832952361627},
# 3: {'solver': 'lbfgs', 'weight': 'balanced', 'score': 0.9672179287446555},
# 4: {'solver': 'lbfgs', 'weight': None, 'score': 0.9690288867079827},
# 5: {'solver': 'liblinear', 'weight': 'balanced', 'score': 0.9683536389698094},
# 6: {'solver': 'liblinear', 'weight': None, 'score': 0.9692887647188503},
# 7: {'solver': 'sag', 'weight': 'balanced', 'score': 0.9663827765699269},
# 8: {'solver': 'sag', 'weight': None, 'score': 0.9676217029385783},
# 9: {'solver': 'saga', 'weight': 'balanced', 'score': 0.9660558187369735},
# 10: {'solver': 'saga', 'weight': None, 'score': 0.9661239274087464}}
roc_scores_df = pd.DataFrame.from_dict(roc_scores).T

print("~~~~~~~~~~~ f1 ~~~~~~~~~~~")
f1_scores = {}
n = 0
for solver in solvers:
    for weight in weights:
        print("Solver = %s, weight = %s" % (solver, weight))
        n += 1
        # Create the RFE object and compute a cross-validated score.
        reg = LogisticRegressionCV(cv=6, random_state=0, class_weight=weight, solver = solver, max_iter = 5000)
        #reg = LogisticRegression(class_weight=None, solver = 'saga')
        # The "accuracy" scoring is proportional to the number of correct
        # classifications
        rfecv_tuned_f1 = RFECV(estimator=reg, step=1, cv=StratifiedKFold(6),
                      scoring='f1')
        rfecv_tuned_f1.fit(X, y)
        print("Optimal number of features : %d (%.4f)" % (rfecv_tuned_f1.n_features_, rfecv_tuned_f1.grid_scores_[rfecv_tuned_f1.n_features_-1]))
        print('Best features :', X.columns[rfecv_tuned_f1.support_])   
        f1_scores[n] = {'solver': solver, 'weight': weight, 'score': rfecv_tuned_f1.grid_scores_[rfecv_tuned_f1.n_features_-1]}
        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv_tuned_f1.grid_scores_) + 1), rfecv_tuned_f1.grid_scores_)
        plt.show()     
        
#f1_scores = {1: {'solver': 'newton-cg', 'weight': 'balanced', 'score': 0.6790100509690579},
# 2: {'solver': 'newton-cg', 'weight': None, 'score': 0.7071632408441024},
# 3: {'solver': 'lbfgs', 'weight': 'balanced', 'score': 0.6781767253801284},
# 4: {'solver': 'lbfgs', 'weight': None, 'score': 0.7038958118061075},
# 5: {'solver': 'liblinear', 'weight': 'balanced', 'score': 0.6819255155058564},
# 6: {'solver': 'liblinear', 'weight': None, 'score': 0.7077896535580258},
# 7: {'solver': 'sag', 'weight': 'balanced', 'score': 0.6796477366645592},
# 8: {'solver': 'sag', 'weight': None, 'score': 0.6948914802351789},
# 9: {'solver': 'saga', 'weight': 'balanced', 'score': 0.6778325882258299},
# 10: {'solver': 'saga', 'weight': None, 'score': 0.6965037109727249}}
f1_scores_df = pd.DataFrame.from_dict(f1_scores).T
#g = sns.pairplot(data, hue="score", kind="reg", diag_kind="kde", corner = True)
#g.savefig("output.png")
#
#data.to_csv("2015_ko_data.csv")


#### liblinear, no weight

model = LogisticRegressionCV(cv=10, random_state=0, class_weight='balanced', solver = 'newton-cg', scoring = 'neg_log_loss', max_iter = 5000)
model.fit(X[['offBodySigStrikeAccuracy', 'offBodySigStrikeAttempted',
       'offBodySigStrikeSuccessful', 'offClinchSigStrikeAccuracy',
       'offClinchSigStrikeAttempted', 'offClinchSigStrikeSuccessful',
       'offDistanceSigStrikeAccuracy', 'offDistanceSigStrikeAttempted',
       'offDistanceSigStrikeSuccessful', 'offGroundSigStrikeAccuracy',
       'offGroundSigStrikeAttempted', 'offGroundSigStrikeSuccessful',
       'offHeadSigStrikeAccuracy', 'offHeadSigStrikeAttempted',
       'offHeadSigStrikeSuccessful', 'offKnockdowns',
       'offLegSigStrikeAccuracy', 'offLegSigStrikeAttempted',
       'offLegSigStrikeSuccessful', 'offTotStrikeAccuracy',
       'offTotStrikeAttempted', 'offTotStrikeSuccessful', 'weight',
       'offBodySigStrikeSuccessfulShare', 'offBodySigStrikeAttemptedShare',
       'offClinchSigStrikeSuccessfulShare', 'offClinchSigStrikeAttemptedShare',
       'offDistanceSigStrikeSuccessfulShare',
       'offDistanceSigStrikeAttemptedShare',
       'offGroundSigStrikeSuccessfulShare', 'offGroundSigStrikeAttemptedShare',
       'offHeadSigStrikeSuccessfulShare', 'offHeadSigStrikeAttemptedShare',
       'offLegSigStrikeSuccessfulShare', 'offLegSigStrikeAttemptedShare',
       'offTotStrikeSuccessfulShare', 'offTotStrikeAttemptedShare',
       'offKnockdownsShare']], y)
model.get_params()
param_scores = model.scores_[1]
param_scores.mean(axis = 0)
param_values = model.Cs_






cols = ['BodySigStrike', 'ClinchSigStrike', 'DistanceSigStrike', 'GroundSigStrike', 'HeadSigStrike', 'LegSigStrike', 'TotStrike', 'Takedown']

for col in cols:
    data['off'+col+'Successful'] = data['off'+col+'Successful']/data['seconds']
    





list(data)