#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:25:48 2020

@author: ehens86
"""

import numpy as np
import matplotlib.pyplot as plt
from data import pull_ml_training_corpus
import lightgbm as lgb
from utils.general import reduce_mem_usage
import pandas as pd

df = pull_ml_training_corpus()
df = reduce_mem_usage(df)
mod = lgb.LGBMClassifier(random_state = 1108, n_estimators = 500, verbose=-1, is_unbalance = True)

X = df[[
 'defBodySigStrikeAccuracy',
 'defBodySigStrikeAttempted',
 'defBodySigStrikeSuccessful',
 'defClinchSigStrikeAccuracy',
 'defClinchSigStrikeAttempted',
 'defClinchSigStrikeSuccessful',
 'defControlTime',
 'defDistanceSigStrikeAccuracy',
 'defDistanceSigStrikeAttempted',
 'defDistanceSigStrikeSuccessful',
 'defGroundSigStrikeAccuracy',
 'defGroundSigStrikeAttempted',
 'defGroundSigStrikeSuccessful',
 'defHeadSigStrikeAccuracy',
 'defHeadSigStrikeAttempted',
 'defHeadSigStrikeSuccessful',
 'defKnockdowns',
 'defLegSigStrikeAccuracy',
 'defLegSigStrikeAttempted',
 'defLegSigStrikeSuccessful',
 'defReversalSuccessful',
 'defSubmissionAttempted',
 'defTakedownAccuracy',
 'defTakedownAttempted',
 'defTakedownSuccessful',
 'defTotStrikeAccuracy',
 'defTotStrikeAttempted',
 'defTotStrikeSuccessful',
 'offBodySigStrikeAccuracy',
 'offBodySigStrikeAttempted',
 'offBodySigStrikeSuccessful',
 'offClinchSigStrikeAccuracy',
 'offClinchSigStrikeAttempted',
 'offClinchSigStrikeSuccessful',
 'offControlTime',
 'offDistanceSigStrikeAccuracy',
 'offDistanceSigStrikeAttempted',
 'offDistanceSigStrikeSuccessful',
 'offGroundSigStrikeAccuracy',
 'offGroundSigStrikeAttempted',
 'offGroundSigStrikeSuccessful',
 'offHeadSigStrikeAccuracy',
 'offHeadSigStrikeAttempted',
 'offHeadSigStrikeSuccessful',
 'offKnockdowns',
 'offLegSigStrikeAccuracy',
 'offLegSigStrikeAttempted',
 'offLegSigStrikeSuccessful',
 'offReversalSuccessful',
 'offSubmissionAttempted',
 'offTakedownAccuracy',
 'offTakedownAttempted',
 'offTakedownSuccessful',
 'offTotStrikeAccuracy',
 'offTotStrikeAttempted',
 'offTotStrikeSuccessful',
 'round',
 'weight',
 'gender',
 'class',
 'offBodySigStrikeSuccessfulShare',
 'offBodySigStrikeAttemptedShare',
 'offClinchSigStrikeSuccessfulShare',
 'offClinchSigStrikeAttemptedShare',
 'offDistanceSigStrikeSuccessfulShare',
 'offDistanceSigStrikeAttemptedShare',
 'offGroundSigStrikeSuccessfulShare',
 'offGroundSigStrikeAttemptedShare',
 'offHeadSigStrikeSuccessfulShare',
 'offHeadSigStrikeAttemptedShare',
 'offLegSigStrikeSuccessfulShare',
 'offLegSigStrikeAttemptedShare',
 'offTotStrikeSuccessfulShare',
 'offTotStrikeAttemptedShare',
 'offTakedownSuccessfulShare',
 'offTakedownAttemptedShare',
 'offKnockdownsShare',
 'offControlTimeShare',
 'offReversalSuccessfulShare',
 'offSubmissionAttemptedShare']]

X = pd.get_dummies(X, columns=['gender'])
X = pd.get_dummies(X, columns=['class'])
X = pd.get_dummies(X, columns=['round'])

mod.fit(X, df['offTkoKo'])
importances = mod.feature_importances_
indices = np.argsort(importances)[::-1]


# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, list(X)[indices[f]], importances[indices[f]]))

# Plot the impurity-based feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

