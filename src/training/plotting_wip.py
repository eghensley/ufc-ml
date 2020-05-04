#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:30:42 2020

@author: ehens86
"""
import time
import warnings
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from itertools import cycle, islice
import pandas as pd
from sklearn.decomposition import PCA
random_state = 170

[ 'round',
 'weightClass',
  'fighterOid'
]
import sys, os
if __name__ == "__main__":
    sys.path.append("..")
    
from data.generate import generate_raw_round_data

if not os.path.exists("../data/raw_round_data.csv"):
    generate_raw_round_data()

data = pd.read_csv("../data/raw_round_data.csv")
data.set_index("Unnamed: 0", inplace = True)
data.rename(columns = {'defBodySigStrikeAttemped':'defBodySigStrikeAttempted',
                       'offBodySigStrikeAttemped':'offBodySigStrikeAttempted',
                       'defClinchSigStrikeAttemped':'defClinchSigStrikeAttempted',
                       'offClinchSigStrikeAttemped':'offClinchSigStrikeAttempted',
                       'offDistanceSigStrikeAttemped':'offDistanceSigStrikeAttempted',
                       'defDistanceSigStrikeAttemped':'defDistanceSigStrikeAttempted',
                       'offGroundSigStrikeAttemped':'offGroundSigStrikeAttempted',
                       'defGroundSigStrikeAttemped':'defGroundSigStrikeAttempted',
                       'offHeadSigStrikeAttemped':'offHeadSigStrikeAttempted',
                       'defHeadSigStrikeAttemped':'defHeadSigStrikeAttempted',
                       'offLegSigStrikeAttemped':'offLegSigStrikeAttempted',
                       'defLegSigStrikeAttemped':'defLegSigStrikeAttempted'                       
                       }, inplace = True) 

list(data)


data['weightClass'].unique()


    
#{'WW': 1, 'LW': 2, 'FFW': 3, 'WFFW': 3, 'MW': 4, 'HW': 5, 'FW': 6, 'WFW': 6, 'BW': 7, 'WBW': 7, 'CW': 8, 'LHW': 9}


principle_components = PCA(n_components=2).fit_transform(data[strike_feats])
plt.scatter([i[0] for i in principle_components], [i[1] for i in principle_components], s = 1)

export_df = pd.DataFrame(principle_components)
export_df.set_index(data.index, inplace = True)
export_df['gender'] = data['gender']
export_df['class'] = data['class']
export_df['weightClass'] = data['weightClass']
export_df.rename({0: 'PCA_1', 1: 'PCA_2'}, inplace = True)

export_df.to_csv('round_gender_class_cluster.csv')





from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e5,class_weight="balanced")
X = np.array(export_df[[0,1]])
Y = export_df['gender']
# Create an instance of Logistic Regression Classifier and fit the data.
logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()






list(export_df)
fig, ax = plt.subplots()
for color in ['tab:blue', 'tab:orange', 'tab:green']:
    n = 750
    x, y = np.random.rand(2, n)
    scale = 200.0 * np.random.rand(n)
    ax.scatter(x, y, c=color, s=scale, label=color,
               alpha=0.3, edgecolors='none')

ax.legend()
ax.grid(True)

plt.show()



tech_feats = []
for tech_col in ['TakedownSuccessful', 'PassSuccessful', 'SubmissionAttempted', 'ReversalSuccessful']:
    tech_feats.append('off'+tech_col+'PerSec')    
    data['off'+tech_col+'PerSec'] = data['off'+tech_col]/data['seconds']
    
scaled_inputs = MinMaxScaler().fit_transform(data[strike_feats])
principle_components = PCA(n_components=2).fit_transform(scaled_inputs)
plt.scatter([i[0] for i in principle_components], [i[1] for i in principle_components], s = 1)









strike_feats = []
for strike_col in ['BodySig', 'ClinchSig', 'DistanceSig', 'GroundSig', 'HeadSig', 'LegSig', 'Tot']:
    strike_feats.append('off'+strike_col+'StrikeAccuracy')
    strike_feats.append('off'+strike_col+'StrikeSuccessfulPerSec')    
    data['off'+strike_col+'StrikeSuccessfulPerSec'] = data['off'+strike_col+'StrikeSuccessful']/data['seconds']
    
    
    
    
    data['off'+strike_col+'StrikeSuccessful'] / (data['off'+strike_col+'StrikeSuccessful'] + data['def'+strike_col+'StrikeSuccessful'])
tech_feats = []
for tech_col in ['TakedownSuccessful', 'PassSuccessful', 'SubmissionAttempted', 'ReversalSuccessful']:
    tech_feats.append('off'+tech_col+'PerSec')    
    data['off'+tech_col+'PerSec'] = data['off'+tech_col]/data['seconds']
tech_feats.append('offTakedownAccuracy')
data['offTakedownAccuracy'] = data['offTakedownSuccessful']/data['offTakedownAttempted']









offense_feats = strike_feats + tech_feats

offense_data = data[offense_feats]
offense_data.fillna(0, inplace = True)

scaled_inputs = MinMaxScaler().fit_transform(offense_data)
principle_components = PCA(n_components=3).fit_transform(scaled_inputs)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([i[0] for i in principle_components], [i[1] for i in principle_components], [i[2] for i in principle_components], s = 1)


scaled_inputs = RobustScaler().fit_transform(offense_data)
principle_components = PCA(n_components=2).fit_transform(scaled_inputs)
plt.scatter([i[0] for i in principle_components], [i[1] for i in principle_components], s = 1)





inputs = data[['defBodySigStrikeAccuracy', 'defBodySigStrikeAttemped', 'defBodySigStrikeSuccessful', 'defClinchSigStrikeAccuracy',
 'defClinchSigStrikeAttemped', 'defClinchSigStrikeSuccessful', 'defDistanceSigStrikeAccuracy', 'defDistanceSigStrikeAttemped',
 'defDistanceSigStrikeSuccessful', 'defGroundSigStrikeAccuracy', 'defGroundSigStrikeAttemped', 'defGroundSigStrikeSuccessful',
 'defHeadSigStrikeAccuracy', 'defHeadSigStrikeAttemped', 'defHeadSigStrikeSuccessful', 'defKnockdowns',
 'defLegSigStrikeAccuracy', 'defLegSigStrikeAttemped', 'defLegSigStrikeSuccessful', 'defPassSuccessful',
 'defReversalSuccessful', 'defSubmissionAttempted', 'defSubmissionSuccessful', 'defTakedownAttempted',
 'defTakedownSuccessful', 'defTkoKo', 'defTotStrikeAccuracy', 'defTotStrikeAttempted',
 'defTotStrikeSuccessful', 'offBodySigStrikeAccuracy', 'offBodySigStrikeAttemped', 'offBodySigStrikeSuccessful',
 'offClinchSigStrikeAccuracy', 'offClinchSigStrikeAttemped', 'offClinchSigStrikeSuccessful', 'offDistanceSigStrikeAccuracy',
 'offDistanceSigStrikeAttemped', 'offDistanceSigStrikeSuccessful', 'offGroundSigStrikeAccuracy', 'offGroundSigStrikeAttemped',
 'offGroundSigStrikeSuccessful', 'offHeadSigStrikeAccuracy', 'offHeadSigStrikeAttemped', 'offHeadSigStrikeSuccessful',
 'offKnockdowns', 'offLegSigStrikeAccuracy', 'offLegSigStrikeAttemped', 'offLegSigStrikeSuccessful',
 'offPassSuccessful', 'offReversalSuccessful', 'offSubmissionAttempted', 'offSubmissionSuccessful',
 'offTakedownAttempted', 'offTakedownSuccessful', 'offTkoKo', 'offTotStrikeAccuracy', 'offTotStrikeAttempted', 'offTotStrikeSuccessful']]


scaled_inputs = RobustScaler().fit_transform(inputs)
principle_components = PCA(n_components=2).fit_transform(scaled_inputs)


plt.scatter([i[0] for i in principle_components], [i[1] for i in principle_components], s = 1)


['defBodySigStrikeAccuracy', 'defBodySigStrikeAttemped', 'defBodySigStrikeSuccessful', 'defClinchSigStrikeAccuracy',
 'defClinchSigStrikeAttemped', 'defClinchSigStrikeSuccessful', 'defDistanceSigStrikeAccuracy', 'defDistanceSigStrikeAttemped',
 'defDistanceSigStrikeSuccessful', 'defGroundSigStrikeAccuracy', 'defGroundSigStrikeAttemped', 'defGroundSigStrikeSuccessful',
 'defHeadSigStrikeAccuracy', 'defHeadSigStrikeAttemped', 'defHeadSigStrikeSuccessful', 'defKnockdowns',
 'defLegSigStrikeAccuracy', 'defLegSigStrikeAttemped', 'defLegSigStrikeSuccessful', 'defPassSuccessful',
 'defReversalSuccessful', 'defSubmissionAttempted', 'defSubmissionSuccessful', 'defTakedownAttempted',
 'defTakedownSuccessful', 'defTkoKo', 'defTotStrikeAccuracy', 'defTotStrikeAttempted',
 'defTotStrikeSuccessful', 'offBodySigStrikeAccuracy', 'offBodySigStrikeAttemped', 'offBodySigStrikeSuccessful',
 'offClinchSigStrikeAccuracy', 'offClinchSigStrikeAttemped', 'offClinchSigStrikeSuccessful', 'offDistanceSigStrikeAccuracy',
 'offDistanceSigStrikeAttemped', 'offDistanceSigStrikeSuccessful', 'offGroundSigStrikeAccuracy', 'offGroundSigStrikeAttemped',
 'offGroundSigStrikeSuccessful', 'offHeadSigStrikeAccuracy', 'offHeadSigStrikeAttemped', 'offHeadSigStrikeSuccessful',
 'offKnockdowns', 'offLegSigStrikeAccuracy', 'offLegSigStrikeAttemped', 'offLegSigStrikeSuccessful',
 'offPassSuccessful', 'offReversalSuccessful', 'offSubmissionAttempted', 'offSubmissionSuccessful',
 'offTakedownAttempted', 'offTakedownSuccessful', 'offTkoKo', 'offTotStrikeAccuracy', 'offTotStrikeAttempted', 'offTotStrikeSuccessful']

