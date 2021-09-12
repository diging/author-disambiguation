from __future__ import division
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, ShuffleSplit
import math
import pickle
from pprint import pprint
import pandas as pd
import os
from pprint import pprint

data_set = '../Training_data/scores.csv'
output_filename = "random_forest_CV.pkl"
output_directory = "/content/AuthorDisambiguation/Serialized_models"

output_file = os.path.join(output_directory, output_filename)

features = [
    'INSTIT_SCORE',
    'BOTH_NAME_SCORE',
    'FNAME_SCORE',
    'FNAME_PARTIAL_SCORE',
    'LNAME_SCORE',
    'LNAME_PARTIAL_SCORE',
    'EMAIL_ADDR_SCORE',
    'AUTH_KW_SCORE',
    'COAUTHOR_SCORE'
]

df = pd.read_csv(data_set)
samplesize = math.floor((1/2)*len(df))
print (samplesize)
sample_df = df.sample(samplesize)
clf = RandomForestClassifier(verbose=1)

scores = cross_val_score(clf, df[features], df['MATCH'], cv=10, scoring='f1_macro')

pprint(scores)
pprint("K-FOLD Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

with open("/content/author-disambiguation/Serialized_models/random_forest_CV.pkl", 'r') as output:
    clf2 = pickle.loads(output.read())

scores = cross_val_score(clf2, df[features], df['MATCH'], cv=10, scoring='f1_macro')
pprint(scores)
pprint("K-FOLD Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

