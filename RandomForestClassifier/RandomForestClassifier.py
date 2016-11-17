from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, ShuffleSplit

import pandas as pd
from pprint import pprint

data_set = '../Training_data/scores.csv'

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
clf = RandomForestClassifier()
clf.fit(df[features], df['MATCH'])

pprint(pd.crosstab(clf.predict(df[features]), df['MATCH']))
