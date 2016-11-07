from sklearn import  svm
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
clf = svm.SVC(kernel='poly', degree=2, verbose=True)
clf.fit(df[features], df['MATCH'])
pprint(pd.crosstab(clf.predict(df[features]), df['MATCH']))


