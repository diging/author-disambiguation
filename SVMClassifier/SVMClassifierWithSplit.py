import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
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

X_train, X_test, y_train, y_test = train_test_split(df[features], df['MATCH'], test_size=0.4, random_state=0)
clf = svm.SVC(kernel='rbf')
clf = svm.SVC(kernel='poly', degree=1, verbose=True)

clf.fit(X_train, y_train)
pprint(clf.score(X_test, y_test))