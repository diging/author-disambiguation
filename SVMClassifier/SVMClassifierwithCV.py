import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, ShuffleSplit

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

print ("Using K-FOLD CROSS VALIDATION...")
df = pd.read_csv(data_set)
clf = svm.SVC(kernel='poly', degree=2, verbose=True)
scores = cross_val_score(clf, df[features], df['MATCH'], cv=10, scoring='f1_macro')
pprint(scores)
pprint("K-FOLD Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



print ("Using Shuffle Split...")
cv = ShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
scores = cross_val_score(clf, df[features], df['MATCH'], cv=cv)
pprint("ShuffleSplit Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
