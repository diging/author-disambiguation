# Author-Disambiguation
This is a research project in which we address the problem of disambiguation in a co authorship network. Say, we are looking at all the papers by the same author. The same author may have slight variations in his first name or last name or both.

For example, the author David Albertini has the following variations in his name :
* D.F. Albertini
* David. F. Albertini
* D. Albertini
* AlbertniDF

Now, how do we say that 2 papers belong to the same author career? This is technically the problem of disambiguation. In literature some also call it as Record Linkage.

We took a machine learning approach to solve this problem. It turns out to be binary classification problem where 2 Author-Paper instances either match or don't match. (1 or 0).

### Clone this Repo
Clone this repository into whatever directory you'd like to work on it from:

```bash
git clone https://github.com/diging/author-disambiguation.git
```

### Install the following
*   [Python 2.7](https://www.python.org/download/releases/2.7/)
*   [Tethne](http://pythonhosted.org/tethne/)
    *   `pip install tethne`
*   [pandas](http://pandas.pydata.org/)
    *   `pip install pandas`
*   [scikit-learn](http://scikit-learn.org/stable/)
    *   `pip install -U scikit-learn`

### Files 
* `TrainingDataGenerator.py`
   *    This module is responsible for generating Training records.By training records, we mean the following 2 things.
    1  Records in the form of : AUTHOR_INSTANCE_1, AUTHOR_INSTANCE_2, MATCH(0,1)
        To be precise, following are the column names

        [FIRST_NAME1, FIRST_NAME2, LAST_NAME1, LAST_NAME2,
        EMAILADDRESS1, EMAILADDRESS2, INSTITUTE1, INSTITUTE2,
        AUTHOR_KW1, AUTHOR_KW2, COAUTHORS1, COAUTHORS2, MATCH]

        We call the corresponding CSV as train.csv

    2  Scores in between the 2 AUTHOR INSTANCES.
        Each column is a score in between 0 and 1. We will train our classifiers on these records. They have the following features
        
        ['INSTIT_SCORE','BOTH_NAME_SCORE','FNAME_SCORE',
        'FNAME_PARTIAL_SCORE','LNAME_SCORE','LNAME_PARTIAL_SCORE',
        'EMAIL_ADDR_SCORE','AUTH_KW_SCORE','COAUTHOR_SCORE','MATCH']

        This CSV is called scores.csv

        Please read https://diging.atlassian.net/wiki/display/DILS/Training+the+classifier for more details


