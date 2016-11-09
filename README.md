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
This module is responsible for generating Training records. By training records, we mean the following 2 things.

    * Generate records in the form of : AUTHOR_INSTANCE_1, AUTHOR_INSTANCE_2, MATCH(0,1). The corresponding output file is called `train.csv`
    
        To be precise, following are the column names in the file `train.csv`
        ```python 
                   ['FIRST_NAME1', 
                   'FIRST_NAME2', 
                   'LAST_NAME1', 
                   'LAST_NAME2',
                   'EMAILADDRESS1', 
                   'EMAILADDRESS2', 
                   'INSTITUTE1', 
                   'INSTITUTE2',
                   'AUTHOR_KW1', 
                   'AUTHOR_KW2', 
                   'COAUTHORS1', 
                   'COAUTHORS2', 
                   'MATCH']
        ```
    * Scores in between the 2 AUTHOR INSTANCES.Each column is a score in between 0 and 1. 
        
        We will train our classifiers on these records. The corresponding output file is called `scores.csv`
    
        To be precise, following are the column names in the file `scores.csv`
        ```python 
                    ['INSTIT_SCORE',
                    'BOTH_NAME_SCORE',
                    'FNAME_SCORE',
                    'FNAME_PARTIAL_SCORE',
                    'LNAME_SCORE',
                    'LNAME_PARTIAL_SCORE',
                    'EMAIL_ADDR_SCORE',
                    'AUTH_KW_SCORE',
                    'COAUTHOR_SCORE',
                    'MATCH']
        ```

* `DistanceMetric.py`
    * We can define various similarity metrics in this module. As of now, I have defined the method to calculate cosine similarity
        ```python
        input1 = "CARNEGIE INST WASHINGTON,DEPT EMBRYOL"
        input2 = "CARNEGIE INST WASHINGTON,DEPT"
        vector1 = sentence_to_vector(input1) #Counter({'EMBRYOL': 1, 'WASHINGTON': 1, 'INST': 1, 'CARNEGIE': 1, 'DEPT': 1})
        vector2 = sentence_to_vector(input2) #Counter({'WASHINGTON': 1, 'INST': 1, 'CARNEGIE': 1, 'DEPT': 1})
        cosine_similarity(vector1, vector2)  #0.894427191
        ```
