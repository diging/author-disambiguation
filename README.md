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




