# Author-Disambiguation
This is a research project in which we address the problem of disambiguation in a co authorship network. Say, we are looking at all the papers by the same author then they may have different names. 
For example, the author David Albertini has the following possible variations in his name :
1. D.F. Albertini
2. David. F. Albertini
3. D. Albertini
4. AlbertniDF
Now, how do we say that 2 papers belong to the same author career? This is the problem of Disambiguation. Some also call it as Record Linkage.
We took a machine learning approach to solve this problem. This problem turns out to be binary classification problem where 2 Author-Paper instances either match or don't match. (1 or 0).

