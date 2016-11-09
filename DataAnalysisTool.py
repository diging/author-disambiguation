from __future__ import division

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re


from ast import literal_eval
from DistanceMetric import cosine_similarity, sentence_to_vector




class DataAnalysisTool:

    """
    This class has methods and tools to analyse a bunch of (World of Science)WOS papers objects.

    The paper details are read from a CSV. `CSV file headers are shown below.

    WOSID,DATE,TITLE,LASTNAME,FIRSTNAME,JOURNAL,EMAILADDRESS,PUBLISHER,SUBJECT,WC,AUTHOR_KEYWORDS.

    A CSV file of this format can be easily created by traversing a directory containing a
    bunch of WOS files and parsing each file using the famous Tethne tool.

    NOTE : To use this tool the CSV file should be readable from a disk location.

    """

    def __init__(self, inputDataSet):
        """
        Initialise the Data Analyzer object by providing the input csv file path.
        The input csv is assumed to be well formed and read into a pandas DataFrame object.
        :param inputDataSet:
        :return:
        """
        self.df = pd.read_csv(inputDataSet)

    def getPapersForlastNames(self, lastname):
        """
        Get all the DataFrame rows(paper objects) which have their lastname in the passed list of lastnames.
        :param lastname: A list of Author Last names
        :return: A pandas DataFrame.
        """
        return self.df[self.df['LASTNAME'].isin(lastname)]

    def getPapersForFirstNames(self, firstnames):
        """
        Get the DataFrame (all paper objects) which have their firstname in the passed list of firstnames
        :param firstnames: A list of Author First names
        :return: A pandas DataFrame.
        """
        return self.df[self.df['FIRSTNAME'].isin(firstnames)]

    def getPapersForAuthor(self, lastnames, firstnames):
        """
        Get the DataFrame(all paper objects) which have both their firstname AND lastname
        in the passed list of firstnames and lastnames.
        It would make sense to call this function, if you know all different variations of the firstnames and Lastnames
        for the SAME author.This way, we can retrieve all papers belonging to the author career.

        An example:
        Say, we want to retrieve all papers for the Author: DAVID ALBERTINI.

        Assuming that we know all the variations, we can achieve this in the following way.

        >>> lastnames = ['ALBERTINI', 'ALBERTIN', 'ALBERTINDF']
        >>> firstnames =['DAVID', 'DF', 'DAVID F', 'D F', 'D']

        >>> analyzer = DataAnalysisTool('/Users/aosingh/TethneDataAnalysis/Dataset/dataset.csv')
        >>> analyzer.getPapersForAuthor(lastnames, firstnames)

        :param lastnames: A list(read Variations) of Author lastnames
        :param firstnames:A list of Author firstnames
        :return: A pandas DataFrame.
        """
        return self.df[self.df['FIRSTNAME'].isin(firstnames) & self.df['LASTNAME'].isin(lastnames)]

    def drawHistogramForOverlapScore(self, papers_dataframe, feature_name):
        """
        This method can be used to plot a histogram to determine the overlap between every 2 papers in the passed
        DataFrame(i.e. List of Papers)

        The method also expects the feature_name to be passed. Overlap is determined only for that specific feature.


        A simple EXAMPLE to understand the score calculation:
        Say we have 2 papers: P1 and P2. And the feature_name passed is 'author_keywords'
        The Author_keywords for P1 are A1 = ['x', 'y']
        The Author_keywords for P2 are A2 = ['x', 'y', 'z']
        Overlap Score is len(A1 intersection A2)/len(A1 union A2)
        So, overlap-score for this example becomes 2/3 = 0.67

        NOW, we repeat the entire process for every pair of paper in the input parameter papers_dataframe.
        In short, what we have is a 2-D matrix where each element (i,j) is the overlap score between the
        ith and the jth paper

        From this 2-D matrix, we take the upper triangular matrix and convert it into a simple list of non-zero values.
        With the simple list, we plot the desired histogram.

        :param papers_dataframe:
        :param feature_name:
        :return:
        """
        score_matrix = np.zeros((len(papers_dataframe), len(papers_dataframe))) #initilise the score matrix with all zeros
        count = 0
        for index, rows in papers_dataframe.iterrows():
            count_child = 0
            feature_set = set()
            for x in literal_eval(rows[feature_name]):
                feature_set.add(x)
            for indexChild, rowChild in papers_dataframe.iterrows():
                    child_feature_set = set()
                    for x in literal_eval(rowChild[feature_name]):
                        child_feature_set.add(x)
                    cardinality_intersect = len(feature_set.intersection(child_feature_set))
                    cardinality_union = len(feature_set.union(child_feature_set))
                    if cardinality_union != 0:
                        score = cardinality_intersect/cardinality_union
                        score_matrix[count][count_child] = score
                    count_child += 1
            count += 1
        print score_matrix

        list_upper_triangular = score_matrix[np.triu_indices(len(papers_dataframe),1)]
        list_upper_triangular = list_upper_triangular[np.nonzero(list_upper_triangular)]
        #return list_upper_triangular
        pd.Series(list_upper_triangular).plot(kind='hist')
        plt.show(block=True)


    def drawhistrogramForCosineScore(self, listOfFeatures):
        """
        For a given list of (string)features, computes the cosine similarity scores between every pair of the items
        in the list.

        EXAMPLE : This method can be called called after the method ``getInstituteNamesForAuthor`` which returns
        the list of institute names.


        :param listOfFeatures:
        :return:
        """
        score_matrix = np.zeros((len(listOfFeatures), len(listOfFeatures)))
        count = 0
        for x in listOfFeatures:
            count_child = 0
            for y in listOfFeatures:
                vector1 = sentence_to_vector(x)
                vector2 = sentence_to_vector(y)
                score = cosine_similarity(vector1, vector2)
                score_matrix[count][count_child] = score
                count_child += 1
            count += 1
        print score_matrix
        list_upper_triangular = score_matrix[np.triu_indices(len(listOfFeatures),1)]
        list_upper_triangular = list_upper_triangular[np.nonzero(list_upper_triangular)]
        #return list_upper_triangular
        #pd.Series(list_upper_triangular).plot(kind='hist')
        sns.distplot(pd.Series(list_upper_triangular))
        plt.show(block=True)



    def getInstituteNamesForAuthor(self, authorNames, currentAuthorPapers):
        """

        :param authorNames:
        :param currentAuthorPapers:
        :return:
        """

        WORD = re.compile(r'\w+')
        institutes = []
        for index, row in currentAuthorPapers.iterrows():
            institute = row['INSTITUTE']
            if type(institute) is str:
                if institute.startswith('[') and institute.endswith(']'):
                    a = literal_eval(institute)
                    for entry in a:
                        m = re.search(r"\[(.*?)\]", entry)
                        if m is not None:
                            author_name = m.group(1)
                            tokens = set(x.lower() for x in WORD.findall(author_name))
                            intersection = set([x.lower() for x in authorNames]) & tokens
                            n = re.search(r"(?<=\]).*", entry)
                            if n is not None:
                                institute_name = n.group(0)
                            if len(intersection) > 0:
                                institutes.append(institute_name)
                elif not institute.startswith('[') and not institute.endswith(']'):
                    institutes.append(institute)
                    a = str(map(str,institute.split(',')))
        return institutes

    def getInstituteNamesForRandomRecords(self, randomPapers):
        institutes = []
        for index, row in randomPapers.iterrows():
            institute = row['INSTITUTE']
            if type(institute) is str:
                if institute.startswith('[') and institute.endswith(']'):
                    a = literal_eval(institute)
                    for entry in a:
                        n = re.search(r"(?<=\]).*", entry)
                        if n is not None:
                            institute_name = n.group(0)
                            institutes.append(institute_name)
                elif not institute.startswith('[') and not institute.endswith(']'):
                    institutes.append(institute)
        return institutes

