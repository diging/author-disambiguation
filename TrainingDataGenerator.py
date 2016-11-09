from __future__ import division

import pandas as pd
import re
import os

from ast import literal_eval
from DistanceMetric import cosine_similarity, sentence_to_vector
from fuzzywuzzy import fuzz
from DataAnalysisTool import DataAnalysisTool


input_dataset = '/Users/aosingh/AuthorDisambiguation/Dataset'

split_institute = lambda institute:institute.split(',')
join_institute_names = lambda institute : " ".join(x for x in institute)

features = ['INSTIT_SCORE',
        'BOTH_NAME_SCORE',
        'FNAME_SCORE',
        'FNAME_PARTIAL_SCORE',
        'LNAME_SCORE',
        'LNAME_PARTIAL_SCORE',
        'EMAIL_ADDR_SCORE',
        'AUTH_KW_SCORE',
        'COAUTHOR_SCORE',
        'MATCH']

#Possible variations in author names.
ALBERITNI_LASTNAME = ['ALBERTINI', 'ALBERTIN', 'ALBERTINDF']
ALBERTINI_FIRSTNAME = ['DAVID', 'DF', 'DAVID F', 'D F', 'D']

ARNOLD_LASTNAME = ['ARNOLD']
ARNOLD_FIRSTNAME = ['JM', '']

BOYER_LASTNAME = ['BOYER']
BOYER_FIRSTNAME = ['BC', '']

BRANDIFF_LASTNAME = ['BRANDRIFF', 'BRANDRIF']
BRANDIFF_FIRSTNAME = ['B', 'BF', '']


DAWID_LASTNAME = ['DAWID']
DAWID_FIRSTNAME = ['IGOR B', 'IB', 'I B', '']


TRISHCMAN_LASTNAME = ['TRISCHMANN']
TRISHCMAN_FIRSTNAME = ['TM', '']

CALVET_LASTNAME = ['CALVET']
CALVET_FIRSTNAME = ['JP', 'JAMES P', 'J', 'J P']

HONJO_LASTNAME = ['HONJO']
HONJO_FIRSTNAME = ['T', 'TASUKU', 'TSUNEO']

FRANKEL_LASTNAME = ['FRANKEL']
FRANKEL_FIRSTNAME = ['FR', 'FRED R']

fileLastName = {}
fileFirstName = {}

#Declare a dictionary to map each (author specific)csv file corresponding to the possible firstnames and lastnames.
#These dictionaries will be used when we call the generate() method to generate the training data.

fileLastName['Albertini_David.csv'] = ALBERITNI_LASTNAME
fileLastName['Arnold_John.csv'] = ARNOLD_LASTNAME
fileLastName['Boyer_Barbara.csv'] = BOYER_LASTNAME
fileLastName['Brigitte_Brandiff.csv'] = BRANDIFF_LASTNAME
fileLastName['Dawid_Igor.csv'] = DAWID_LASTNAME
fileLastName['Trischmann_Thomas.csv'] = TRISHCMAN_LASTNAME
fileLastName['Calvet_James.csv'] = CALVET_LASTNAME
fileLastName['Honjo_Tasuka_part1.csv'] = HONJO_LASTNAME
fileLastName['Honjo_Tasuka_part2.csv'] = HONJO_LASTNAME
fileLastName['Frankel_Fred.csv'] = FRANKEL_LASTNAME

fileFirstName['Albertini_David.csv'] = ALBERTINI_FIRSTNAME
fileFirstName['Arnold_John.csv'] = ARNOLD_FIRSTNAME
fileFirstName['Boyer_Barbara.csv'] = BOYER_FIRSTNAME
fileFirstName['Brigitte_Brandiff.csv'] = BRANDIFF_FIRSTNAME
fileFirstName['Dawid_Igor.csv'] = DAWID_FIRSTNAME
fileFirstName['Trischmann_Thomas.csv'] = TRISHCMAN_FIRSTNAME
fileFirstName['Calvet_James.csv'] = CALVET_FIRSTNAME
fileFirstName['Honjo_Tasuka_part1.csv'] = HONJO_FIRSTNAME
fileFirstName['Honjo_Tasuka_part2.csv'] = HONJO_FIRSTNAME
fileFirstName['Frankel_Fred.csv'] = FRANKEL_FIRSTNAME


class TrainingDataGenerator:
    """
    This module is responsible for generating Training records.
    By training records, we mean the following 2 things.

    1. Records in the form of : AUTHOR_INSTANCE_1, AUTHOR_INSTANCE_2, MATCH(0,1)
        To be precise, following are the column names:

        [FIRST_NAME1, FIRST_NAME2, LAST_NAME1, LAST_NAME2,
        EMAILADDRESS1, EMAILADDRESS2, INSTITUTE1, INSTITUTE2,
        AUTHOR_KW1, AUTHOR_KW2, COAUTHORS1, COAUTHORS2, MATCH]

        We call the corresponding CSV as train.csv


    2. Scores in between the 2 AUTHOR INSTANCES.
        Each column is a score in between 0 and 1. We will train our classifiers on these records.
        They have the following features:

        ['INSTIT_SCORE','BOTH_NAME_SCORE','FNAME_SCORE',
        'FNAME_PARTIAL_SCORE','LNAME_SCORE','LNAME_PARTIAL_SCORE',
        'EMAIL_ADDR_SCORE','AUTH_KW_SCORE','COAUTHOR_SCORE','MATCH']

        This CSV is called scores.csv

        Please read https://diging.atlassian.net/wiki/display/DILS/Training+the+classifier for more details

    """

    def __init__(self, papers_df, random=True):
        """
        We need the papers(pandas dataframe), to iterate on and generate the scores.
        An example shown below, will make the usage of this class clear.

        ``Example``

        >>> fileName = '/Users/aosingh/AuthorDisambiguation/Dataset/Albertini_David.csv'
        >>> analyzer = DataAnalysisTool(fileName) # Please check the class DataAnalysisTool for more details
        >>> ALBERTINI_FIRSTNAME = ['DAVID', 'DF', 'DAVID F', 'D F', 'D']
        >>> ALBERITNI_LASTNAME = ['ALBERTINI', 'ALBERTIN', 'ALBERTINDF']
        >>> papers = analyzer.getPapersForAuthor(ALBERITNI_LASTNAME, ALBERTINI_FIRSTNAME)
        >>> training_data_generator = TrainingDataGenerator(papers, random=False)
        >>> training_data_generator.generate_records() # generate train.csv.
        >>> training_data_generator.calculate_scores() # Generate scores.csv

        :param papers_df: Papers DataFrame
        :param random: If random then 'MATCH' = 0, else 'MATCH' = 1
        :return:

        *TODO* - rename attribute 'random' to 'match'

        """
        self.papers_df = papers_df
        self.random = random
        self.training_df = None
        self.training_scores_df = None

    @staticmethod
    def set_feature_value(paper_sample, training_record, attribute_name, column_name):
        """

        ``Example`` The following example explains the usage.
        >>> TrainingDataGenerator.set_feature_value(row, d, 'FIRST_NAME1', 'FIRSTNAME')
            This means, do the following
        >>> d['FIRST_NAME1'] = row['FIRSTNAME']

        :param paper_sample:
        :param training_record:
        :param attribute_name:
        :param column_name:
        :return: training_record after setting the feature value.
        """
        training_record[attribute_name] = paper_sample[column_name]
        return training_record

    @staticmethod
    def get_score_for_coauthors(training_record):
        """
        Given a training_record, calculate the overlap score in between Co-authors
        The overlap score is calculated in between the fields COAUTHORS1, COAUTHORS2

                >>>intersection = COAUTHORS1 & COAUTHORS2
                >>>union = COAUTHORS1 | COAUTHORS2
                >>>score = len(intersection)/len(union)

        Please read :https://diging.atlassian.net/wiki/display/DILS/Co-authors+for+disambiguation for more details

        :param training_record:
        :return: A score between 0 and 1
        """
        coauthor1 = training_record['COAUTHORS1']
        coauthor2 = training_record['COAUTHORS2']
        if coauthor1 is None or coauthor1 == "[]":
            return 0
        if coauthor2 is None or coauthor2 == "[]":
            return 0
        coauthor2 = set(literal_eval(coauthor2))
        coauthor1 = set(literal_eval(coauthor1))

        kw_intersection = coauthor1 & coauthor2
        kw_union = coauthor1 | coauthor2
        if len(kw_union) > 0:
            return len(kw_intersection)/len(kw_union)
        return 0

    @staticmethod
    def get_score_for_author_keywords(training_record):
        """
        Given a training_record, calculate the overlap score in between Author Keywords
        The overlap score is calculated in between the fields AUTHOR_KW1, AUTHOR_KW2

                >>>intersection = AUTHOR_KW1 & AUTHOR_KW2
                >>>union = AUTHOR_KW1 | AUTHOR_KW2
                >>>score = len(intersection)/len(union)

        Please read : https://diging.atlassian.net/wiki/display/DILS/Author+keywords+for+dismbiguation for more details

        :param training_record:
        :return: A score between 0 and 1
        """
        author_kw1 = training_record['AUTHOR_KW1']
        author_kw2 = training_record['AUTHOR_KW2']
        if author_kw1 is None or author_kw1 == "[]":
            return 0
        if author_kw2 is None or author_kw2 == "[]":
            return 0
        author_kw2 = set(literal_eval(author_kw2))
        author_kw1 = set(literal_eval(author_kw1))

        kw_intersection = author_kw1 & author_kw2
        kw_union = author_kw1 | author_kw2
        if len(kw_union) > 0:
            return len(kw_intersection)/len(kw_union)
        return 0

    @staticmethod
    def get_score_for_email_address(training_record):
        """
        Return 1 if both the email-addresses match else return 0.

        Please read : https://diging.atlassian.net/wiki/pages/viewpage.action?pageId=46432257 for more details

        :param training_record:
        :return: A score of either 0 or 1
        """

        email1 = training_record['EMAILADDRESS1']
        email2 = training_record['EMAILADDRESS2']
        if email1 is None or email1 == "[]":
            return 0
        if email2 is None or email2 == "[]":
            return 0

        if email2.startswith('[') and email1.startswith('['):
            list1 = set(literal_eval(email1))
            list2 = set(literal_eval(email2))
            intersection = list1 & list2
            union = list1 | list2
            if len(union) > 0:
                return len(intersection)/len(union)
        if email2.startswith('[') and not email1.startswith('['):
            if email1 in set(literal_eval(email2)):
                return 1
        if email1.startswith('[') and not email2.startswith('['):
            if email2 in set(literal_eval(email1)):
                return 1
        if email1 == email2:
            return 1
        return 0

    @staticmethod
    def get_score_for_name(training_record):
        return 1 if training_record['FIRST_NAME1'] == training_record['FIRST_NAME2'] \
                    and training_record['LAST_NAME1'] == training_record['LAST_NAME2'] \
               else 0


    @staticmethod
    def get_institute_name(institutions, author):
        """
        This method finds the institute name to which the author belongs.

        If we look at a training record and specifically the 'INSTITUTE' field, We have 3 different cases here
        CASE 1. Institute name is a String, For example :
                        "Univ Kansas, Med Ctr, Kansas City, KS 66160 USA."

        CASE 2. Institute name is a List, For example:
                        [u'MARINE BIOL LAB,WOODS HOLE,MA.', u'UNIV MASSACHUSETTS,AMHERST,MA.',
                        u'REED COLL,PORTLAND,OR.', u'UNIV CONNECTICUT,BIOL SCI GRP,STORRS,CT 06268.']

        CASE 3. Institutions name is a map, where each author is mapped to his/her institute. For example:
                        [u'[Telfer, Evelyn E.] Univ Edinburgh, Inst Cell Biol, Edinburgh, Midlothian, Scotland.',
                        u'[Telfer, Evelyn E.] Univ Edinburgh, Ctr Integrat Physiol, Edinburgh, Midlothian, Scotland.',
                        u'[Albertini, David F.] Univ Kansas, Med Ctr, Inst Reprod Hlth & Regenerat Med, Ctr Reprod Sci,
                                                                                            Kansas City, KS 66103 USA.']

        We deal with the above scenarios in the following way.

        For CASE 1 : Return the institute name as-is.

        For CASE 2 : No way to link author to its institute. So return ``None``

        For CASE 3 : We try to find the correct mapping. If there is match, we return the found institute name

        :param institutions:
        :param author:
        :return:
        """
        WORD = re.compile(r'\w+')
        if type(institutions) is str:
            if institutions.startswith('[') and institutions.endswith(']'):
                    a = literal_eval(institutions)
                    for entry in a:
                        m = re.search(r"\[(.*?)\]", entry)
                        if m is not None:
                            author_name = m.group(1)
                            tokens = set(x.lower() for x in WORD.findall(author_name))
                            intersection = set([x.lower() for x in author]) & tokens
                            n = re.search(r"(?<=\]).*", entry)
                            if n is not None:
                                institute_name = n.group(0)
                            if len(intersection) > 0:
                                return institute_name
            elif not institutions.startswith('[') and not institutions.endswith(']'):
                    return institutions

    @staticmethod
    def compare_institute_names(training_record):
        '''
        For the training record, passed in input.
        1. GET INSTITUTE 1 using the method get_institute_name()
        2. GET INSTITUTE 2
        3. Return a cosine similarity score in between the 2 institute names.

        Please read https://diging.atlassian.net/wiki/pages/viewpage.action?pageId=46432257 for more details.

        :param training_record:
        :return: A score between 0 and 1.
        '''
        institute1 = TrainingDataGenerator.get_institute_name(training_record['INSTITUTE1'], training_record['LAST_NAME1'])
        institute2 = TrainingDataGenerator.get_institute_name(training_record['INSTITUTE2'], training_record['LAST_NAME2'])
        if institute1 is not None and institute2 is not None:
            institute1 = join_institute_names(split_institute(institute1)[0:3])
            institute2 = join_institute_names(split_institute(institute2)[0:3])
            return cosine_similarity(sentence_to_vector(institute1), sentence_to_vector(institute2))
        return 0

    def calculate_scores(self):
        """
        Calculate the scores for each feature defined below.

        >>> features = ['INSTIT_SCORE','BOTH_NAME_SCORE','FNAME_SCORE','FNAME_PARTIAL_SCORE','LNAME_SCORE',
        >>> 'LNAME_PARTIAL_SCORE','EMAIL_ADDR_SCORE','AUTH_KW_SCORE','COAUTHOR_SCORE','MATCH']
        
        :return:
        """
        self.training_df['INSTIT_SCORE'] = self.training_df.apply(lambda row: TrainingDataGenerator.compare_institute_names(row), axis=1)
        self.training_df['BOTH_NAME_SCORE'] = self.training_df.apply(lambda row: TrainingDataGenerator.get_score_for_name(row), axis=1)
        self.training_df['FNAME_SCORE'] = self.training_df.apply(lambda row: fuzz.ratio(row['FIRST_NAME1'], row['FIRST_NAME2'])/100, axis=1)
        self.training_df['LNAME_SCORE'] = self.training_df.apply(lambda row: fuzz.ratio(row['LAST_NAME1'], row['LAST_NAME2'])/100, axis=1)
        self.training_df['LNAME_PARTIAL_SCORE'] = self.training_df.apply(lambda row: fuzz.partial_ratio(row['LAST_NAME1'], row['LAST_NAME2'])/100, axis=1)
        self.training_df['FNAME_PARTIAL_SCORE'] = self.training_df.apply(lambda row: fuzz.partial_ratio(row['FIRST_NAME1'], row['FIRST_NAME2'])/100, axis=1)
        self.training_df['EMAIL_ADDR_SCORE'] = self.training_df.apply(lambda row: TrainingDataGenerator.get_score_for_email_address(row), axis=1)
        self.training_df['AUTH_KW_SCORE'] = self.training_df.apply(lambda row: TrainingDataGenerator.get_score_for_author_keywords(row), axis=1)
        self.training_df['COAUTHOR_SCORE'] = self.training_df.apply(lambda row: TrainingDataGenerator.get_score_for_coauthors(row), axis=1)
        self.training_scores_df = self.training_df[features]


    def generate_records(self):
        records = []
        for index, row in self.papers_df.iterrows():
            for index_child, row_child in self.papers_df.iterrows():
                if index != index_child:
                    d = {}
                    d = TrainingDataGenerator.set_feature_value(row, d, 'FIRST_NAME1', 'FIRSTNAME')
                    d = TrainingDataGenerator.set_feature_value(row, d, 'LAST_NAME1', 'LASTNAME')
                    d = TrainingDataGenerator.set_feature_value(row, d, 'EMAILADDRESS1', 'EMAILADDRESS')
                    d = TrainingDataGenerator.set_feature_value(row, d, 'INSTITUTE1', 'INSTITUTE')
                    d = TrainingDataGenerator.set_feature_value(row, d, 'AUTHOR_KW1', 'AUTHOR_KEYWORDS')
                    d = TrainingDataGenerator.set_feature_value(row, d, 'COAUTHORS1', 'CO-AUTHORS')

                    d = TrainingDataGenerator.set_feature_value(row_child, d, 'FIRST_NAME2', 'FIRSTNAME')
                    d = TrainingDataGenerator.set_feature_value(row_child, d, 'LAST_NAME2', 'LASTNAME')
                    d = TrainingDataGenerator.set_feature_value(row_child, d, 'EMAILADDRESS2', 'EMAILADDRESS')
                    d = TrainingDataGenerator.set_feature_value(row_child, d, 'INSTITUTE2', 'INSTITUTE')
                    d = TrainingDataGenerator.set_feature_value(row_child, d, 'AUTHOR_KW2', 'AUTHOR_KEYWORDS')
                    d = TrainingDataGenerator.set_feature_value(row_child, d, 'COAUTHORS2', 'CO-AUTHORS')
                    records.append(d)
        self.training_df = pd.DataFrame(records)
        if not self.random:
            self.training_df['MATCH'] = [1] * len(self.training_df)
        else:
            self.training_df['MATCH'] = [0] * len(self.training_df)

    def write_to_csv(self, path):
        self.training_df.to_csv(path)




def generate():
    dataFrame = None
    scoresDF = None
    for root, subfolders, files in os.walk(input_dataset):
        for file in files:
            fileName = os.path.join(root, file)
            samples = None
            random = True
            analyzer = DataAnalysisTool(fileName)
            if file in fileLastName:
                samples = analyzer.getPapersForAuthor(fileLastName[file], fileFirstName[file])
                random = False
            if 'random' in file:
                print len(analyzer.df)
                analyzer.df.drop_duplicates(subset='WOSID', inplace=True, keep='first')
                analyzer.df.drop_duplicates(subset=['LASTNAME', 'FIRSTNAME'], inplace = True, keep='first')
                analyzer.df.drop_duplicates(subset=['LASTNAME'], inplace=True, keep='first')
                #analyzer.df.drop_duplicates(subset=['EMAILADDRESS'], inplace=True, keep='first')
                print len(analyzer.df)
                samples = analyzer.df.sample(500)
            if samples is not None:
                training_data_generator = TrainingDataGenerator(samples, random=random)
                training_data_generator.generate_records()
                training_data_generator.calculate_scores()
                if dataFrame is None:
                    dataFrame = training_data_generator.training_df
                    scoresDF = training_data_generator.training_scores_df
                else:
                    dataFrame = dataFrame.append(training_data_generator.training_df)
                    scoresDF = scoresDF.append(training_data_generator.training_scores_df)
    if dataFrame is not None:
        dataFrame.to_csv('/Users/aosingh/AuthorDisambiguation/Training_data/train.csv', index=False)
        scoresDF.to_csv('/Users/aosingh/AuthorDisambiguation/Training_data/scores.csv', columns=features, index=False)


generate()







