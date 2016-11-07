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

fileLastName = {}
fileFirstName = {}

fileLastName['Albertini_David.csv'] = ALBERITNI_LASTNAME
fileLastName['Arnold_John.csv'] = ARNOLD_LASTNAME
fileLastName['Boyer_Barbara.csv'] = BOYER_LASTNAME
fileLastName['Brigitte_Brandiff.csv'] = BRANDIFF_LASTNAME
fileLastName['Dawid_Igor.csv'] = DAWID_LASTNAME
fileLastName['Trischmann_Thomas.csv'] = TRISHCMAN_LASTNAME

fileFirstName['Albertini_David.csv'] = ALBERTINI_FIRSTNAME
fileFirstName['Arnold_John.csv'] = ARNOLD_FIRSTNAME
fileFirstName['Boyer_Barbara.csv'] = BOYER_FIRSTNAME
fileFirstName['Brigitte_Brandiff.csv'] = BRANDIFF_FIRSTNAME
fileFirstName['Dawid_Igor.csv'] = DAWID_FIRSTNAME
fileFirstName['Trischmann_Thomas.csv'] = TRISHCMAN_FIRSTNAME


class TrainingDataGenerator:

    def __init__(self, papers_df, random=True):
        self.papers_df = papers_df
        self.random = random
        self.training_df = None
        self.training_scores_df = None

    @staticmethod
    def set_feature_value(paper_sample, training_record, attribute_name, column_name):
        training_record[attribute_name] = paper_sample[column_name]
        return training_record

    @staticmethod
    def get_score_for_coauthors(training_record):
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
        '''

        :param institutions:
        :param author:
        :return:
        '''
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
        institute1 = TrainingDataGenerator.get_institute_name(training_record['INSTITUTE1'], training_record['LAST_NAME1'])
        institute2 = TrainingDataGenerator.get_institute_name(training_record['INSTITUTE2'], training_record['LAST_NAME2'])
        if institute1 is not None and institute2 is not None:
            institute1 = join_institute_names(split_institute(institute1)[0:3])
            institute2 = join_institute_names(split_institute(institute2)[0:3])
            return cosine_similarity(sentence_to_vector(institute1), sentence_to_vector(institute2))
        return 0

    def calculate_scores(self):
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
                analyzer.df.drop_duplicates(subset='WOSID', inplace=True, keep='first')
                analyzer.df.drop_duplicates(subset=['LASTNAME', 'FIRSTNAME'], inplace = True, keep='first')
                analyzer.df.drop_duplicates(subset=['LASTNAME'], inplace=True, keep='first')
                analyzer.df.drop_duplicates(subset=['EMAILADDRESS'], inplace=True, keep='first')
                samples = analyzer.df.sample(300)
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













