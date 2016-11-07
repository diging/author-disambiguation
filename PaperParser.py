import csv
import os

from tethne.readers import wos


header = ["WOSID","DATE","TITLE","LASTNAME","FIRSTNAME","JOURNAL","EMAILADDRESS",\
          "PUBLISHER","SUBJECT","WC","AUTHOR_KEYWORDS","INSTITUTE","CO-AUTHORS"]


class PaperParser:
    """
    This module uses Tethne to parse WOS tagged-file data and writes the output to a CSV file.

    There are 2 possible use-cases
            1. Parse a file
            2. Parse a directory of files - Recursively searches for WOS txt files and parses them

    Notice(in the examples below) the difference in the number of arguments while creating the parser object.

    ``Example 1 : Parse a single WOS text file``
            >>> parser = PaperParser('/Users/aosingh/TethneDataAnalysis/MBL History Data/1971/Albertini_David.txt',
            >>>                     '/Users/aosingh/AuthorDisambiguation/Dataset',)
            >>> parser.parseFile()

    ``Example 2 : Parse a directory of WOS text files``
            >>> parser = PaperParser('/Users/aosingh/TethneDataAnalysis/MBL History Data/',
            >>>                     '/Users/aosingh/AuthorDisambiguation/Dataset', output_filename='records.csv')
            >>> parser.parseDirectory()

    If you intend to parse a directory of WOS text files then the argument ``ouput_filename`` is to be passed.

    else if you intend to parse a single WOS text file then the output file name is not to be passed.
        The output file name will be the same as input file name

    """

    def __init__(self, inputlocation, outputlocation, output_filename=None):
        """
        Initialise the parser object.


        :param inputlocation (str):  Could be a directory or a single text file location
        :param outputlocation (str):  The directory where you want the output file to be written
        :param output_filename (str): The name of the output csv file. default value is None.
                                      This argument is only to be passed when your input location is a directory
        :return:
        """
        self.location = inputlocation
        if '.txt' in inputlocation:
            csv = os.path.basename(inputlocation).replace('txt', 'csv')
        else:
            csv = output_filename
        self.csv = os.path.join(outputlocation, csv)

    def parseDirectory(self):
        """
        Recursively searches for WOS txt files and parses each one of them.
        The output CSV will have the following columns

        >>>header = ["WOSID","DATE","TITLE","LASTNAME","FIRSTNAME","JOURNAL","EMAILADDRESS",\
          "PUBLISHER","SUBJECT","WC","AUTHOR_KEYWORDS","INSTITUTE","CO-AUTHORS"]

        :return:
        """
        with(open(self.csv, 'wb')) as headerRecord:
            headerWriter = csv.writer(headerRecord, delimiter=",")
            headerWriter.writerow(header)
        for root, subfolders, files in os.walk(self.location):
            for file in files:
                if '.txt' in file:
                    fullPath = os.path.join(root, file)
                    papers = wos.read(fullPath)
                    print "total length of papers",len(papers)
                    with open(self.csv, 'a') as csvfile:
                        paperWriter = csv.writer(csvfile, delimiter=",")

                        for paper in papers:
                            set_of_authors = set(paper.authors_full)
                            for author in set_of_authors:
                                currentAuthorSet = set()
                                currentAuthorSet.add(author)
                                coauthorSet = set_of_authors.difference(currentAuthorSet)
                                lastname = author[0]
                                firstname = author[1]
                                row = getattr(paper, 'wosid', ''), \
                                      str(getattr(paper, 'date', '')), \
                                      getattr(paper, 'title', ''), \
                                      lastname, \
                                      firstname, \
                                      getattr(paper, 'journal', ''), \
                                      getattr(paper, 'emailAddress', []),\
                                      getattr(paper, 'publisher', ''),\
                                      getattr(paper, 'subject', []),\
                                      getattr(paper, 'WC', ''),\
                                      getattr(paper, 'authorKeywords', []),\
                                      getattr(paper, 'authorAddress', ""),\
                                      list(coauthorSet)
                                paperWriter.writerow(row)

    def parseFile(self):
        """
        Parses a single WOS file passed in the input.

        The output CSV will have the following columns

        >>> header = ["WOSID","DATE","TITLE","LASTNAME","FIRSTNAME","JOURNAL","EMAILADDRESS",\
          "PUBLISHER","SUBJECT","WC","AUTHOR_KEYWORDS","INSTITUTE","CO-AUTHORS"]

        :return:
        """
        with(open(self.csv, 'wb')) as headerRecord:
            headerWriter = csv.writer(headerRecord, delimiter=",")
            headerWriter.writerow(header)
        papers = wos.read(self.location)
        print "total length of papers",len(papers)
        with open(self.csv, 'a') as csvfile:
            paperWriter = csv.writer(csvfile, delimiter=",")
            for paper in papers:
                set_of_authors = set(paper.authors_full)
                for author in set_of_authors:
                    currentAuthorSet = set()
                    currentAuthorSet.add(author)
                    coauthorSet = set_of_authors.difference(currentAuthorSet)
                    print coauthorSet
                    lastname = author[0]
                    firstname = author[1]
                    row = getattr(paper, 'wosid', ''), \
                        str(getattr(paper, 'date', '')), \
                        getattr(paper, 'title', ''), \
                        lastname, \
                        firstname, \
                        getattr(paper, 'journal', ''), \
                        getattr(paper, 'emailAddress', []),\
                        getattr(paper, 'publisher', ''),\
                        getattr(paper, 'subject', []),\
                        getattr(paper, 'WC', ''),\
                        getattr(paper, 'authorKeywords', []),\
                        getattr(paper, 'authorAddress', ""),\
                        list(coauthorSet)
                    paperWriter.writerow(row)


parser = PaperParser('/Users/aosingh/TethneDataAnalysis/MBL History Data/1971/Albertini_David.txt', '/Users/aosingh/AuthorDisambiguation/Dataset',)
parser.parseFile()