import nltk
import csv              # used for accessing data held in CSV format
import os.path          # need this to use relative filepaths
import re               # regular expressions for extracting named entities

import preprocessing


def pos_tags():
    """ Create new CSV containing all relevant sentences """

    # set filepath to input
    basepath = os.path.dirname(__file__)
    filepath = 'data/biotext/sentences_with_roles_and_relations.txt'
    filepath = os.path.abspath(os.path.join(basepath, '..', '..', filepath))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'biotext/sentences_POS.csv'))

    with open(filepath, 'r') as f_in:
        with open(file_out, 'wb') as csv_out:
            csv_writer = csv.writer(csv_out, delimiter=',')

            csv_writer.writerow(['CLASS_TAG', 'SENTENCE', 'POS_TAGS'])

            for line in f_in:
                # split sentence and class tag and strip newline
                line = line.split('||')
                line[1] = line[1].rstrip()
                # repr can be used to see raw string ie include newlines etc
                #print repr(line[1])

                # only looking at treatment and disease sentences for now, need to get negative examples though...
                if line[1] == 'TREAT_FOR_DIS':
                    csv_writer.writerow([line[1], line[0], preprocessing.clean_and_tag_sentence(line[0])])


def entity_extraction():
    """
    What do I want to extract? How to format the entities?
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'biotext/sentences_chunk.csv'))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'biotext/sentences_entities.csv'))

    with open(file_in, 'r') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.reader(csv_in, delimiter=',')
            csv_writer = csv.writer(csv_out, delimiter=',')

            # write first row
            row = csv_reader.next()
            row.extend(['DISEASES', 'TREATMENTS'])
            csv_writer.writerow(row)

            for row in csv_reader:
                # find all tagged entities
                # [^x] matches any character != x, hacky way to stop at closing tag
                diseases = re.findall(r'<DIS>[^<]*', row[1])
                treatments = re.findall(r'<TREAT>[^<]*', row[1])
                diseases = [d[5:].strip(' ') for d in diseases]
                treatments = [t[7:].strip(' ') for t in treatments]

                # below may be a more 'correct' regex but it seems slower
                # having ? after the * operator makes it non-greedy ie stop at first occurence of next thing
                #diseases = re.findall(r'<DIS>.*?</DIS>', row[1])
                #treatments = re.findall(r'<TREAT>.*?</TREAT>', row[1])
                #diseases = [d[5:-6] for d in diseases]
                #treatments = [t[7:-8] for t in treatments]

                # now can clean up tags - don't actually want to do this?
                row[1] = nltk.clean_html(row[1])

                # replace spaces with underscores so we get single 'words'
                for d in diseases:
                    original = d
                    d = d.replace(' ', '_')
                    row[1] = row[1].replace(original, d)

                for t in treatments:
                    original = t
                    t = t.replace(' ', '_')
                    row[1] = row[1].replace(original, t)

                row.extend([diseases, treatments])
                csv_writer.writerow(row)


def boom():
    #pos_tags()
    preprocessing.chunk('biotext/sentences_POS.csv', 'biotext/sentences_chunk.csv')
    entity_extraction()


def feature_extraction():
    """
    Extract useful(?) features from preprocessed CSV and create input file for WEKA
    """
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'biotext/sentences_entities.csv'))

    with open(file_in, 'r') as csv_in:
        csv_reader = csv.reader(csv_in, delimiter=',')

        csv_reader.next()
        row = csv_reader.next()
        words = nltk.clean_html(row[1])
        words = nltk.word_tokenize(words)
        print words
        diseases = eval(row[3])
        treatments = eval(row[4])

        # fix this or throw it away!
        for dis in diseases:
            print dis, words.index(dis.replace(' ', '_'))
        # create feature vector for each row
        #for row in csv_reader:
            # count number of words between mentions

            # bag of words (of mentions and between mentions)

            # POS tags of words (what words?)

            # something to do with chunking (count NPs or VPs or something?)


if __name__ == '__main__':
    #entity_extraction()
    feature_extraction()
    #boom()
