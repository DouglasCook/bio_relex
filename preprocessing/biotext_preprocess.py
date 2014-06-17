import csv              # used for accessing data held in CSV format
import os.path          # need this to use relative filepaths

import tagging
import chunking


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
                if line[1] == 'TREAT_FOR_DIS':
                    csv_writer.writerow([line[1], line[0], tagging.clean_and_tag_sentence(line[0])])


def boom():
    basic_csv()
    chunking.chunk('biotext/sentences_POS.csv', 'biotext/sentences_chunk.csv')


def only_easy():
    """
    Strip out any sentences not mentioning both drug and company
    """
    with open('data/sentences_chunk.csv', 'rb') as csv_in:
        with open('data/sentences_easy_set.csv', 'wb') as csv_out:
            csv_reader = csv.reader(csv_in, delimiter=',')
            csv_writer = csv.writer(csv_out, delimiter=',')

            row = csv_reader.next()
            csv_writer.writerow(row)

            for row in csv_reader:
                if row[3] in row[1] and row[5] in row[1]:
                    csv_writer.writerow(row)

if __name__ == '__main__':
    boom()
