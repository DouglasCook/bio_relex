import os
import csv
import pickle
import xml.etree.cElementTree as ET     # python XML manipulation library, C version because it's way faster!

import nltk.tokenize.punkt as punkt


def set_up_tokenizer():
    """
    Set up sentence splitter with custom parameters and return to caller
    """
    punkt_params = punkt.PunktParameters()
    # sentences are not split ending on the given parameters, using {} creates a set literal
    punkt_params.abbrev_types = {'inc', '.tm', 'tm', 'no', 'i.v', 'dr', 'drs', 'u.s', 'u.k', 'ltd', 'vs', 'vol', 'corp',
                                 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec',
                                 'pm', 'p.m', 'am', 'a.m', 'mr', 'mrs', 'ms', 'i.e', 'e.g',
                                 # above is from reuters, below for eu-adr specifically
                                 'spp'}
    # the tokenizer has to be unpickled so better do it once here than every time it is used
    return punkt.PunktSentenceTokenizer(punkt_params)


def fix_html(text):
    """
    Sort out HTML nonsense
    """
    # TODO make sure this covers everything
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&amp;', '&')
    return text


def abstracts_to_csv():
    """
    Create one record per text
    """
    # TODO parse the HTML properly, maybe use beautiful soup?
    basepath = os.path.dirname(__file__)
    # unpickle list of pubmed ids
    files = pickle.load(open('pubmed_ids.p', 'rb'))

    # need to deal with the fucking DS store file...
    files = [os.path.abspath(os.path.join(basepath, 'abstracts', f + '.xml')) for f in files]
    f_out = os.path.abspath(os.path.join(basepath, 'csv', 'sentences.csv'))

    sentence_splitter = set_up_tokenizer()

    with open(f_out, 'wb') as csv_out:
        csv_writer = csv.DictWriter(csv_out, ['id', 'sent_num', 'text'], delimiter=',')
        csv_writer.writeheader()

        for f in files:
            # parse the tree
            tree = ET.parse(f)

            # use xpath paths here to search entire tree
            pubmed_id = tree.findtext('.//PMID')
            title = tree.findtext('.//ArticleTitle')
            nodes = tree.findall('.//AbstractText')

            #abstract = ' '.join([e.text for e in tree.findall('.//AbstractText')])
            #text = title + ' ' + abstract

            # problems arise here if the abstract is in several parts eg method, conclusion etc
            # so need to construct text in this longwinded way instead of nice comprehension above
            abstract = ''
            for n in nodes:
                if n.attrib != {} and n.attrib['Label'] is not None and n.attrib['Label'] != 'UNLABELLED':
                    abstract += ' ' + n.attrib['Label'] + ': ' + n.text
                else:
                    abstract += ' ' + n.text
            text = title + abstract
            text = fix_html(text)

            sentences = sentence_splitter.tokenize(text)

            for i, s in enumerate(sentences):
                # dict comprehension here to hack the unicode into csv writer
                dict_row = {'id': pubmed_id, 'sent_num': i, 'text': s.encode('utf-8')}
                csv_writer.writerow(dict_row)


def relations_to_dict():
    """
    Put all relations into dictionary to pickle
    """
    # load pubmed ids
    pubmed_ids = pickle.load(open('pubmed_ids.p', 'rb'))
    basepath = os.path.dirname(__file__)

    relation_dict = {}

    for pid in pubmed_ids:
        f = os.path.abspath(os.path.join(basepath, '..', '..', 'data', 'euadr_corpus', pid + '.csv'))
        r_dict = {'start1': [], 'end1': [], 'start2': [], 'end2': [], 'true_relation': []}

        with open(f, 'rb') as csv_in:
            csv_reader = csv.reader(csv_in, delimiter='\t')

            for row in csv_reader:
                # if the row describes a relation add details to dict
                if row[2] == 'relation':
                    # want to split into start and end points for ease of use later
                    indices = row[7].split(':')
                    r_dict['start1'].append(int(indices[0]))
                    r_dict['end1'].append(int(indices[1]))

                    indices = row[8].split(':')
                    r_dict['start2'].append(int(indices[0]))
                    r_dict['end2'].append(int(indices[1]))

                    r_dict['true_relation'].append(row[1])

        # apparently some of the files contain no relations so need to check that here
        if len(r_dict['start1']) > 0:
            # now we need to order them by location since the spreadsheet is not in order
            sorter = zip(r_dict['start1'], r_dict['end1'], r_dict['start2'], r_dict['end2'], r_dict['true_relation'])
            sorter.sort()
            r_dict['start1'], r_dict['end1'], r_dict['start2'], r_dict['end2'], r_dict['true_relation'] = zip(*sorter)

            relation_dict[pid] = r_dict

    # pickle it
    print relation_dict['16950808']
    pickle.dump(relation_dict, open('relation_dict.p', 'wb'))


def create_output_row(relations, row, i, length):
    """
    Generate row to be written to csv with entities located
    """
    # TODO sort out the accents - unicode nonsense
    sent = unicode(row['text'], 'utf-8')

    # want first first entity to be that which appears first in text
    if relations['start1'][i] < relations['start2'][i]:
        start1 = relations['start1'][i] - length
        end1 = relations['end1'][i] - length
        start2 = relations['start2'][i] - length
        end2 = relations['end2'][i] - length
    else:
        start1 = relations['start2'][i] - length
        end1 = relations['end2'][i] - length
        start2 = relations['start1'][i] - length
        end2 = relations['end1'][i] - length

    e1 = sent[start1:end1]
    e2 = sent[start2:end2]
    rel = relations['true_relation'][i]
    between = sent[end1:start2]

    # need to re-encode everything as utf-8 ffs
    return [row['id'], row['sent_num'], rel, e1.encode('utf-8'), e2.encode('utf-8'), start1, end1, start2, end2,
            sent.encode('utf-8'), between.encode('utf-8')]


def create_input_file():
    """
    Set up input file with only those sentences containing relations
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, 'csv/sentences.csv'))
    file_out = os.path.abspath(os.path.join(basepath, 'csv/relevant_sentences.csv'))

    relation_dict = pickle.load(open('relation_dict.p', 'rb'))

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, delimiter=',')
            csv_writer = csv.writer(csv_out, ['pid', 'true_relation', 'e1', 'e2', 'start1', 'end1', 'start2', 'end2',
                                              'sent_num', 'sentence', 'between'], delimiter=',')
            length, i = 0, 0
            csv_writer.writerow(['pid', 'sent_num', 'true_relation', 'e1', 'e2', 'start1', 'end1', 'start2', 'end2',
                                 'sentence', 'between'])

            # loop through all sentences
            for row in csv_reader:
                # check that the key exists
                if row['id'] in relation_dict:
                    # if we are at start of next text reset counters
                    if row['sent_num'] == '0':
                        length, i = 0, 0

                        # get the relation info for this sentence
                        relations = relation_dict[row['id']]

                    # if not reached the end of these relations and
                    # if the next entity is in this sentence
                    while i < len(relations['start1']) and relations['start1'][i] < length + len(row['text']):
                        csv_writer.writerow(create_output_row(relations, row, i, length))
                        i += 1

                    # add length of sentence to total, need to add one for the missing space between sentences
                    length += len(row['text']) + 1


if __name__ == '__main__':
    abstracts_to_csv()
    create_input_file()
    #relations_to_dict()
