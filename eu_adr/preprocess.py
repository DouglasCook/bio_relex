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
                                 'pm', 'p.m', 'am', 'a.m', 'mr', 'mrs', 'ms', 'i.e'}
    # the tokenizer has to be unpickled so better do it once here than every time it is used
    return punkt.PunktSentenceTokenizer(punkt_params)


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
            abstract = ' '.join([e.text for e in tree.findall('.//AbstractText')])
            text = title + ' ' + abstract

            sentences = sentence_splitter.tokenize(text)

            for i, s in enumerate(sentences):
                # dict comprehension here to hack the unicode into csv writer
                dict_row = {'id': pubmed_id, 'sent_num': i, 'text': s.encode('utf-8')}
                csv_writer.writerow(dict_row)


def relations_to_dict():
    """
    Put all relations into
    """
    # load pubmed ids
    pubmed_ids = pickle.load(open('pubmed_ids.p', 'rb'))
    basepath = os.path.dirname(__file__)

    relation_dict = {'pubmed_id': [], 'pos1': [], 'pos2': [], 'true_relation': []}

    for pid in pubmed_ids:
        f = os.path.abspath(os.path.join(basepath, '..', '..', 'data', 'euadr_corpus', pid + '.csv'))

        with open(f, 'rb') as csv_in:
            csv_reader = csv.reader(csv_in, delimiter='\t')

            for row in csv_reader:
                # if the row describes a relation add details to dict
                if row[2] == 'relation':
                    relation_dict['pubmed_id'].append(pid)
                    relation_dict['pos1'].append(row[7])
                    relation_dict['pos2'].append(row[8])
                    relation_dict['true_relation'].append(row[1])

    # pickle it
    pickle.dump(relation_dict, open('relation_dict.p', 'wb'))


if __name__ == '__main__':
    relations_to_dict()
    #abstracts_to_csv()
