import os
import csv
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


def convert_to_csv():
    """
    Create one record per text
    """
    # TODO parse the HTML properly, maybe use beautiful soup?
    basepath = os.path.dirname(__file__)
    file_path = os.path.abspath(os.path.join(basepath, 'abstracts'))
    files = os.listdir(file_path)

    # need to deal with the fucking DS store file...
    files = [os.path.abspath(os.path.join(basepath, 'abstracts', f)) for f in files if f != '.DS_Store']
    f_out = os.path.abspath(os.path.join(basepath, 'csv', 'texts.csv'))

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


if __name__ == '__main__':
    convert_to_csv()
