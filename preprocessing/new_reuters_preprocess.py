import nltk
import csv              # used for accessing data held in CSV format
import os.path          # need this to use relative filepaths
import sys              # using sys for std out
import re

import preprocessing

import nltk.tokenize.punkt as punkt


def collate_texts():
    """
    Create one record per text fragment, with lists for all drugs and companies
    Only keep those texts that mention at least one of the drugs and one of the companies
    """

    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = 'data/reuters/press_releases/PR_drug_company_500.csv'
    file_in = os.path.abspath(os.path.join(basepath, '..', '..', file_in))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'reuters_new/single_records.csv'))

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, delimiter=',')
            csv_writer = csv.writer(csv_out, delimiter=',')

            drugs = set([])
            companies = set([])
            src = '174077'
            text = ''

            csv_writer.writerow(['SOURCE_ID', 'DRUGS', 'COMPANIES', 'FRAGMENT'])

            # think that the dict reader skips header row automagically
            for row in csv_reader:
                if row['SOURCE_ID'] != src:

                    # first check if the text contains at least on of each drug and company tagged
                    d_relevant = False
                    c_relevant = False
                    for d in drugs:
                        if d in text:
                            d_relevant = True
                            break

                    if d_relevant:
                        for c in companies:
                            if c in text:
                                c_relevant = True
                                break

                    if d_relevant and c_relevant:
                        csv_writer.writerow([src, list(drugs), list(companies), text])

                    # reset lists
                    drugs = set([])
                    companies = set([])

                # append drugs and companies to lists
                drugs.add(row['DRUG_NAME'])
                companies.add((row['COMPANY_NAME']))
                src = row['SOURCE_ID']
                text = row['FRAGMENT']


def clean_and_tag_all():
    """ Create new CSV containing tagged versions of all sentences """

    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters_new/single_records.csv'))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'reuters_new/sentences_POS.csv'))

    # set up sentence splitter with custom parameters
    punkt_params = punkt.PunktParameters()
    # sentences are not split ending on the given parameters, using {} creates a set literal
    punkt_params.abbrev_types = {'inc', 'inc ', '.tm', 'tm', 'no', 'i.v', 'dr', 'drs', 'u.s', 'ltd', 'vs', 'vol',
                                 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
                                 'pm', 'p.m', 'am', 'a.m'}
    # the tokenizer has to be unpickled so better do it once here than every time it is used
    sentence_splitter = punkt.PunktSentenceTokenizer(punkt_params)

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, delimiter=',')
            csv_writer = csv.writer(csv_out, delimiter=',')

            # write column headers on first row
            csv_writer.writerow(['SOURCE_ID', 'SENTENCE', 'DRUGS', 'COMPANIES', 'POS_TAGS'])

            for row in csv_reader:
                # use stdout to avoid spaces and newlines
                sys.stdout.write('.')
                # need to flush the buffer to display immediately
                sys.stdout.flush()

                # clean up html tags
                plaintext = nltk.clean_html(row['FRAGMENT'])
                # split into sentences
                sentences = sentence_splitter.tokenize(plaintext)
                # TODO coreference resolution to find more relevant sentences

                if len(sentences) > 0:
                    for s in sentences:
                        # remove punctuation, still want to add original sentence to CSV though
                        #no_punct = re.findall(r'[\w\$\xc2()-]+', s)

                        # want to keep hyphenated words but none of the other hyphens
                        # replace any hyphenated words' hyphens with underscores
                        for hyphen in re.findall(r'\w-\w', s):
                            underscore = hyphen.replace('-', '_')
                            s = s.replace(hyphen, underscore)

                        no_punct = re.findall(r'[\w\$\xc2_]+', s)
                        no_punct = ' '.join(no_punct)
                        # put the hyphens back
                        s = s.replace('_', '-')

                        tokens = nltk.word_tokenize(no_punct)
                        tags = nltk.pos_tag(tokens)

                        # TODO parse tree info, something to do with stemming?
                        # write row to file for each sentence
                        csv_writer.writerow([row['SOURCE_ID'], s, row['DRUGS'], row['COMPANIES'], tags])


def locate_entities():
    """
    Locate named drugs and companies, indexed by word
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters_new/sentences_POS.csv'))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'reuters_new/entities_marked.csv'))

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, delimiter=',')
            csv_writer = csv.writer(csv_out, delimiter=',')

            # write headers
            csv_writer.writerow(['SOURCE_ID', 'SENTENCE', 'DRUGS', 'COMPANIES', 'POS_TAGS'])

            for row in csv_reader:
                drug_dict = {}
                comp_dict = {}
                tags = eval(row['POS_TAGS'])

                for drug in eval(row['DRUGS']):
                    # locate first word if entity is made of multiple words
                    space = drug.find(' ')
                    if space > 0:
                        head_word = drug[:space]
                    else:
                        head_word = drug

                    # add indices of head word to dict entry for this drug
                    drug_dict[drug] = [i for i, x in enumerate(tags) if x[0] == head_word]

                for company in eval(row['COMPANIES']):
                    # locate first word if entity is made of multiple words
                    space = company.find(' ')
                    if space > 0:
                        head_word = company[:space]
                    else:
                        head_word = company

                    # add indices of head word to dict entry for this drug
                    comp_dict[company] = [i for i, x in enumerate(tags) if x[0] == head_word]

                csv_writer.writerow([row['SOURCE_ID'], row['SENTENCE'], drug_dict, comp_dict, tags])


if __name__ == '__main__':
    #collate_texts()
    #clean_and_tag_all()
    locate_entities()
