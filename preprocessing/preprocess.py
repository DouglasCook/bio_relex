import nltk
import nltk.tokenize.punkt as punkt

import csv  # used for accessing data held in CSV format
import os.path  # need this to use relative filepaths
import sys  # using sys for std out
import re

import chunking


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


def remove_punctuation(sentence):
    """
    Remove all punctuation from sentence
    Return original sentence and that with no punctuation
    """
    # want to keep hyphenated words but none of the other hyphens
    # replace any hyphenated words' hyphens with underscores
    for hyphenated in re.findall(r'\w-\w', sentence):
        underscored = hyphenated.replace('-', '_')
        sentence = sentence.replace(hyphenated, underscored)

    # TODO this is removing decimal points and percentages, need to fix it
    # %% is how to escape a percentage in python regex

    # remove punctuation, still want to add original sentence to CSV though
    #no_punct = re.findall(r'[\w\$\xc2()-]+', s)
    no_punct = re.findall(r'[\w\$\xc2_]+', sentence)
    no_punct = ' '.join(no_punct)

    # may not want to do this, underscores work nicer when chunking
    # put the hyphens back
    #sentence = sentence.replace('_', '-')

    return sentence, no_punct


def collate_texts(delimiter):
    """
    Create one record per text fragment, with lists for all drugs and companies
    Only keep those texts that mention at least one of the drugs and one of the companies
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    #file_in = 'data/reuters/press_releases/PR_drug_company_500.csv'
    file_in = 'data/reuters/TR_PR_DrugCompany.csv'
    file_in = os.path.abspath(os.path.join(basepath, '..', '..', file_in))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'reuters/single_records.csv'))

    #with open(file_in, 'rb') as csv_in:
    # may need to open with rU to deal with universal newlines - something to do with excel
    with open(file_in, 'rU') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, delimiter=delimiter)
            csv_writer = csv.writer(csv_out, delimiter=',')

            drugs = set([])
            companies = set([])
            # TODO need to fix this so it always gets first record regardless of source ID
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
                        csv_writer.writerow([src, list(drugs), list(companies), nltk.clean_html(text)])

                    # reset lists
                    drugs = set([])
                    companies = set([])

                # append drugs and companies to lists
                drugs.add(row['DRUG_NAME'])
                companies.add((row['COMPANY_NAME']))
                src = row['SOURCE_ID']
                text = row['FRAGMENT']

    print 'Written to single_records_clean.csv'


def clean_and_tag_all():
    """
    Create new CSV containing tagged versions of all sentences
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters/single_records.csv'))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'reuters/sentences_POS.csv'))

    sentence_splitter = set_up_tokenizer()
    chunker = chunking.set_up_chunker()
    stemmer = nltk.SnowballStemmer('english')

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, ['SOURCE_ID', 'DRUGS', 'COMPANIES', 'SENTENCE'], delimiter=',')
            csv_writer = csv.DictWriter(csv_out, ['SOURCE_ID', 'SENT_NUM', 'SENTENCE', 'NO_PUNCT',
                                                  'DRUGS', 'COMPANIES', 'POS_TAGS', 'CHUNKS'], delimiter=',')
            csv_writer.writeheader()
            #csv_reader.next()

            for row in csv_reader:
                # display progress bar
                sys.stdout.write('.')
                sys.stdout.flush()

                # clean up html tags
                # named SENTENCE in the reader so it works nicely when writing row
                plaintext = nltk.clean_html(row['SENTENCE'])
                # this in particular seems to be screwing up some of the sentence splitting
                plaintext = plaintext.replace('Inc .', 'Inc.')
                # split into sentences
                sentences = sentence_splitter.tokenize(plaintext)

                if len(sentences) > 0:
                    for i, s in enumerate(sentences):

                        # TODO integrate stanford NER recognition output into this

                        # clean up sentence
                        s, no_punct = remove_punctuation(s)

                        # CHUNKING - need to include punctuation for this to be anywhere near accurate
                        tokens = nltk.pos_tag(nltk.word_tokenize(s))
                        chunks = chunker.parse(tokens)

                        # POS TAGS - don't want to include punctuation
                        tokens = nltk.word_tokenize(no_punct)
                        # put the hyphens back after tokenisation
                        # underscores mean that the tokens are better recognised when tagging
                        no_punct = no_punct.replace('_', '-')
                        s = s.replace('_', '-')
                        tags = nltk.pos_tag(tokens)

                        # STEMMING - add stemmed version of word to end of each tagged token
                        tags = [(token, tag, stemmer.stem(token.lower())) for (token, tag) in tags]

                        # TODO parse tree info, chunking, something to do with stemming?
                        # ignore any rogue bits of punctuation etc
                        if len(tags) > 1:
                            # write row to file for each sentence
                            new_fields = {'SENT_NUM': i, 'SENTENCE': s, 'NO_PUNCT': no_punct, 'POS_TAGS': tags,
                                          'CHUNKS': chunks}
                            row.update(new_fields)
                            csv_writer.writerow(row)

    print 'Written to sentences_POS.csv'


def stanford_entity_recognition():
    """
    Produce NE chunks from POS tags - this uses the Stanford tagger implementation
    This is actually too slow to be of any use, there must be a way to batch it but for now just using bash script
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters/sentences_POS.csv'))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'reuters/sentences_NE.csv'))

    # set up tagger
    st = nltk.tag.stanford.NERTagger(
        '/Users/Dug/Imperial/individual_project/tools/stanford_NLP/stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz',
        '/Users/Dug/Imperial/individual_project/tools/stanford_NLP/stanford-ner-2014-06-16/stanford-ner.jar')

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, delimiter=',')
            csv_writer = csv.writer(csv_out, delimiter=',')

            # write column headers on first row
            csv_writer.writerow(['SOURCE_ID', 'SENTENCE', 'DRUGS', 'COMPANIES', 'POS_TAGS', 'NE_CHUNKS'])

            for row in csv_reader:
                ne_chunks = st.tag(row['NO_PUNCT'].split())
                csv_writer.writerow([row['SOURCE_ID'], row['SENTENCE'], row['DRUGS'], row['COMPANIES'],
                                     row['POS_TAGS'], ne_chunks])

    print 'Written to sentences_NE.csv'


def nltk_entity_recognition():
    """
    Produce NE chunks from POS tags - this NLTK implementation is not great though so should use Stanford output instead
    This needs to be done before the punctuation is removed
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters/sentences_POS.csv'))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'reuters/sentences_NE.csv'))

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, delimiter=',')
            csv_writer = csv.writer(csv_out, delimiter=',')

            # write column headers on first row
            csv_writer.writerow(['SOURCE_ID', 'SENTENCE', 'DRUGS', 'COMPANIES', 'POS_TAGS', 'NE_CHUNKS'])

            for row in csv_reader:
                print row
                # use NLTK NE recognition, binary means relations are not classified
                # it's based on the ACE corpus so may not work completely as desired...
                tags = eval(row['POS_TAGS'])
                ne_chunks = nltk.ne_chunk(tags, binary=True)
                row.append(ne_chunks)
                csv_writer.writerow(row)
                # csv_writer.writerow([row['SOURCE_ID'], row['SENTENCE'], row['DRUGS'], row['COMPANIES'],
                # row['POS_TAGS'], ne_chunks])


def preprocessing():
    """
    Step 1 in the pipeline so far...
    Retrieve and clean relevant texts from CSV and carry out POS tagging
    """
    collate_texts()
    clean_and_tag_all()


if __name__ == '__main__':
    collate_texts('\t')
    #preprocessing()
    #stanford_entity_recognition()