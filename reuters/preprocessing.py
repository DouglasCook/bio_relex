import nltk
import csv              # used for accessing data held in CSV format
import os.path          # need this to use relative filepaths

def test():
    # in order to access a file relative to current directory need to build a path like this...
    basepath = os.path.dirname(__file__)
    filepath = 'data/reuters/press_releases/PR_drug_company_500.csv'
    filepath = os.path.abspath(os.path.join(basepath, '..', '..', filepath))

    # using with ... as here insures that the file will be closed at the end
    # it is a sort of shorthand for a try ... finally block
    with open(filepath, 'rb') as csvfile:

        # the reader is an iterable object
        csvreader = csv.reader(csvfile, delimiter=',')
        
        # skip first row since it just contains names of columns
        csvreader.next()
        record = csvreader.next()

        # second field is the text, clean_html removes HTML tags
        raw = nltk.clean_html(record[1])
        tokens = nltk.word_tokenize(raw)
        text = nltk.Text(tokens)

        # part-of-speech tagging for all tokens
        tagged = nltk.pos_tag(tokens) 
        tag_fd = nltk.FreqDist(tag for (word, tag) in tagged)
        tag_fd.plot()

        # returns words that are in similar context to the given word
        #print text.similar('repifermin')


def pos_tagging():
    """ Reduce dataset to only those where text contains mentions of both entities """

    # set filepath to input
    basepath = os.path.dirname(__file__)
    filepath = 'data/reuters/press_releases/PR_drug_company_500.csv'
    filepath = os.path.abspath(os.path.join(basepath, '..', '..', filepath))

    previous_id = 0

    with open(filepath, 'rb') as csv_in:

        with open('out.csv', 'wb') as csv_out:
            csv_reader = csv.reader(csv_in, delimiter=',')
            csv_writer = csv.writer(csv_out, delimiter=',')

            # write column headers on first row
            row = csv_reader.next()
            csv_writer.writerow([row[0], row[1], 'POS tags', row[2], row[3], row[4], row[5]])

            for row in csv_reader:
                add = True
                # remove HTML tags
                raw = nltk.clean_html(row[1])
                source_id = row[0]
                drug = row[3]
                company = row[5]

                # check if the drug is mentioned in the text
                if drug not in raw:
                    print source_id, drug 
                    add = False
                # check if company is mentioned in text
                if company not in raw:
                    print source_id, company 
                    add = False
                
                if add:
                    # further process the string
                    tokens = nltk.word_tokenize(raw)
                    #text = nltk.Text(tokens)

                    # only compute pos tags if haven't already
                    if source_id != previous_id:
                        pos_tags = nltk.pos_tag(tokens)

                    # write new row to file
                    out_row = [source_id, raw, pos_tags, row[2], drug, row[4], company]
                    csv_writer.writerow(out_row)

                previous_id = source_id


def count_stats():
    """ Reduce dataset to only those where text contains mentions of both entities """

    # set filepath to input
    basepath = os.path.dirname(__file__)
    filepath = 'data/reuters/press_releases/PR_drug_company_500.csv'
    filepath = os.path.abspath(os.path.join(basepath, '..', '..', filepath))

    with open(filepath, 'rb') as csv_in:

        csv_reader = csv.reader(csv_in, delimiter=',')

        # skip column names
        csv_reader.next()
        #row = csv_reader.next()

        d_count = 0
        c_count = 0
        prev_id = 0
        docs = 0

        for row in csv_reader:
            # clean up html tags
            plaintext = nltk.clean_html(row[1])
            drug = row[3]
            company = row[5]
            src = row[0]

            if src != prev_id:
                docs += 1
            prev_id = src

            if drug in plaintext:
                d_count += 1

            if company in plaintext:
                c_count += 1

        print d_count, 'drug texts and', c_count, 'company texts', docs, 'unique texts'


def preprocess_view():
    """ Display all sentences containing drug, company or both"""

    # set filepath to input
    basepath = os.path.dirname(__file__)
    filepath = 'data/reuters/press_releases/PR_drug_company_500.csv'
    filepath = os.path.abspath(os.path.join(basepath, '..', '..', filepath))

    with open(filepath, 'rb') as csv_in:

        csv_reader = csv.reader(csv_in, delimiter=',')

        # skip column names
        csv_reader.next()
        #row = csv_reader.next()
        count = 0

        for row in csv_reader:
            # clean up html tags
            plaintext = nltk.clean_html(row[1])
            drug = row[3]
            company = row[5]

            # only consider texts containing both the drug and company
            if drug in plaintext and company in plaintext:
                print '------ RECORD NUMBER', row[0], ', COMPANY =', company, ', DRUG =', drug, '------\n'

                sentences = nltk.sent_tokenize(plaintext)

                # filter for only sentences mentioning drug, company or both
                sentences = [s for s in sentences if drug in s or company in s]

                # filter for only those mentioning both drug and company 
                #sentences = [s for s in sentences if company in s]

                #sentences = [nltk.word_tokenize(s) for s in sentences]
                #sentences = [nltk.pos_tag(s) for s in sentences]

                if len(sentences) > 0:
                    count += 1
                    for s in sentences:
                        print s, '\n'

        print count, 'sentences in total'


def preprocess():
    """ Create new CSV containing all relevant sentences """
    # using sys for std out
    import sys
    import nltk.tokenize.punkt as punkt

    # set filepath to input
    basepath = os.path.dirname(__file__)
    filepath = 'data/reuters/press_releases/PR_drug_company_500.csv'
    filepath = os.path.abspath(os.path.join(basepath, '..', '..', filepath))

    # set up sentence splitter with custom parameters
    punkt_params = punkt.PunktParameters()
    # sentences are not split ending on the given parameters, using {} creates a set literal
    punkt_params.abbrev_types = {'inc', 'inc ', '.tm', 'tm', 'no', 'i.v', 'drs', 'u.s'}
    # the tokenizer has to be unpickled so creating it here is more efficient than every time it is used
    sentence_splitter = punkt.PunktSentenceTokenizer(punkt_params)

    with open(filepath, 'rb') as csv_in:
        with open('sentences_POS.csv', 'wb') as csv_out:
            csv_reader = csv.reader(csv_in, delimiter=',')
            csv_writer = csv.writer(csv_out, delimiter=',')

            # write column headers on first row
            row = csv_reader.next()
            csv_writer.writerow([row[0], 'SENTENCE', 'POS TAGS', row[2], row[3], row[4], row[5]])

            for row in csv_reader:
                # use stdout to avoid spaces and newlines
                sys.stdout.write('.')
                # need to flush the buffer to display immediately
                sys.stdout.flush()

                # clean up html tags
                plaintext = nltk.clean_html(row[1])
                drug = row[3]
                company = row[5]
                src = row[0]

                # only consider texts containing both the drug and company
                if drug in plaintext and company in plaintext:
                    sentences = sentence_splitter.tokenize(plaintext)

                    # filter for only sentences mentioning drug, company or both
                    sentences = [s for s in sentences if drug in s or company in s]

                    # TODO clean up text more, remove stop words and punctuation

                    if len(sentences) > 0:
                        for s in sentences:
                            tokens = nltk.word_tokenize(s)
                            tags = nltk.pos_tag(tokens)
                            # TODO add chunk info, parse tree info, something to do with stemming?
                            # write row to file for each sentence
                            csv_writer.writerow([src, s, tags, row[2], drug, row[4], company])


def experiment():
    """ Create new CSV containing all relevant sentences """

    with open('sentences_POS.csv', 'rb') as csv_in:
        csv_reader = csv.reader(csv_in, delimiter=',')
        for row in csv_reader:
            print nltk.ne_chunk(row[2])

if __name__ == '__main__':
    preprocess()
