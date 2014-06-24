import nltk
import csv          # for reading and writing CSV files
import ast          # ast is needed to convert list from string to proper list
import os           # for filepath stuff
import sys          # for displaying progress as ...

import nltk.tokenize.punkt as punkt


def clean_and_tag(row, text_col, csv_writer):
    """
    Clean given text and write each sentence to CSV
    """
    # set up sentence splitter with custom parameters
    punkt_params = punkt.PunktParameters()
    # sentences are not split ending on the given parameters, using {} creates a set literal
    punkt_params.abbrev_types = {'inc', 'inc ', '.tm', 'tm', 'no', 'i.v', 'drs', 'u.s'}
    # the tokenizer has to be unpickled so better do it once here than every time it is used
    sentence_splitter = punkt.PunktSentenceTokenizer(punkt_params)

    # clean up html tags
    plaintext = nltk.clean_html(row[text_col])
    # TODO coreference resolution to find more relevant sentences
    sentences = sentence_splitter.tokenize(plaintext)

    # maybe unecessary defensiveness...
    if len(sentences) > 0:
        for s in sentences:
            # remove punctuation, still want to add original sentence to CSV though
            #no_punct = re.findall(r'[\w\$\xc2()-]+', s)
            #no_punct = ' '.join(no_punct)
            tokens = nltk.word_tokenize(s)
            tags = nltk.pos_tag(tokens)

            # TODO parse tree info, something to do with stemming?
            # write row to file for each sentence
            row.append(tags)
            csv_writer.writerow(row)


def clean_and_tag_sentence(sentence):
    """
    Clean and tag sentence, return POS tags
    """
    # clean up html tags
    # use stdout to avoid spaces and newlines
    sys.stdout.write('.')
    # need to flush the buffer to display immediately
    sys.stdout.flush()
    plaintext = nltk.clean_html(sentence)
    tokens = nltk.word_tokenize(plaintext)
    tags = nltk.pos_tag(tokens)

    return tags


def chunk(f_in, f_out):
    """
    Chunk the POS tagged sentences using basic regex grammar
    The IOB tags include POS tags so should replace existing field in CSV
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    # TODO work out what directory structure to use...
    filepath = os.path.abspath(os.path.join(basepath, '..', f_in))
    file_out = os.path.abspath(os.path.join(basepath, '..', f_out))

    # TODO improve the grammar! should split into multiple rules
    # first need a grammar to define the chunks
    # can use nltk.app.chunkparser() to evaluate your grammar

    # don't know how this should be formatted?
    grammar = r"""
                NP: {(<DT>|<PRP.>|<POS>)?<CD>*<JJ.*>*<CD>*<NN.*>+}
                    {<PRP>}
               """
    cp = nltk.RegexpParser(grammar)

    with open(filepath, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.reader(csv_in, delimiter=',')
            csv_writer = csv.writer(csv_out, delimiter=',')

            # write column headers
            row = csv_reader.next()
            row[-1] = 'CHUNKS'
            #row.append('CHUNKS')
            csv_writer.writerow(row)

            for row in csv_reader:
                # evaluating the string converts it back to list
                l = ast.literal_eval(row[-1])
                result = cp.parse(l)
                # can also draw the tree with result.draw()
                # convert to IOB tags
                #result = nltk.chunk.util.tree2conlltags(result)

                # TODO clean up text more, remove stop words
                # strip out any punctuation at this point, chunking works better if some punctuation is left in
                #result = [(x, y, z) for (x, y, z) in result if re.search(r'\w+', y)]

                # write row
                row[-1] = result
                csv_writer.writerow(row)

if __name__ == '__main__':
    chunk('data/sentences_POS.csv', 'data/sentences_chunk.csv')
