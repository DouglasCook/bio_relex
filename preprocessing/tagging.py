import nltk
import sys

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


if __name__ == '__main__':
    clean_and_tag()
