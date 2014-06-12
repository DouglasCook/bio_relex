""" This module will be used to record essential tidbits on how to use NLTK """
from __future__ import division     # use python3 division
import nltk
import re                           # regular expression module
import pprint                       # pretty printing!

#from nltk.corpus import webtext      # webtext is web stuff
#from nltk.corpus import brown        # Brown corpus contains 500 texts from various genres
#from nltk.corpus import reuters      # includes training and test sets from news documents


def gutenberg():
    """ Gutenberg contains books 
        This shows some of the annotations included in the NLTK corpora """
    from nltk.corpus import gutenberg 

    # loop through all files in corpus
    for fileid in gutenberg.fileids():
        # raw treats whole text as char string 
        num_chars = len(gutenberg.raw(fileid))
        # words and sents depend on the corpus being annotated 
        num_words = len(gutenberg.words(fileid))
        num_sents = len(gutenberg.sents(fileid))
        num_vocab = len(set([w.lower for w in gutenberg.words(fileid)]))

        print int(num_chars/num_words), int(num_words/num_sents), int(num_words/num_vocab), fileid


def inaugural():
    """ Inaugural speeches given by presidents 
        This calculates a conditional frequency distribution """
    from nltk.corpus import inaugural

    # plots a conditional frequency distribution
    cfd = nltk.ConditionalFreqDist(
            # this dictates what is plotted and the x axis
            (target, fileid[:4])
            for fileid in inaugural.fileids()
            for w in inaugural.words(fileid)
            for target in ['america', 'citizen']
            if w.lower().startswith(target))

    # show results in a table
    cfd.tabulate()
    # or graph
    cfd.plot()


def own_corpora():
    """ How to use your own corpora """
    # import plaintext utilities
    from nltk.corpus import PlaintextCorpusReader

    # provide the root of directory for corpus
    corpus_root = 'DDFsamples'
    wordlists = PlaintextCorpusReader(corpus_root, '.*')

    print 'Files are', wordlists.fileids()


def names():
    names = nltk.corpus.names
    males = names.words('male.txt')
    females = names.words('female.txt')

    print len(males), 'male and', len(females), 'female names'

    cfd = nltk.ConditionalFreqDist(
            (f, n[0])
            for f in names.fileids()
            for n in names.words(f)) 

    cfd.plot()

def wordnet(word):
    """ English wordnet is installed with the NLTK data """
    from nltk.corpus import wordnet as wn

    print 'SYNONYMS', wn.synsets(word)
    print 'DEFINITION', wn.synset('knock.n.03').definition
    print 'EXAMPLES', wn.synset('knock.n.03').examples
    print 'LEMMA NAMES', wn.synset('knock.n.03').lemma_names


def online_resources():
    """ Accessing online resources """
    # require urlopen to be able to access online resources
    from urllib import urlopen

    url = "http://www.gutenberg.org/files/2554/2554.txt"
    raw = urlopen(url).read()

    # split the string into tokens, words and punctuation
    tokens = nltk.word_tokenize(raw)

    # now can convert the tokens into NLTK text
    # before computing anything may want to slice off irrelevant headers/footers
    text = nltk.Text(tokens)
    print text.collocations()

if __name__ == '__main__':
    online_resources()
