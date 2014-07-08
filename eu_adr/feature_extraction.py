import os
import sys
import csv

import nltk


class BigramChunker(nltk.ChunkParserI):
    """
    Bigram based chunker
    Should give reasonable results and require much less work than regex based one
    Actually doesn't work particularly well... maybe should use Stanford instead?
    """

    def __init__(self, train_sents):
        """
        The constructor takes a training data set and trains the classifier
        """
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)

    def parse(self, sentence):
        """
        Takes POS tagged sentence and returns a chunk tree
        """
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag) in zip(sentence, chunktags)]
        #return nltk.chunk.util.conlltags2tree(conlltags)
        return conlltags


def set_up_chunker():
    """
    Return trained chunker
    """
    # other option is the treebank chunk corpus
    train_sents = nltk.corpus.conll2000.chunked_sents('train.txt')
    return BigramChunker(train_sents)


def tagging():
    """
    Tags and chunk words between the two entities
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, 'csv/relevant_sentences.csv'))
    file_out = os.path.abspath(os.path.join(basepath, 'csv/tagged_sentences.csv'))

    chunker = set_up_chunker()
    stopwords = nltk.corpus.stopwords.words('english')

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, delimiter=',')
            csv_writer = csv.DictWriter(csv_out, ['pid', 'true_relation', 'e1', 'e2', 'start1', 'end1', 'start2',
                                                  'end2', 'sent_num', 'sentence', 'between', 'between_tags'],
                                        delimiter=',')
            csv_writer.writeheader()

            for row in csv_reader:
                # display progress bar
                sys.stdout.write('.')
                sys.stdout.flush()

                between = nltk.word_tokenize(row['between'])
                #between = [b for b in between if b not in stopwords]
                tags = nltk.pos_tag(between)
                chunks = chunker.parse(tags)
                row.update({'between_tags': chunks})
                csv_writer.writerow(row)

if __name__ == '__main__':
    tagging()
