import os
import sys
import csv
import re

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
        # return nltk.chunk.util.conlltags2tree(conlltags)
        return conlltags


def set_up_chunker():
    """
    Return trained chunker
    """
    # other option is the treebank chunk corpus
    train_sents = nltk.corpus.conll2000.chunked_sents('train.txt')
    return BigramChunker(train_sents)


def pos_and_chunk_tags(text, chunker):
    """
    Return word, pos tag, chunk triples
    """
    # TODO remove undesirable words or tokens here? also remove CONCLUSION, METHODS etc since they aren't important
    text = nltk.word_tokenize(text)
    # text = [b for b in between if b not in stopwords]
    tags = nltk.pos_tag(text)
    chunks = chunker.parse(tags)

    # now want to remove any punctuation - maybe don't want to remove absolutely all punctuation?
    # match returns true without needing to match whole string
    chunks = [c for c in chunks if not re.match('\W', c[0])]

    return chunks


def tagging():
    """
    Tags and chunk words between the two entities
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, 'csv/relevant_sentences.csv'))
    file_out = os.path.abspath(os.path.join(basepath, 'csv/tagged_sentences.csv'))

    chunker = set_up_chunker()
    # don't think I actually want to remove stopwords?
    # stopwords = nltk.corpus.stopwords.words('english')

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            # set columns here so they can be more easily changed
            cols = ['pid',
                    'sent_num',
                    'true_relation',
                    'rel_type',
                    'e1',
                    'e2',
                    'type1',
                    'type2',
                    'start1',
                    'end1',
                    'start2',
                    'end2',
                    'sentence',
                    'before_tags',
                    'between_tags',
                    'after_tags',
                    'before',  # TODO get rid of these, need to change the write cols instead of using update
                    'between',
                    'after']
            csv_reader = csv.DictReader(csv_in, delimiter=',')
            csv_writer = csv.DictWriter(csv_out, cols, delimiter=',')
            csv_writer.writeheader()

            for row in csv_reader:
                # display progress bar
                sys.stdout.write('.')
                sys.stdout.flush()

                row.update({'before_tags': pos_and_chunk_tags(row['before'], chunker)})
                row.update({'between_tags': pos_and_chunk_tags(row['between'], chunker)})
                row.update({'after_tags': pos_and_chunk_tags(row['after'], chunker)})
                csv_writer.writerow(row)


def part_feature_vectors(tags, stopwords, count):
    """
    Generate features for a set of words, before, between or after
    """
    # TODO ask about punctuation, keep it in or chuck it out?
    # lets try removing stopwords and things not in chunks here
    tags = [t for t in tags if t[0] not in stopwords and t[2] != 'O']
    word_gap = len(tags)

    # WORDS - remove numbers here, they should not be considered when finding most common words
    #       - maybe also want to remove proper nouns?
    words = '"' + ' '.join([t[0] for t in tags if not re.match('.?\d', t[0])]) + '"'
    # POS - remove NONE tags here, seems to improve results slightly, shouldn't use untaggable stuff
    pos = '"' + ' '.join([t[1] for t in tags if t[1] != '-NONE-']) + '"'
    # CHUNKS - only consider beginning tags of phrases
    phrases = [t[2] for t in tags if t[2] and t[2][0] == 'B']
    # slice here to remove 'B-'
    phrase_path = '"' + '-'.join([p[2:] for p in phrases]) + '"'
    # COMBO - combination of tag and phrase type
    combo = '"' + ' '.join(['-'.join([t[1], t[2][2:]]) for t in tags if t[2]]) + '"'

    # count number of each type of phrase
    nps = sum(1 for p in phrases if p == 'B-NP')
    vps = sum(1 for p in phrases if p == 'B-VP')
    pps = sum(1 for p in phrases if p == 'B-PP')

    if count:
        return [words, pos, phrase_path, combo, nps, vps, pps], word_gap
    else:
        return [words, pos, phrase_path, combo, nps, vps, pps]


def generate_features():
    """
    Create basic feature vector for each record
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, 'csv/tagged_sentences.csv'))

    feature_vectors = []
    stopwords = nltk.corpus.stopwords.words('english')

    with open(file_in, 'rb') as csv_in:
        csv_reader = csv.DictReader(csv_in, delimiter=',')

        for row in csv_reader:
            f_vector = []
            # calculate these first to get the word gap
            between_features, word_gap = part_feature_vectors(eval(row['between_tags']), stopwords, True)
            # general features first
            # TODO add other features from reuters: phrase counts, stemmed words
            f_vector.extend([row['true_relation'],
                             row['sent_num'],
                             word_gap,
                             row['rel_type'],
                             row['type1'],
                             row['type2']])

            # now add the features for each part of text
            f_vector.extend(part_feature_vectors(eval(row['before_tags']), stopwords, False))
            f_vector.extend(between_features)
            f_vector.extend(part_feature_vectors(eval(row['after_tags']), stopwords, False))

            # now add whole vector to set
            feature_vectors.append(f_vector)

    return feature_vectors


if __name__ == '__main__':
    tagging()
    #generate_features()
