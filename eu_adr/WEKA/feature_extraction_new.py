import csv
import re

import nltk


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
    words = [t[0] for t in tags if not re.match('.?\d', t[0])]
    # zip up the list with offset list to create bigrams
    bigrams = ['-'.join([b[0], b[1]]) for b in zip(words, words[1:])]

    bigrams = '"' + ' '.join(bigrams) + '"'
    words = '"' + ' '.join(words) + '"'

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
        return [words, bigrams, pos, phrase_path, combo, nps, vps, pps], word_gap
    else:
        return [words, bigrams, pos, phrase_path, combo, nps, vps, pps]


def generate_features(stem=False):
    """
    Create basic feature vector for each record
    """
    # set filepath to input
    if stem:
        file_in = 'csv/tagged_sentences_stemmed.csv'
    else:
        file_in = 'csv/tagged_sentences.csv'

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
                             word_gap,
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
    generate_features()
