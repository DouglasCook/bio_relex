import re
import nltk
import pickle
import sqlite3
import random


class FeatureExtractor():
    """
    Class for extracting features from previously tagged and chunked sets of words
    """
    def __init__(self, word_features=False, word_gap=True, count_dict=True, phrase_count=True):
        """
        Store variables for which features to use
        """
        # stopwords for use in cleaning up sentences
        self.stopwords = nltk.corpus.stopwords.words('english')

        # types of features
        self.word_features = word_features
        self.word_gap = word_gap
        self.count_dict = count_dict
        self.phrase_count = phrase_count

    def generate_features(self, records, no_class=False, balance_classes=False):
        """
        Generate feature vectors and class labels for given records
        If no_class is true then the relation has not been annotated so don't return a class vector
        """
        feature_vectors = []
        class_vector = []

        for row in records:
            # need to evaluate field if coming from pickled data
            if not no_class:
                try:
                    if eval(row['true_rel']):
                        class_vector.append(1)
                    else:
                        class_vector.append(0)
                # otherwise can use it straight?
                except:
                    if row['true_rel']:
                        class_vector.append(1)
                    else:
                        class_vector.append(0)

            # don't think there's any need to have both types since one implies the other
            #f_vector = {'type1': row['type1'], 'type2': row['type2']}
            f_vector = {'type1': row['type1']}
            # now add the features for each part of text
            f_vector.update(self.part_feature_vectors(eval(row['before_tags']), 'before'))
            f_vector.update(self.part_feature_vectors(eval(row['between_tags']), 'between'))
            f_vector.update(self.part_feature_vectors(eval(row['after_tags']), 'after'))

            # now add whole dictionary to list
            feature_vectors.append(f_vector)

        if no_class:
            return feature_vectors
        # if a class is to be undersampled
        elif balance_classes:
            feature_vectors, class_vector = self.balance_classes(feature_vectors, class_vector)

        return feature_vectors, class_vector

    def part_feature_vectors(self, tags, which_set):
        """
        Generate features for a set of words, before, between or after
        """
        f_dict = {}

        # TODO is word gap actually useful? maybe for between only?
        if self.word_gap:
            f_dict[which_set] = len(tags)
        # add word gap for between words
        #if which_set == 'between':
            #f_dict['word_gap'] = len(tags)

        # remove stopwords and things not in chunks
        tags = [t for t in tags if t[0] not in self.stopwords and t[2] != 'O']

        # TODO see if some sort of word features could be added back in
        # WORDS - remove numbers here, they should not be considered when finding most common words
        #       - maybe also want to remove proper nouns?
        #words = [t[0] for t in tags if not re.match('.?\d', t[0])]
        # zip up the list with offset list to create bigrams
        #bigrams = ['-'.join([b[0], b[1]]) for b in zip(words, words[1:])]

        #bigrams = '"' + ' '.join(bigrams) + '"'
        #words = '"' + ' '.join(words) + '"'

        if self.word_features:
            # WORDS - check for presence of particular words
            if which_set != 'after':
                words = [t[0] for t in tags if not re.match('.?\d', t[0])]
                f_dict.update(self.word_check(words, which_set))

        # POS - remove NONE tags here, seems to improve results slightly, shouldn't use untaggable stuff
        pos = [t[1] for t in tags if t[1] != '-NONE-']
        # use counting or non counting based on input parameter
        if self.count_dict:
            f_dict.update(self.counting_dict(pos))
        else:
            f_dict.update(self.non_counting_dict(pos))
        #pos = '"' + ' '.join(pos) + '"'

        # CHUNKS - only consider beginning tags of phrases
        phrases = [t[2] for t in tags if t[2] and t[2][0] == 'B']
        # TODO add some sort of bigram chunk features?
        #phrase_path = '"' + '-'.join([p[2:] for p in phrases]) + '"'

        # COMBO - combination of tag and phrase type
        # slice here to remove 'B-'
        combo = ['-'.join([t[1], t[2][2:]]) for t in tags if t[2]]
        # use counting or non counting based on input parameter
        if self.count_dict:
            f_dict.update(self.counting_dict(combo))
        else:
            f_dict.update(self.non_counting_dict(combo))
        #combo = '"' + ' '.join(['-'.join(combo) + '"'

        if self.phrase_count:
            # count number of each type of phrase
            f_dict['nps'] = sum(1 for p in phrases if p == 'B-NP')
            f_dict['vps'] = sum(1 for p in phrases if p == 'B-VP')
            f_dict['pps'] = sum(1 for p in phrases if p == 'B-PP')

        return f_dict

    @staticmethod
    def balance_classes(feature_vectors, class_vector):
        """
        Undersample the over-represented class so it contains same number of samples
        """
        true_count = sum(class_vector)
        false_count = len(class_vector) - true_count

        # zip together with the labels
        # TODO use list comprehension instead
        together = sorted(zip(class_vector, feature_vectors))
        # split into classes
        false = together[:false_count]
        true = together[false_count+1:]

        # undersample the over represented class
        if true_count < false_count:
            false = random.sample(false, true_count)
        elif false_count < true_count:
            true = random.sample(true, false_count)

        # put back together again and shuffle so the classes are not ordered
        print len(true), len(false)
        together = false + true
        random.shuffle(together)

        # unzip before returning
        classes, features = zip(*together)

        return features, classes

    @staticmethod
    def word_check(words, which_set):
        """
        Create features for words deemed particularly useful for classification dependent on part of sentence
        Words are already stemmed so list contains stemmed versions
        """
        if which_set == 'before':
            stem_list = ['conclus', 'effect', 'result', 'studi', 'use', 'background', 'compar', 'method',
                         'object']
        elif which_set == 'between':
            stem_list = ['effect', 'therapi', 'treat', 'treatment', 'efficac', 'reliev', 'relief', 'increas', 'reduc']
        else:
            stem_list = ['patient', 'studi']

        stem_dict = {}
        # don't need to do this since scikit does it for you
        # everything defaults to false
        #stem_dict = {stem: 0 for stem in stem_list}

        for stem in stem_list:
            # if the stem is found in the words set to true in dictionary
            if stem in words:
                stem_dict[stem] = 1

        #print stem_dict
        return stem_dict

    @staticmethod
    def counting_dict(tags):
        """
        Record counts of each tag present in tags and return as dictionary
        """
        c_dict = {}
        for t in tags:
            # if key exists increment count otherwise add the key
            if t in c_dict.keys():
                c_dict[t] += 1
            else:
                c_dict[t] = 1
        return c_dict

    @staticmethod
    def non_counting_dict(tags):
        """
        Record counts of each tag present in tags and return as dictionary
        """
        c_dict = {}
        for t in tags:
            # if key exists add it to dict
            if t not in c_dict.keys():
                c_dict[t] = 1
        return c_dict


def pickle_features(eu_adr_only=False):
    """
    Create basic feature vector for each record
    """
    # only do this with the original data, don't want things getting overwritten
    with sqlite3.connect('database/original_data.db') as db:
        # using Row as row factory means can reference fields by name instead of index
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        # may only want to look at sentences from eu-adr to start with
        if eu_adr_only:
            cursor.execute('''SELECT relations.*
                              FROM relations NATURAL JOIN sentences
                              WHERE sentences.source = 'EU-ADR';''')
        else:
            # want to create features for all relations in db, training test split will be done by scikit-learn
            cursor.execute('''SELECT relations.*
                              FROM relations NATURAL JOIN sentences
                              WHERE sentences.source != 'pubmed';''')

        records = cursor.fetchall()
        extractor = FeatureExtractor()
        feature_vectors, class_vector = extractor.generate_features(records)

    if eu_adr_only:
        pickle.dump(feature_vectors, open('pickles/scikit_data_eu_adr_only.p', 'wb'))
        pickle.dump(class_vector, open('pickles/scikit_target_eu_adr_only.p', 'wb'))
    else:
        pickle.dump(feature_vectors, open('pickles/scikit_data.p', 'wb'))
        pickle.dump(class_vector, open('pickles/scikit_target.p', 'wb'))


if __name__ == '__main__':
    pickle_features(eu_adr_only=True)
    pickle_features(eu_adr_only=False)
    #f = pickle.load(open('pickles/scikit_data.p', 'rb'))
    #c = pickle.load(open('pickles/scikit_target.p', 'rb'))
    #print c
    #print f
    #for i in xrange(20):
        #print c[i], f[i]

