import sqlite3
import operator
import pickle
from time import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from app.feature_extractor import FeatureExtractor
from app.utility import time_stamped

# TODO does this make sense? global thing here?
vec = DictVectorizer()
db_path = 'database/relex.db'


def load_records():
    """
    Load original and new data sets
    """
    with sqlite3.connect(db_path) as db:
        # using Row as row factory means can reference fields by name instead of index
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        # all records from original corpus
        cursor.execute('''SELECT relations.*
                          FROM relations NATURAL JOIN sentences
                          WHERE sentences.source != 'pubmed';''')

        orig = cursor.fetchall()

        # now all newly annotated records
        cursor.execute('''SELECT relations.*
                          FROM relations NATURAL JOIN sentences
                          WHERE sentences.source = 'pubmed' AND
                                true_rel IS NOT NULL;''')

        new = cursor.fetchall()

    return orig, new


def build_pipeline():
    """
    Set up classfier here to avoid repetition
    """
    clf = Pipeline([('normaliser', preprocessing.Normalizer(norm='l2')),
                    #('svm', SVC(kernel='rbf', gamma=10))])
                    #('svm', SVC(kernel='sigmoid'))])
                    #('svm', SVC(kernel='poly', coef0=1, degree=2, gamma=1, cache_size=2000, C=1000))])
                    #('svm', SVC(kernel='poly', coef0=1, degree=3, gamma=2, cache_size=2000, C=10000))])
                    ('svm', SVC(kernel='rbf', gamma=30, cache_size=1000, C=1000))])
    #('svm', SVC(kernel='linear'))])
    #('random_forest', RandomForestClassifier(n_estimators=10, max_features='sqrt', bootstrap=False,
    #n_jobs=-1))])
    return clf


def random_sampling(clf, extractor, orig_records, new_records, train_indices, test_indices, splits):
    """
    Get set of data points for one curve
    Want to add to the training data incrementally to mirror real life situtation
    Folds are picked randomly
    """
    # set up array to hold scores
    scores = np.zeros(shape=(splits, 3, 2))
    accuracy = np.zeros(shape=splits)
    no_samples = len(train_indices)/splits

    # now take first split for training and leave remainder
    # NEED TO TAKE REST FIRST SINCE TRAIN WILL CHANGE
    rest_indices = train_indices[no_samples:]
    train_indices = train_indices[:no_samples]

    scores[0], accuracy[0] = get_scores(clf, extractor, orig_records, new_records, train_indices, test_indices)

    # now loop to create remaining training sets
    for i in xrange(1, splits):
        # add the new training data to existing
        # NOW NEED TO TAKE TRAIN FIRST SINCE REST WILL CHANGE
        train_indices = np.append(train_indices, rest_indices[:no_samples])
        rest_indices = rest_indices[no_samples:]
        #print 'random', len(train_indices)
        scores[i], accuracy[i] = get_scores(clf, extractor, orig_records, new_records, train_indices, test_indices)

    return scores, accuracy


def uncertainty_sampling(clf, extractor, orig_records, new_records, train_indices, test_indices, splits):
    """
    Get set of data points for one curve
    Want to add to the training data incrementally to mirror real life situtation
    Samples the classifier is least confident about predicting are selected first
    """
    # set up array to hold scores
    scores = np.zeros(shape=(splits, 3, 2))
    accuracy = np.zeros(shape=splits)
    no_samples = len(train_indices)/splits

    # now take first split for training and leave remainder
    # NEED TO TAKE REST FIRST SINCE TRAIN WILL CHANGE
    rest_indices = train_indices[no_samples:]
    train_indices = train_indices[:no_samples]

    scores[0], accuracy[0], rest_data = get_scores(clf, extractor, orig_records, new_records, train_indices,
                                                   test_indices, rest_indices)

    # now loop to create remaining training sets
    for i in xrange(1, splits):
        # calculate uncertainty of classifier on remaining data
        confidence = clf.decision_function(rest_data).flatten()
        # absolute value so both classes are considered
        confidence = np.absolute(confidence)

        # zip it all together and order by confidence
        remaining = sorted(zip(confidence, rest_indices), key=operator.itemgetter(0))
        #print 'remaining', len(remaining)
        confidence, rest_indices = zip(*remaining)

        # add the new training data to existing
        train_indices = np.append(train_indices, rest_indices[:no_samples])
        rest_indices = rest_indices[no_samples:]
        #print 'uncertainty', len(train_indices)

        if len(rest_indices) > 0:
            scores[i], accuracy[i], rest_data = get_scores(clf, extractor, orig_records, new_records, train_indices,
                                                           test_indices, rest_indices)
        else:
            scores[i], accuracy[i] = get_scores(clf, extractor, orig_records, new_records, train_indices, test_indices,
                                                None)

    return scores, accuracy


def density_sampling(clf, extractor, orig_records, new_records, train_indices, test_indices, sim, splits):
    """
    Get set of data points for one curve
    Want to add to the training data incrementally to mirror real life situtation
    Samples selected based on confidence measure weighted by similarity to other samples
    """
    # set up array to hold scores
    scores = np.zeros(shape=(splits, 3, 2))
    accuracy = np.zeros(shape=splits)
    no_samples = len(train_indices)/splits

    # now take first split for training and leave remainder
    # NEED TO TAKE REST FIRST SINCE TRAIN WILL CHANGE
    rest_indices = train_indices[no_samples:]
    train_indices = train_indices[:no_samples]

    scores[0], accuracy[0], rest_data = get_scores(clf, extractor, orig_records, new_records, train_indices,
                                                   test_indices, rest_indices)

    # now loop to create remaining training sets
    for i in xrange(1, splits):
        # calculate uncertainty of classifier on remaining data
        confidence = clf.decision_function(rest_data).flatten()
        # absolute value so both classes are considered
        confidence = np.absolute(confidence)

        # TODO may want to scale weighting between uncertainty and similarity score
        rest_sim = sim[np.array(rest_indices)]
        #rest_sim **= 0.8

        # weigh the confidence based on similarity measure
        #confidence = np.multiply(confidence, rest_sim)
        confidence = np.divide(confidence, rest_sim)
        # zip it all together and order by confidence
        remaining = sorted(zip(confidence, rest_indices), key=operator.itemgetter(0))
        #print 'remaining', len(remaining)
        confidence, rest_indices = zip(*remaining)

        # add the new training data to existing
        train_indices = np.append(train_indices, rest_indices[:no_samples])
        rest_indices = rest_indices[no_samples:]
        #print 'density', len(train_indices)

        if len(rest_indices) > 0:
            scores[i], accuracy[i], rest_data = get_scores(clf, extractor, orig_records, new_records, train_indices,
                                                           test_indices, rest_indices)
        else:
            scores[i], accuracy[i] = get_scores(clf, extractor, orig_records, new_records, train_indices, test_indices,
                                                None)

    return scores, accuracy


def get_similarities(vectors):
    """
    Calculate similarities of vectors ie one to all others
    """
    print 'calculating similarities'
    similarities = np.zeros(len(vectors))
    for i, v in enumerate(vectors):
        print i
        total = 0
        others = np.delete(vectors, i, 0)

        # loop through all other vectors and get total cosine distance
        for x in others:
            total += distance.cosine(v, x)

        # cos_similarity = 1 - av_cos_dist
        similarities[i] = 1 - total/len(others)
    print 'finished calculating similarities'

    return similarities


def pickle_similarities():
    """
    Pickle similarities for newly annotated data only
    """
    # TODO this is kind of wrong since the similarities will change as the word features are generated per split
    orig_records, new_records = load_records()
    all_records = orig_records + new_records
    orig_length = len(orig_records)

    # set up extractor using desired features
    extractor = FeatureExtractor(word_gap=True, count_dict=True, phrase_count=True, word_features=5)
    extractor.create_dictionaries(all_records, how_many=5)

    data, _ = extractor.generate_features(all_records)
    data = vec.fit_transform(data).toarray()
    similarities = get_similarities(data)

    # only want to pickle new data since orig always used for training
    pickle.dump(similarities[orig_length:], open('pickles/similarities_all.p', 'wb'))


def get_scores(clf, extractor, orig_records, new_records, train_indices, test_indices, rest_indices=None):
    """
    Return array of scores
    """
    # add sample of new records to original records (always used for training)
    train_records = [new_records[i] for i in train_indices] + orig_records
    test_records = [new_records[i] for i in test_indices]

    # word features must be selected based on training set only otherwise test data contaminates training set
    extractor.create_dictionaries(train_records, how_many=5)

    train_data, train_labels = extractor.generate_features(train_records)
    test_data, test_labels = extractor.generate_features(test_records)

    # need to smash everything together before generating features so same features are generated for each set
    if rest_indices is None:
        train_length = len(train_data)
        data = vec.fit_transform(train_data + test_data).toarray()
        train_data = data[:train_length]
        test_data = data[train_length:]
    else:
        rest_records = [new_records[i] for i in rest_indices]
        rest_data, _ = extractor.generate_features(rest_records)
        train_length = len(train_data)
        test_length = len(test_data)
        data = vec.fit_transform(train_data + test_data + rest_data).toarray()
        train_data = data[:train_length]
        test_data = data[train_length:train_length + test_length]
        rest_data = data[train_length + test_length:]

    # train model
    clf.fit(train_data, train_labels)
    # calculate mean accuracy since not included in other set of scores
    accuracy = clf.score(test_data, test_labels)
    # classify the test data
    predicted = clf.predict(test_data)
    # evaluate accuracy of output compared to correct classification
    scores = precision_recall_fscore_support(test_labels, predicted)

    # for non random sampling need to return remaining data so confidence can be measured
    if rest_indices is not None:
        return np.array([scores[0], scores[1], scores[2]]), accuracy, rest_data

    else:
        return np.array([scores[0], scores[1], scores[2]]), accuracy


def draw_learning_comparison(splits, r_score, u_score, d_score, samples_per_split, repeats, scoring):
    """
    Plot the different learning methods on same graph
    """
    # create ticks for x axis
    ticks = np.linspace(samples_per_split, splits*samples_per_split, splits)

    # set up the figure
    plt.figure()
    plt.grid()
    plt.xlabel('Training Instances')
    plt.ylabel(scoring)
    plt.title('%s Comparison using %s batches and %s repeats' % (scoring, splits, repeats))

    plt.plot(ticks, r_score, label='Random Sampling')
    plt.plot(ticks, u_score, label='Uncertainty Sampling')
    plt.plot(ticks, d_score, label='Density Sampling')

    plt.legend(loc='best')

    plt.savefig('plots/new_learning_comparison_' + scoring + '_' + time_stamped('.png'), format='png')
    plt.clf()


def learning_method_comparison(splits, repeats, seed):
    """
    Plot learning curves to compare accuracy of different learning methods
    """
    clf = build_pipeline()
    # set up extractor using desired features
    extractor = FeatureExtractor(word_gap=True, count_dict=True, phrase_count=True, word_features=True)

    # orig will always be use for training, new will be used for testing and added incrementally
    orig_records, new_records = load_records()

    # TODO what is the deal here???
    # this needs to match whatever percentage is being used for testing
    # samples per split = number of records remaining after removing test set divided by number of splits
    #samples_per_split = (0.8/splits) * len(new)
    samples_per_split = 4 * len(new_records)/(5 * splits)
    #print 'samples per split', samples_per_split

    # if using density sampling only want to calculate similarities once
    sim = pickle.load(open('pickles/similarities_all.p', 'rb'))

    r_scores = np.zeros(shape=(repeats, splits, 3, 2))
    u_scores = np.zeros(shape=(repeats, splits, 3, 2))
    d_scores = np.zeros(shape=(repeats, splits, 3, 2))

    r_accuracy = np.zeros(shape=(repeats, splits))
    u_accuracy = np.zeros(shape=(repeats, splits))
    d_accuracy = np.zeros(shape=(repeats, splits))

    # loop number of times to generate average scores
    for i in xrange(repeats):
        print i
        # going to split the data here, then pass identical indices to the different learning methods
        all_indices = np.arange(len(new_records))

        # seed the shuffle here so can repeat experiment for different numbers of splits
        np.random.seed(seed * i)
        np.random.shuffle(all_indices)

        # take off 20% for testing
        # done this awkward way so the training set will be of a nice fixed size, rounding errors go into test set
        # this means that learning curves will always start and finish at same points
        test_indices = all_indices[4*(len(new_records)/5):]
        #print 'testing', len(test_indices)
        train_indices = all_indices[:4*(len(new_records)/5)]
        #print 'training', len(train_indices)

        # now use same test and train indices to generate scores for each learning method
        u_scores[i], u_accuracy[i] = uncertainty_sampling(clf, extractor, orig_records, new_records, train_indices,
                                                          test_indices, splits)
        r_scores[i], r_accuracy[i] = random_sampling(clf, extractor, orig_records, new_records, train_indices,
                                                     test_indices, splits)
        d_scores[i], d_accuracy[i] = density_sampling(clf, extractor, orig_records, new_records, train_indices,
                                                      test_indices, sim, splits)

    # create array of scores to pass to plotter
    scores = [['Accuracy'], ['Precision'], ['Recall'], ['F-Score']]
    # accuracy scores
    scores[0].append(r_accuracy.mean(axis=0, dtype=np.float64))
    scores[0].append(u_accuracy.mean(axis=0, dtype=np.float64))
    scores[0].append(d_accuracy.mean(axis=0, dtype=np.float64))

    # average over the repeats
    r_scores = r_scores.mean(axis=0, dtype=np.float64)
    u_scores = u_scores.mean(axis=0, dtype=np.float64)
    d_scores = d_scores.mean(axis=0, dtype=np.float64)
    # then true and false
    r_scores = r_scores.mean(axis=2, dtype=np.float64)
    u_scores = u_scores.mean(axis=2, dtype=np.float64)
    d_scores = d_scores.mean(axis=2, dtype=np.float64)

    # using numpy slicing to select correct scores
    for i in xrange(3):
        scores[i+1].append(r_scores[:, i])
        scores[i+1].append(u_scores[:, i])
        scores[i+1].append(d_scores[:, i])

    f_name = 'pickles/newCurves_seed%s_splits%s.p' % (seed, splits)
    pickle.dump(scores, open(f_name, 'wb'))

    for i in xrange(4):
        draw_learning_comparison(splits, scores[i][1], scores[i][2], scores[i][3], samples_per_split, repeats,
                                 scores[i][0])


if __name__ == '__main__':
    #pickle_similarities()
    start = time()
    #learning_method_comparison(repeats=10, splits=5)
    learning_method_comparison(repeats=1, splits=5, seed=1)
    #learning_method_comparison(repeats=20, splits=10, seed=1)
    #learning_method_comparison(repeats=20, splits=20, seed=1)
    #learning_method_comparison(repeats=20, splits=40)
    end = time()
    print 'running time =', end - start

    #start = time()
    #learning_method_comparison(repeats=2, splits=5)
    #end = time()
    #print 'running time =', end - start
