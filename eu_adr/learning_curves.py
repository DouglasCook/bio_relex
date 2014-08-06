import sqlite3
import operator
import random
import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from app.feature_extractor import FeatureExtractor
from app.utility import time_stamped


def learning_curves(repeats):
    """
    Plot learning curve thingies
    """
    clf = build_pipeline()

    data, labels = load_features_data(eu_adr_only=False)
    # convert from dict into np array
    vec = DictVectorizer()
    data = vec.fit_transform(data).toarray()

    samples_per_split = 0.09*len(data)
    scores = np.zeros(shape=(repeats, 10, 3, 2))
    accuracy = np.zeros(shape=(repeats, 10))

    for i in xrange(repeats):
        #scores[i], accuracy[i] = get_data_points(clf, data, labels, i)
        #scores[i], accuracy[i] = uncertainty_sampling(clf, data, labels, i)
        scores[i], accuracy[i] = random_sampling(clf, data, labels, i)

    # now need to average it out somehow
    av_scores = scores.mean(axis=0)
    av_accuracy = accuracy.mean(axis=0)
    draw_plots(av_scores, av_accuracy, samples_per_split)

    #pickle.dump(scores, open('scores.p', 'wb'))
    #pickle.dump(av_scores, open('av_scores.p', 'wb'))


def load_features_data(eu_adr_only=False):
    """
    Load some part of data
    """
    # set up feature extractor with desired features
    extractor = FeatureExtractor(word_gap=True, count_dict=True, phrase_count=True, word_features=True)
    with sqlite3.connect('database/euadr_biotext.db') as db:
        # using Row as row factory means can reference fields by name instead of index
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

    # may only want to look at sentences from eu-adr to start with
    if eu_adr_only:
        cursor.execute('''SELECT relations.*
                              FROM relations NATURAL JOIN sentences
                              WHERE sentences.source = 'eu-adr';''')
    else:
        # want to create features for all relations in db, training test split will be done by scikit-learn
        cursor.execute('''SELECT relations.*
                          FROM relations NATURAL JOIN sentences
                          WHERE sentences.source != 'pubmed';''')

    records = cursor.fetchall()
    feature_vectors, class_vector = extractor.generate_features(records, balance_classes=False)

    return feature_vectors, class_vector


def build_pipeline():
    """
    Set up classfier here to avoid repetition
    """
    # TODO what type of kernel to use?
    clf = Pipeline([('normaliser', preprocessing.Normalizer(norm='l2')),
                    #('svm', SVC(kernel='rbf', gamma=10))])
                    #('svm', SVC(kernel='sigmoid'))])
                    ('svm', SVC(kernel='poly', coef0=1, degree=2, gamma=1, cache_size=1000))])
                    #('svm', SVC(kernel='poly', coef0=1, degree=2, gamma=1, cache_size=1000, class_weight='auto'))])
                    #('svm', SVC(kernel='rbf', gamma=10, cache_size=1000))])
                    #('svm', SVC(kernel='rbf', gamma=10, cache_size=1000, class_weight='auto'))])
                    #('svm', SVC(kernel='linear'))])
                    #('random_forest', RandomForestClassifier(n_estimators=10, max_features='sqrt', bootstrap=False,
                                                             #n_jobs=-1))])
    return clf


def random_sampling(clf, data, labels, j):
    """
    Get set of data points for one curve
    Want to add to the training data incrementally to mirror real life situtation
    Folds are picked randomly
    """
    # set up array to hold scores
    scores = np.zeros(shape=(10, 3, 2))
    accuracy = np.zeros(shape=10)

    # first take off 10% for testing
    remaining_data, test_data, remaining_labels, test_labels = train_test_split(data, labels, train_size=0.9,
                                                                                random_state=3*j)

    # now take first split for training
    train_data, remaining_data, train_labels, remaining_labels = train_test_split(remaining_data, remaining_labels,
                                                                                  train_size=0.1,
                                                                                  random_state=3*j)
    no_samples = len(train_data) - 1
    scores[0], accuracy[0] = get_scores(clf, train_data, train_labels, test_data, test_labels)

    # now loop to create remaining training sets
    for i in xrange(1, 10):
        more_data, remaining_data, more_labels, remaining_labels = train_test_split(remaining_data, remaining_labels,
                                                                                    train_size=no_samples,
                                                                                    random_state=None)
        # add the new training data to existing
        train_data = np.append(train_data, more_data, axis=0)
        train_labels = np.append(train_labels, more_labels)
        scores[i], accuracy[i] = get_scores(clf, train_data, train_labels, test_data, test_labels)

    return scores, accuracy


def uncertainty_sampling(clf, data, labels, j):
    """
    Get set of data points for one curve
    Want to add to the training data incrementally to mirror real life situtation
    Samples the classifier is least confident about predicting are selected first
    """
    # set up array to hold scores
    scores = np.zeros(shape=(10, 3, 2))
    accuracy = np.zeros(shape=10)

    # first take off 10% for testing
    remaining_data, test_data, remaining_labels, test_labels = train_test_split(data, labels, train_size=0.9,
                                                                                random_state=3*j)

    # now take first split for training
    train_data, remaining_data, train_labels, remaining_labels = train_test_split(remaining_data, remaining_labels,
                                                                                  train_size=0.1,
                                                                                  random_state=3*j)
    no_samples = len(train_data) - 1
    scores[0], accuracy[0] = get_scores(clf, train_data, train_labels, test_data, test_labels)

    # now loop to create remaining training sets
    for i in xrange(1, 10):
        # calculate uncertainty of classifier on remaining data
        # absolute value so both classes are considered
        confidence = [abs(x) for x in clf.decision_function(remaining_data)]

        # zip it all together and order by confidence
        remaining = sorted(zip(confidence, remaining_data, remaining_labels), key=operator.itemgetter(0))
        #print 'remaining', len(remaining)
        confidence, remaining_data, remaining_labels = zip(*remaining)

        # add the new training data to existing
        train_data = np.append(train_data, remaining_data[:no_samples], axis=0)
        train_labels = np.append(train_labels, remaining_labels[:no_samples])

        remaining_data = np.array(remaining_data[no_samples:])
        remaining_labels = np.array(remaining_labels[no_samples:])

        scores[i], accuracy[i] = get_scores(clf, train_data, train_labels, test_data, test_labels)

    return scores, accuracy


def get_data_points(clf, data, labels, j):
    """
    Get set of data points for one curve
    Want to add to the training data incrementally to mirror real life situtation
    Fold are picked randomly
    """
    # set up array to hold scores
    scores = np.zeros(shape=(9, 3, 2))
    accuracy = np.zeros(shape=9)

    # first split at 10%
    train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(data, labels, train_size=0.1,
                                                                                         #random_state=j)
                                                                                         random_state=None)
    no_samples = len(train_data)
    scores[0], accuracy[0] = get_scores(clf, train_data, train_labels, test_data, test_labels)

    # now loop to create remaining training sets
    for i in xrange(1, 9):
        more_data, test_data, more_labels, test_labels = cross_validation.train_test_split(test_data, test_labels,
                                                                                           train_size=no_samples,
                                                                                           #random_state=i*j)
                                                                                           random_state=None)
        # add the new training data to existing
        train_data = np.append(train_data, more_data, axis=0)
        train_labels = np.append(train_labels, more_labels)
        scores[i], accuracy[i] = get_scores(clf, train_data, train_labels, test_data, test_labels)

    return scores, accuracy


def nicer_get_data_points(clf, data, labels, j):
    """
    Get set of data points for one curve
    Want to add to the training data incrementally to mirror real life situtation
    """
    # TODO problem here is with the random state, how can I make the experiment repeatable?
    # set up arrays to hold scores and indices
    scores = np.zeros(shape=(9, 3, 2))
    accuracy = np.zeros(shape=9)
    train_indices = np.array([], dtype=int)

    # set up stratified 10 fold cross validator, use specific random state for proper comparison
    # passing specific random state in here means same split is used every time
    cv = cross_validation.StratifiedKFold(labels, shuffle=True, n_folds=10, random_state=None)

    # iterating through the cv gives lists of indices for each fold
    # use test set for training since it is 10% of total data
    # TODO better way to do this?
    for i, (_, train) in enumerate(cv):
        if i == 9:
            print train_indices
            break
        # first add new fold to existing for training data and labels
        train_indices = np.append(train_indices, train)
        train_data = data[train_indices]
        train_labels = labels[train_indices]

        # then use complement for testing
        test_data = np.delete(data, train_indices, 0)
        test_labels = np.delete(labels, train_indices, 0)

        scores[i], accuracy[i] = get_scores(clf, train_data, train_labels, test_data, test_labels)

    return scores, accuracy


def get_scores(clf, train_data, train_labels, test_data, test_labels):
    """
    Return array of scores
    """
    # train model
    clf.fit(train_data, train_labels)
    # calculate mean accuracy since not included in other set of scores
    accuracy = clf.score(test_data, test_labels)
    # classify the test data
    predicted = clf.predict(test_data)
    # evaluate accuracy of output compared to correct classification
    scores = precision_recall_fscore_support(test_labels, predicted)

    # return precision, recall and f1
    return np.array([scores[0], scores[1], scores[2]]), accuracy


def draw_plots(scores, av_accuracy, samples_per_split):
    """
    Create plots for precision, recall and f-score
    """
    #scores = pickle.load(open('av_scores.p', 'rb'))
    false_p = [s[0][0] for s in scores]
    true_p = [s[0][1] for s in scores]
    false_r = [s[1][0] for s in scores]
    true_r = [s[1][1] for s in scores]
    false_f = [s[2][0] for s in scores]
    true_f = [s[2][1] for s in scores]

    # create ticks for x axis
    ticks = np.linspace(samples_per_split, 10*samples_per_split, 10)

    '''
    plot(ticks, true_p, false_p, 'Precision', 'plots/' + time_stamped('precision_2j_random.png'))
    plot(ticks, true_r, false_r, 'Recall', 'plots/' + time_stamped('recall_2j_random.png'))
    plot(ticks, true_f, false_f, 'F-score', 'plots/' + time_stamped('fscore_2j_random.png'))
    plot(ticks, av_accuracy, None, 'Accuracy', 'plots/' + time_stamped('accuracy_2j_random.png'))
    '''

    plot(ticks, true_p, false_p, 'Precision', 'plots/uncertainty_comparison/precision_3j_random.png')
    plot(ticks, true_r, false_r, 'Recall', 'plots/uncertainty_comparison/recall_3j_random.png')
    plot(ticks, true_f, false_f, 'F-score', 'plots/uncertainty_comparison/fscore_3j_random.png')
    plot(ticks, av_accuracy, None, 'Accuracy', 'plots/uncertainty_comparison/accuracy_3j_random.png')


def plot(ticks, true, false, scoring, filepath):
    """
    Plot given values
    """
    # set up the figure
    plt.figure()
    plt.grid()
    plt.xlabel('Training Instances')
    plt.ylabel('Score')
    plt.title(scoring)

    # if false not none then we are dealing with normal scores
    if false:
        # plot raw data points
        plt.plot(ticks, true, label='True relations')
        plt.plot(ticks, false, label='False relations')
    # else must be accuracy
    else:
        plt.plot(ticks, true, label='Average accuracy')

    # now fit polynomial (straight line) to the points and extend plot out
    x_new = np.linspace(ticks.min(), 2*ticks.max())
    true_coefs = np.polyfit(ticks, true, deg=1)
    true_fitted = np.polyval(true_coefs, x_new)
    plt.plot(x_new, true_fitted)

    # only plot false if not on accuracy score
    if false:
        false_coefs = np.polyfit(ticks, false, deg=1)
        false_fitted = np.polyval(false_coefs, x_new)
        plt.plot(x_new, false_fitted)

    plt.legend(loc='best')

    plt.savefig(filepath, format='png')
    plt.clf()

if __name__ == '__main__':
    learning_curves(repeats=20)
