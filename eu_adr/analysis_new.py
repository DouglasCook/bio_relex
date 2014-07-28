import sqlite3

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from eu_adr.app import scikit_feature_extraction


def load_data(eu_adr_only=False):
    """
    Load some part of data
    Biotext instances are at end of data set so will be sliced off and balance set
    """
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
    feature_vectors, class_vector = scikit_feature_extraction.generate_features(records)

    return feature_vectors, class_vector


def learning_curves(repeats=10):
    """
    Plot learning curve thingies
    """
    features, labels = load_data(eu_adr_only=False)
    # convert from dict into np array
    vec = DictVectorizer()
    data = vec.fit_transform(features).toarray()

    samples_per_split = len(data)/10
    scores = np.zeros(shape=(repeats, 9, 3, 2))

    for i in xrange(repeats):
        scores[i] = get_data_points(data, labels)

    # now need to average it out somehow
    av_scores = scores.mean(axis=0)
    draw_plots(av_scores, samples_per_split)

    #pickle.dump(scores, open('scores.p', 'wb'))
    #pickle.dump(av_scores, open('av_scores.p', 'wb'))


def build_pipeline():
    """
    Set up classfier here to avoid repetition
    """
    # TODO what type of kernel to use?
    clf = Pipeline([('normaliser', preprocessing.Normalizer()),
                    #('svm', SVC(kernel='rbf', gamma=10))])
                    #('svm', SVC(kernel='sigmoid'))])
                    #('svm', SVC(kernel='poly', coef0=1, degree=2, gamma=2, cache_size=1000, class_weight='auto'))])
                    ('svm', SVC(kernel='rbf', gamma=1, cache_size=1000, class_weight='auto'))])
                    #('svm', SVC(kernel='linear'))])
                    #('random_forest', RandomForestClassifier(n_estimators=10, max_features='sqrt', bootstrap=False,
                                                             #n_jobs=-1))])
    return clf


def get_data_points(data, labels):
    """
    Get set of data points for one curve
    Want to add to the training data incrementally to mirror real life situtation
    """
    # set up array to hold scores
    scores = np.zeros(shape=(9, 3, 2))

    # first split at 10%
    train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(data, labels, train_size=0.1)
    no_samples = len(train_data)
    scores[0] = get_scores(train_data, train_labels, test_data, test_labels)

    # now loop to create remaining training sets
    for i in xrange(1, 9):
        more_data, test_data, more_labels, test_labels = cross_validation.train_test_split(test_data, test_labels,
                                                                                           train_size=no_samples)
        # add the new training data to existing
        train_data = np.append(train_data, more_data, axis=0)
        train_labels = np.append(train_labels, more_labels)
        scores[i] = get_scores(train_data, train_labels, test_data, test_labels)

    return scores


def get_scores(train_data, train_labels, test_data, test_labels):
    """
    Return array of scores
    """
    # set up classifier and train
    clf = build_pipeline()
    clf.fit(train_data, train_labels)
    # classify the test data
    predicted = clf.predict(test_data)
    # evaluate accuracy of output compared to correct classification
    scores = precision_recall_fscore_support(test_labels, predicted)

    # return precision, recall and f1
    return np.array([scores[0], scores[1], scores[2]])


def draw_plots(scores, samples_per_split):
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
    ticks = np.linspace(samples_per_split, 9*samples_per_split, 9)

    plot(ticks, true_p, false_p, 'Precision', 'plots/balanced_precision.tif')
    plot(ticks, true_r, false_r, 'Recall', 'plots/balanced_recall.tif')
    plot(ticks, true_f, false_f, 'F-score', 'plots/balanced_fscore.tif')


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

    # plot raw data points
    plt.plot(ticks, true, label='True relations')
    plt.plot(ticks, false, label='False relations')

    # now fit polynomial (straight line) to the points and extend plot out
    true_coefs = np.polyfit(ticks, true, deg=1)
    false_coefs = np.polyfit(ticks, false, deg=1)
    x_new = np.linspace(ticks.min(), 2*ticks.max())
    true_fitted = np.polyval(true_coefs, x_new)
    false_fitted = np.polyval(false_coefs, x_new)
    plt.plot(x_new, true_fitted, label='True relations best fit')
    plt.plot(x_new, false_fitted, label='False relations best fit')

    plt.legend(loc='best')

    plt.savefig(filepath, format='tif')
    plt.clf()

if __name__ == '__main__':
    learning_curves(repeats=20)
