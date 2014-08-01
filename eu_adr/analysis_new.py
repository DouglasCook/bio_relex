import sqlite3
import random
import datetime
import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from eu_adr.app.feature_extractor import FeatureExtractor


def create_results():
    """
    Get cross validated scores for classifiers built with various parameters
    Write results to csv file for easy human analysis
    """
    # test a variety of features and algorithms
    clf = build_pipeline()

    # set up output file
    with open('results/feature_selection.csv', 'wb') as f_out:
        csv_writer = csv.writer(f_out, delimiter=',')
        csv_writer.writerow(['features', 'accuracy', 'true_P', 'true_R', 'true_F',
                             'false_P', 'false_R', 'false_F', 'average_P', 'average_R', 'average_F'])

        # first with all features
        extractor = FeatureExtractor(word_gap=True, count_dict=True, phrase_count=True)
        write_scores(csv_writer, clf, extractor, 'all')

        # no word counting
        extractor = FeatureExtractor(word_gap=False, count_dict=True, phrase_count=True)
        write_scores(csv_writer, clf, extractor, 'no word count')

        # no phrase counting
        extractor = FeatureExtractor(word_gap=True, count_dict=True, phrase_count=False)
        write_scores(csv_writer, clf, extractor, 'no phrase count')

        # non counting dictionaries
        extractor = FeatureExtractor(word_gap=True, count_dict=False, phrase_count=True)
        write_scores(csv_writer, clf, extractor, 'non-counting dict')


def write_scores(csv_writer, clf, extractor, features):
    """
    Write one set of scores to csv
    """
    scores, accuracy = cross_validated_scores(clf, extractor)

    for i in xrange(10):
        row = [features, accuracy[i],
               scores[i, 0, 1], scores[i, 1, 1], scores[i, 2, 1],  # true relations
               scores[i, 0, 0], scores[i, 1, 0], scores[i, 2, 0],  # false relations
               (scores[i, 0, 1] + scores[i, 0, 0])/2, (scores[i, 1, 1] + scores[i, 1, 0])/2,  # averages
               (scores[i, 2, 1] + scores[i, 2, 0])/2]

        csv_writer.writerow(row)


def cross_validated_scores(clf, extractor):
    """
    Calculate scores using 10 fold cross validation
    """
    # set up array to hold scores
    scores = np.zeros(shape=(10, 3, 2))
    accuracy = np.zeros(shape=10)

    features, labels = load_features_data(extractor)
    # transform from dict into array for training
    vec = DictVectorizer()
    data = vec.fit_transform(features).toarray()

    # set up stratified 10 fold cross validator, use specific random state for proper comparison
    cv = cross_validation.StratifiedKFold(labels, shuffle=True, n_folds=10, random_state=1)

    # iterating through the cv gives lists of indices for each fold
    for i, (train, test) in enumerate(cv):
        train_data, test_data = data[train], data[test]
        train_labels, test_labels = labels[train], labels[test]

        scores[i], accuracy[i] = get_scores(clf, train_data, train_labels, test_data, test_labels)

    #print 'precision, recall, f1\n', scores
    #print 'accuracy\n', accuracy
    return scores, accuracy


def load_features_data(extractor):
    """
    Load some part of data
    """

    with sqlite3.connect('database/euadr_biotext.db') as db:
        # using Row as row factory means can reference fields by name instead of index
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        cursor.execute('''SELECT relations.*
                          FROM relations NATURAL JOIN sentences
                          WHERE sentences.source != 'pubmed';''')

    records = cursor.fetchall()

    return extractor.generate_features(records, balance_classes=False)


def build_pipeline():
    """
    Set up classfier here to avoid repetition
    """
    # TODO what type of kernel to use?
    clf = Pipeline([('normaliser', preprocessing.Normalizer()),
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


def get_scores(clf, train_data, train_labels, test_data, test_labels):
    """
    Return array of scores
    """
    # set up classifier and train
    clf.fit(train_data, train_labels)
    # calculate mean accuracy since not included in other set of scores
    accuracy = clf.score(test_data, test_labels)
    # classify the test data
    predicted = clf.predict(test_data)
    # evaluate scores
    scores = precision_recall_fscore_support(test_labels, predicted)
    #print metrics.classification_report(test_labels, predicted)

    # return precision, recall and f1
    return np.array([scores[0], scores[1], scores[2]]), accuracy


if __name__ == '__main__':
    create_results()
