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
from sklearn.ensemble import RandomForestClassifier

from app.feature_extractor import FeatureExtractor
from app.utility import time_stamped


def create_results():
    """
    Get cross validated scores for classifiers built with various parameters
    Write results to csv file for easy human analysis
    """
    # test a variety of features and algorithms
    clf = build_pipeline()

    # set up output file
    with open('results/new/feature_selection_all_words.csv', 'wb') as f_out:
        csv_writer = csv.writer(f_out, delimiter=',')
        csv_writer.writerow(['features', 'accuracy', 'auroc', 'true_P', 'true_R', 'true_F',
                             'false_P', 'false_R', 'false_F', 'average_P', 'average_R', 'average_F'])

        # first using all words and no other features
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=False, word_features=True,
                                     combo=False, pos=False, entity_type=False)
        write_scores(csv_writer, clf, extractor, -1, 'words only')
        print 'done'

        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=False, word_features=True,
                                     combo=False, pos=False, entity_type=True)
        write_scores(csv_writer, clf, extractor, -1, 'all words and type')
        print 'done'

        extractor = FeatureExtractor(word_gap=True, count_dict=False, phrase_count=False, word_features=True,
                                     combo=False, pos=False, entity_type=True)
        write_scores(csv_writer, clf, extractor, -1, 'all words, gap, type')
        print 'done'

        extractor = FeatureExtractor(word_gap=True, count_dict=False, phrase_count=False, word_features=True,
                                     combo=False, pos=True)
        write_scores(csv_writer, clf, extractor, -1, 'all words, gap, type, non-count pos')
        print 'done'

        extractor = FeatureExtractor(word_gap=True, count_dict=False, phrase_count=False, word_features=True,
                                     combo=True, pos=False)
        write_scores(csv_writer, clf, extractor, -1, 'all words, gap, type, non-count combo')
        print 'done'

        extractor = FeatureExtractor(word_gap=True, count_dict=False, phrase_count=False, word_features=True,
                                     combo=True, pos=True)
        write_scores(csv_writer, clf, extractor, -1, 'all words, gap, type, non-count pos combo')
        print 'done'

        extractor = FeatureExtractor(word_gap=True, count_dict=True, phrase_count=True, word_features=True,
                                     combo=True, pos=True)
        write_scores(csv_writer, clf, extractor, -1, 'all words, gap, phrase count, count pos combo')
        print 'done'


        '''
        ################################################################

        # first using all words and no other features
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=False, word_features=True,
                                     combo=False, pos=False)
        write_scores(csv_writer, clf, extractor, -1, 'words only')

        extractor = FeatureExtractor(word_gap=True, count_dict=False, phrase_count=False, word_features=True,
                                     combo=False, pos=False)
        write_scores(csv_writer, clf, extractor, -1, 'all words and gap')

        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, word_features=True,
                                     combo=False, pos=False)
        write_scores(csv_writer, clf, extractor, -1, 'all words and phrase count')

        extractor = FeatureExtractor(word_gap=True, count_dict=True, phrase_count=True, word_features=True,
                                     combo=False, pos=True)
        write_scores(csv_writer, clf, extractor, -1, 'all words, gap, phrase count, pos')

        extractor = FeatureExtractor(word_gap=True, count_dict=True, phrase_count=True, word_features=True,
                                     combo=True, pos=True)
        write_scores(csv_writer, clf, extractor, -1, 'all words, gap, phrase count, pos, combo')
        '''


def build_pipeline():
    """
    Set up classfier here to avoid repetition
    """
    # TODO what type of kernel to use? linear for original feature experiments?
    clf = Pipeline([('normaliser', preprocessing.Normalizer()),
                    #('svm', SVC(kernel='poly', coef0=1, degree=2, gamma=1, cache_size=1000))])
                    #('svm', SVC(kernel='poly', coef0=1, degree=3, gamma=2, cache_size=1000, C=1000))])
                    #('svm', SVC(kernel='rbf', gamma=1, cache_size=1000))])
                    ('svm', SVC(kernel='rbf', gamma=1, cache_size=2000, C=10))])
                    #('svm', SVC(kernel='linear', cache_size=2000))])
                    #('random_forest', RandomForestClassifier(n_estimators=10, max_features='sqrt', bootstrap=False,
                    # n_jobs=-1))])
    return clf


def write_scores(csv_writer, clf, extractor, how_many, features):
    """
    Write one set of scores to csv
    """
    scores, accuracy, auroc = cross_validated_scores(clf, extractor, how_many)

    for i in xrange(10):
        row = [features, accuracy[i], auroc[i],
               scores[i, 0, 1], scores[i, 1, 1], scores[i, 2, 1],  # true relations
               scores[i, 0, 0], scores[i, 1, 0], scores[i, 2, 0],  # false relations
               (scores[i, 0, 1] + scores[i, 0, 0])/2, (scores[i, 1, 1] + scores[i, 1, 0])/2,  # averages
               (scores[i, 2, 1] + scores[i, 2, 0])/2]

        csv_writer.writerow(row)


def cross_validated_scores(clf, extractor, how_many):
    """
    Calculate scores using 10 fold cross validation
    """
    # set up array to hold scores
    scores = np.zeros(shape=(10, 3, 2))
    accuracy = np.zeros(shape=10)
    auroc = np.zeros(shape=10)

    records = load_records()
    #bullshit = np.zeros(len(records))

    # TODO ask about stratified or not stratified
    # set up stratified 10 fold cross validator, use specific random state for proper comparison
    cv = cross_validation.KFold(len(records), shuffle=True, n_folds=10, random_state=1)

    # iterating through the cv gives lists of indices for each fold
    for i, (train, test) in enumerate(cv):
        '''
        # set up word features based on training set only
        train_records = [records[j] for j in train]
        print len(train_records)
        extractor.create_dictionaries(train_records, how_many)
        #print extractor.bet_verb_dict
        '''

        # now generate features
        data, labels = extractor.generate_features(records)
        vec = DictVectorizer()
        data = vec.fit_transform(data).toarray()

        train_data, test_data = data[train], data[test]
        train_labels, test_labels = labels[train], labels[test]

        scores[i], accuracy[i], auroc[i] = get_scores(clf, train_data, train_labels, test_data, test_labels)

    #print 'precision, recall, f1\n', scores
    #print 'accuracy\n', accuracy
    return scores, accuracy, auroc


def load_features_data(extractor, how_many):
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

    extractor.create_dictionaries(records, how_many)

    return extractor.generate_features(records, balance_classes=False)


def load_records():
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
    return records


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
    # evaluate auroc and R, P, F scores
    auroc = metrics.roc_auc_score(test_labels, predicted)
    scores = precision_recall_fscore_support(test_labels, predicted)
    #print metrics.classification_report(test_labels, predicted)

    # ROC STUFF
    #confidence = clf.decision_function(test_data)
    #fpr, tpr, thresholds = metrics.roc_curve(test_labels, confidence)
    #print fpr
    #print tpr
    #print thresholds

    return np.array([scores[0], scores[1], scores[2]]), accuracy, auroc


def plot_roc_curve():
    """
    Plot roc curve, not cross validated for now
    """
    clf = build_pipeline()
    extractor = FeatureExtractor(word_gap=True, word_features=True, count_dict=True, phrase_count=True)

    features, labels = load_features_data(extractor)
    # transform from dict into array for training
    vec = DictVectorizer()
    data = vec.fit_transform(features).toarray()

    # split data into train and test, may want to use cross validation later
    train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(data, labels, train_size=0.9,
                                                                                         random_state=1)
    clf.fit(train_data, train_labels)

    confidence = clf.decision_function(test_data)
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, confidence)
    auroc = metrics.auc(fpr, tpr)

    print len(fpr), len(tpr)
    # set up the figure
    plt.figure()
    #plt.grid()
    plt.xlabel('FP rate')
    plt.ylabel('TP rate')
    plt.title('Receiver operating characteristic')
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auroc)
    plt.plot([0, 1], [0, 1], 'k--')

    plt.legend(loc='best')
    filepath = 'results/' + time_stamped('roc.png')
    plt.savefig(filepath, format='png')

if __name__ == '__main__':
    create_results()
    #plot_roc_curve()
