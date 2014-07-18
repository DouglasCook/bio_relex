import pickle
import datetime

import numpy as np

from scipy.interpolate import spline

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.learning_curve import learning_curve


def load_data(total_instances=0):
    # use all instances if zero is passed in
    if total_instances == 0:
        features = pickle.load(open('pickles/scikit_data.p', 'rb'))
        labels = np.array(pickle.load(open('pickles/scikit_target.p', 'rb')))
    # otherwise slice number of instances requested, biotext ones are at end so will be cut off
    else:
        features = pickle.load(open('pickles/scikit_data.p', 'rb'))
        features = features[:total_instances]
        labels = pickle.load(open('pickles/scikit_target.p', 'rb'))
        labels = np.array(labels[:total_instances])

    return features, labels


def check_data_range(data):
    """
    Get some idea of how the data is distributed
    """
    print np.max(data)
    print np.min(data)
    print np.mean(data)


def cross_validated(total_instances=0):
    """
    Calculate stats using cross validations
    """
    features, labels = load_data(total_instances)
    # convert from dict into np array
    vec = DictVectorizer()
    data = vec.fit_transform(features).toarray()

    # set up pipeline to normalise the data then build the model
    # TODO do I want normalise all of the features?
    clf = Pipeline([('normaliser', preprocessing.Normalizer()),
                    # don't think scaling is required if I normalise the data?
                    #('scaler', preprocessing.MinMaxScaler()),
                    #('scaler', preprocessing.StandardScaler()),
                    #('svm', SVC(kernel='linear', C=2.5))])
                    #('svm', SVC(kernel='rbf', gamma=1))])
                    #('svm', SVC(kernel='sigmoid', gamma=10, coef0=10))])
                    ('svm', SVC(kernel='poly', coef0=4, gamma=0.5, degree=3))])

    # TODO what is the first parameter here?
    # when using cross validation there is no need to manually train the model
    cv = cross_validation.StratifiedKFold(labels, n_folds=10, shuffle=True, random_state=0)

    with open('results/results.txt', 'a') as log:
        # TODO must be a better way to calculate the scores, ie not one at a time and separated by class???
        log.write('started: ' + str(datetime.datetime.now()) + '\n')

        # n_jobs parameter is number of cores to use, -1 for all cores
        log.write('accuracy = %f\n' %
                  np.mean(cross_validation.cross_val_score(clf, data, labels, cv=cv, scoring='accuracy', n_jobs=-1)))
        log.write('precision = %f\n' %
                  np.mean(cross_validation.cross_val_score(clf, data, labels, cv=cv, scoring='precision', n_jobs=-1)))
        log.write('recall = %f\n' %
                  np.mean(cross_validation.cross_val_score(clf, data, labels, cv=cv, scoring='recall', n_jobs=-1)))
        log.write('F-score = %f\n' %
                  np.mean(cross_validation.cross_val_score(clf, data, labels, cv=cv, scoring='f1', n_jobs=-1)))

        log.write('finished: ' + str(datetime.datetime.now()) + '\n\n')


def no_cross_validation(total_instances=0):
    features, labels = load_data(total_instances)

    # convert from dict into np array
    vec = DictVectorizer()
    data = vec.fit_transform(features).toarray()

    # set up pipeline to normalise the data then build the model
    clf = Pipeline([('scaler', preprocessing.Normalizer()),
                    #('scaler', preprocessing.StandardScaler()),
                    #('svm', SVC(kernel='linear'))])
                    #('svm', SVC(kernel='rbf', gamma=1))])
                    #('svm', SVC(kernel='sigmoid', gamma=10, coef0=10))])
                    ('svm', SVC(kernel='poly', coef0=4, gamma=0.5, degree=2))])

    # split data into training and test sets
    train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(data, labels, test_size=0.2)
    clf.fit(train_data, train_labels)

    # classify the test data
    predicted = clf.predict(test_data)
    # evaluate accuracy of output compared to correct classification
    print precision_recall_fscore_support(test_labels, predicted)
    print metrics.classification_report(test_labels, predicted, target_names=['True', 'False'])
    print metrics.confusion_matrix(test_labels, predicted)


def learning_curves(total_instances=0):
    """
    Plot learning curves of f-score for training and test data
    """
    features, labels = load_data(total_instances)

    # convert from dict into np array
    vec = DictVectorizer()
    data = vec.fit_transform(features).toarray()

    # set up pipeline to normalise the data then build the model
    # TODO do I want normalise all of the features?
    clf = Pipeline([('normaliser', preprocessing.Normalizer()),
                    #('svm', SVC(kernel='poly', coef0=4, gamma=0.5, degree=3))])
                    ('svm', SVC(kernel='linear'))])

    cv = cross_validation.StratifiedKFold(labels, n_folds=10, shuffle=True)

    # why does this always return results in the same pattern??? something fishy is going on
    sizes, t_scores, v_scores = learning_curve(clf, data, labels,
                                               train_sizes=np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                                               cv=cv, scoring='f1', n_jobs=-1)

    # define new set of points to be used to smooth the plots
    x_new = np.linspace(sizes.min(), sizes.max())
    training_results = np.array([np.mean(t_scores[i]) for i in range(len(t_scores))])
    training_smooth = spline(sizes, training_results, x_new)
    validation_results = np.array([np.mean(v_scores[i]) for i in range(len(v_scores))])
    validation_smooth = spline(sizes, validation_results, x_new)

    #plt.plot(sizes, validation_results)
    plt.plot(x_new, validation_smooth)
    #plt.plot(sizes, training_results)
    plt.plot(x_new, training_smooth)
    plt.show()

if __name__ == '__main__':
    #no_cross_validation(1150)
    #cross_validated(1150)
    learning_curves(1150)
