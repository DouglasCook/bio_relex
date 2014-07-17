import pickle

import numpy as np

from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation
from sklearn import metrics


def bam():
    features = pickle.load(open('pickles/scikit_data.p', 'rb'))
    labels = pickle.load(open('pickles/scikit_target.p', 'rb'))

    # load the dict vectoriser from scikit-learn
    vec = DictVectorizer()

    # split data into training and test sets
    data = vec.fit_transform(features).toarray()
    data_train, data_test, labels_train, label_test = cross_validation.train_test_split(data, labels, test_size=0.1)

    # load and train the classifier
    clf = SVC()
    clf.fit(data_train, labels_train)

    # classify the test data
    #predicted = clf.predict(data_test)
    # evaluate accuracy of output compared to correct classification
    #print metrics.classification_report(label_test, predicted, target_names=['True', 'False'])
    #print metrics.confusion_matrix(label_test, predicted)

    cv = cross_validation.StratifiedKFold(labels, n_folds=10, shuffle=True, random_state=1)
    print np.mean(cross_validation.cross_val_score(clf, data, labels, cv=cv, scoring='f1'))

if __name__ == '__main__':
    bam()
