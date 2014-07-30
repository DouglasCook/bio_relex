import sqlite3
import random

from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import utility


class Classifier():
    """
    Classifier class to make pubmed classification more straight forward
    """
    def __init__(self, f_extractor, optimise_params=False, no_biotext=False):
        # set up connection to database
        self.db_path = utility.build_filepath(__file__, '../database/relex.db')

        # set up the feature extractor with desired features
        self.extractor = f_extractor

        # set up vectoriser for transforming data from dictionary to numpy array
        self.vec = DictVectorizer()

        # set up user record for the classifier
        self.user_id = self.create_user()

        # save records used for training for later experimentation with different classifiers
        self.create_training_set(no_biotext)

        # set up classifier pipeline
        self.clf = self.train(optimise_params)

    def create_user(self):
        """
        Create new user for this classifier and return user id
        """
        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = db.cursor()
            cursor.execute('''INSERT INTO users
                                     VALUES (NULL, 'classifier', 'classifier');''')

            cursor.execute('SELECT MAX(user_id) as max FROM users;')

            return cursor.fetchone()['max']

    def create_training_set(self, no_biotext):
        """
        Write relations to use for training into classifier table
        """
        with sqlite3.connect(self.db_path) as db:
            cursor = db.cursor()
            # by default want to train on all examples with a 'correct' classification
            if no_biotext:
                print 'NO BIOTEXT!'
                cursor.execute('''INSERT INTO classifier_data
                                         SELECT ? as clsf_id,
                                                rel_id
                                         FROM relations NATURAL JOIN sentences
                                         WHERE true_rel IS NOT NULL AND
                                               sentences.source != 'biotext';''', [self.user_id])
            else:
                cursor.execute('''INSERT INTO classifier_data
                                         SELECT ? as clsf_id,
                                                rel_id
                                         FROM relations
                                         WHERE true_rel IS NOT NULL;''', [self.user_id])

    def get_training_data(self):
        """
        Return features and class labels for training set
        """
        # first balance the classes - NOT DOING THIS
        #self.balance_classes()
        # if want to balance need to change query below to use FROM classifier_data_balanced

        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = db.cursor()
            # write balanced sets of training rels to table
            cursor.execute('''SELECT *
                              FROM relations
                              WHERE rel_id IN (SELECT training_rel
                                               FROM classifier_data
                                               WHERE clsf_id = ?);''', [self.user_id])
            records = cursor.fetchall()

        # extract the feature vectors and class labels for training set
        return self.extractor.generate_features(records)

    def balance_classes(self):
        """
        Undersample the over-represented class so it contains same number of samples
        """
        with sqlite3.connect(self.db_path) as db:
            cursor = db.cursor()
            # first get everything that has been classified so far
            cursor.execute('''SELECT true_rel, rel_id
                              FROM relations
                              WHERE rel_id IN (SELECT training_rel
                                               FROM classifier_data
                                               WHERE clsf_id = ?);''', [self.user_id])
            records = cursor.fetchall()

        # count occurrence of each class
        true = [x for x in records if x[0] == 1]
        false = [x for x in records if x[0] == 0]
        true_count, false_count = len(true), len(false)

        # undersample the over represented class
        if true_count < false_count:
            false = random.sample(false, true_count)
        elif false_count < true_count:
            true = random.sample(true, false_count)

        # put back together again
        print len(true), len(false)
        together = [f[1] for f in false] + [t[1] for t in true]

        with sqlite3.connect(self.db_path) as db:
            cursor = db.cursor()
            # loop through all records and add to proper training table
            for t in together:
                cursor.execute('''INSERT INTO classifier_data_balanced
                                  VALUES (?, ?);''', (self.user_id, t))

    @staticmethod
    def tune_parameters(data, labels):
        """
        Tune the parameters using exhaustive grid search
        """
        # set cv here, why not
        cv = cross_validation.StratifiedKFold(labels, n_folds=5, shuffle=True)

        pipeline = Pipeline([('normaliser', preprocessing.Normalizer()),
                             ('svm', SVC(kernel='poly', cache_size=1000))])

        # can test multiple kernels as well if desired
        #param_grid = [{'kernel': 'poly', 'coef0': [1, 5, 10, 20], 'degree': [2, 3, 4, 5, 10]}]
        param_grid = [{'svm__coef0': [1, 2, 3, 4, 5], 'svm__degree': [2, 3, 4, 5], 'svm__C': [1, 2], 'svm__gamma': [0, 1]}]
        print 'tuning params'
        clf = GridSearchCV(pipeline, param_grid, n_jobs=-1, cv=cv)
        clf.fit(data, labels)

        print 'best parameters found:'
        print clf.best_estimator_
        return clf.best_estimator_

    def train(self, optimise_params):
        """
        Train the model on selected training set
        """
        data, labels = self.get_training_data()

        # convert from dict into np array
        data = self.vec.fit_transform(data).toarray()

        # TODO is optimising the parameters worth it or not useful for results?
        if optimise_params:
            optimal = self.tune_parameters(data, labels)
            best_coef = optimal.named_steps['svm'].coef0
            best_degree = optimal.named_steps['svm'].degree
            best_c = optimal.named_steps['svm'].C
            best_gamma = optimal.named_steps['svm'].gamma

            # set up pipeline to normalise the data then build the model
            # TODO check what the deal is with auto weighting the classes...
            clf = Pipeline([('normaliser', preprocessing.Normalizer()),
                            ('svm', SVC(kernel='poly', coef0=best_coef, degree=best_degree, C=best_c, gamma=best_gamma,
                                        cache_size=1000))])
                            #('svm', SVC(kernel='poly', coef0=best_coef, degree=best_degree, gamma=1, cache_size=1000,
                                        #class_weight='auto'))])
        else:
            # set up pipeline to normalise the data then build the model
            clf = Pipeline([('normaliser', preprocessing.Normalizer()),
                            ('svm', SVC(kernel='poly', coef0=1, degree=2, gamma=1, cache_size=1000))])
                            #('svm', SVC(kernel='poly', coef0=1, degree=2, gamma=1, cache_size=1000,
                                        #class_weight='auto'))])
                            #('svm', SVC(kernel='rbf', gamma=1, cache_size=1000))])
                            #('svm', SVC(kernel='sigmoid', cache_size=1000))])
                            #('svm', SVC(kernel='linear', cache_size=1000))])
                            #('random_forest', RandomForestClassifier(n_estimators=10, max_features='sqrt', bootstrap=False, n_jobs=-1))])

        # train the model
        clf.fit(data, labels)

        return clf

    def classify(self, record):
        """
        Classify given record and write prediction to table
        """
        # list is expected when generating features so put the record in a list
        data = self.extractor.generate_features([record], no_class=True)
        #print data
        data = self.vec.transform(data).toarray()
        # TODO speed up the classification by classifying everything in one go?
        # predict returns an array so need to remove element
        prediction = self.clf.predict(data)[0]
        # calculate distance from separating hyperplane as measure of confidence
        confidence = abs(self.clf.decision_function(data)[0][0])

        with sqlite3.connect(self.db_path) as db:
            cursor = db.cursor()
            # by default want to train on all examples with a 'correct' classification
            cursor.execute('''INSERT INTO predictions
                                     VALUES (NULL, ?, ?, ?, ?);''',
                           (record['rel_id'], self.user_id, prediction, confidence))
                           #(record['rel_id'], self.user_id, prediction))
