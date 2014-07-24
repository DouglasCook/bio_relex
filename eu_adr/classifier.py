import sqlite3

from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

import utility
from scikit_feature_extraction import generate_features


class Classifier():
    """
    Classifier class to make pubmed classification more straight forward
    """
    def __init__(self):
        # set up connection to database
        self.db_path = utility.build_filepath(__file__, 'database/test.db')

        # set up vectoriser for transforming data from dictionary to numpy array
        self.vec = DictVectorizer()

        # set up user record for the classifier
        self.user_id = self.create_user()

        # save records used for training for later experimentation with different classifiers
        self.create_training_set()

        # set up classifier pipeline
        self.clf = self.train()

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

    def create_training_set(self):
        """
        Write relations to use for training into classifier table
        """
        with sqlite3.connect(self.db_path) as db:
            cursor = db.cursor()
            # by default want to train on all examples with a 'correct' classification
            cursor.execute('''INSERT INTO classifier_data
                                     SELECT ? as clsf_id,
                                            rel_id
                                     FROM relations
                                     WHERE true_rel IS NOT NULL;''', [self.user_id])

    def get_training_data(self):
        """
        Return features and class labels for training set
        """
        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = db.cursor()
            # training data query
            cursor.execute('''SELECT *
                              FROM relations
                              WHERE rel_id IN (SELECT rel_id
                                               FROM classifier_data
                                               WHERE clsf_id = ?);''', [self.user_id])
            records = cursor.fetchall()

        # extract the feature vectors and class labels for training set
        return generate_features(records)

    def train(self):
        """
        Train the model on selected training set
        """
        # set up pipeline to normalise the data then build the model
        clf = Pipeline([('normaliser', preprocessing.Normalizer()),
                        ('svm', SVC(kernel='poly', coef0=3, degree=2, gamma=1, cache_size=1000, class_weight='auto'))])

        data, labels = self.get_training_data()

        # convert from dict into np array
        data = self.vec.fit_transform(data).toarray()

        # TODO may want to add some sort of parameter optimisation here using scikit?
        # train the model
        clf.fit(data, labels)

        return clf

    def classify(self, record):
        """
        Classify given record and write prediction to table
        """
        # list is expected when generating features so put the record in a list
        data = generate_features([record], no_class=True)
        data = self.vec.transform(data).toarray()
        # TODO speed up the classification by classifying everything in one go?
        # predict returns an array so need to remove element
        prediction = self.clf.predict(data)[0]

        with sqlite3.connect(self.db_path) as db:
            cursor = db.cursor()
            # by default want to train on all examples with a 'correct' classification
            cursor.execute('''INSERT INTO decisions
                                     VALUES (NULL, ?, ?, ?);''',
                           (record['rel_id'], self.user_id, prediction))
