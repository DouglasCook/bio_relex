import sqlite3

from classifier import Classifier

import utility
db_path = utility.build_filepath(__file__, 'database/test.db')


def classify_remaining():
    """
    Call classifier to predict values of remaining unclassified instances
    """
    clf = Classifier()

    with sqlite3.connect(db_path) as db:
        # need to return dictionary so it matches csv stuff
        db.row_factory = sqlite3.Row
        cursor = db.cursor()
        # query for all unclassified instances
        cursor.execute('''SELECT *
                          FROM relations
                          WHERE true_rel IS NULL''')
        # need to fetch all since classifier will want to use db and cannot have it locked
        records = cursor.fetchall()

    for row in records:
        clf.classify(row)


if __name__ == '__main__':
    classify_remaining()