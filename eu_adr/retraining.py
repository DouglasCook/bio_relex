import sqlite3

from classifier import Classifier

import utility
db_path = utility.build_filepath(__file__, 'database/test.db')


def update_correct_classifications():
    """
    Apply the decisions of the annotator(s) to the relations table
    """
    # TODO if there will be multiple annotators this needs to be changed to implement majority decision
    with sqlite3.connect(db_path) as db:
        # need to return dictionary so it matches csv stuff
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        # updated all unclassified relations based on annotators decisions
        # this query is horrific but don't think sqlite offers a nicer way to do it

        cursor.execute(''' UPDATE relations
                           SET true_rel = (SELECT decisions.decision
                                           FROM decisions NATURAL JOIN users
                                           WHERE decision != 2 AND
                                           users.type != 'classifier' AND
                                           decisions.rel_id = relations.rel_id)
                           WHERE relations.rel_id IN(SELECT decisions.rel_id
                                                     FROM decisions NATURAL JOIN users
                                                     WHERE decisions.decision != 2 AND
                                                     users.type != 'classifier');''')


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
    update_correct_classifications()
    classify_remaining()