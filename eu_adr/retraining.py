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


def classify_remaining(optimise_params=False, no_biotext=False):
    """
    Call classifier to predict values of remaining unclassified instances
    """
    clf = Classifier(optimise_params, no_biotext)

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


def count_true_false_predicions():
    """
    See how many relations predicted as true/false by latest run
    """
    with sqlite3.connect(db_path) as db:
        # need to return dictionary so it matches csv stuff
        db.row_factory = sqlite3.row
        cursor = db.cursor()

        # get latest classifier
        cursor.execute('''select max(user_id)
                          from users
                          where type = 'classifier';''')
        clsf_id = cursor.fetchone()[0]

        # count the relations
        cursor.execute('''SELECT decision, count(rel_id)
                          FROM decisions
                          WHERE user_id = ?
                          GROUP BY decision;''', [clsf_id])

        for row in cursor:
            print row[0], row[1]


def delete_decisions():
    """
    Delete all decisions and related records
    """
    with sqlite3.connect(db_path) as db:
        # need to return dictionary so it matches csv stuff
        cursor = db.cursor()
        cursor.execute('DELETE from decisions;')
        cursor.execute('DELETE from classifier_data;')
        cursor.execute('DELETE from users WHERE type = "classifier";')


if __name__ == '__main__':
    #update_correct_classifications()
    #classify_remaining(optimise_params=False, no_biotext=False)
    #count_true_false_predicions()
    delete_decisions()
