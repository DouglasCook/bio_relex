import random
import sys

from flask import Flask
from flask import render_template
from flask import request
from flask import session
from flask import redirect

import sqlite3

# TODO exception catching / error redirects?
app = Flask(__name__)
app.secret_key = 'blahblahblah'

# TODO fix package importing
# for some reason this doesn't work... fuck knows why, it is probably not worth the bother
#from .. import utility
#db_path = utility.build_filepath(__file__, '../database/test.db')
db_path = '../database/test.db'


@app.route('/')
@app.route('/index')
def user_selection():
    user_list = get_user_list()
    return render_template('login.html', users=user_list)

@app.route('/login', methods=['POST'])
def login():
    """
    Store user in session and find next relation for them to annotate
    """
    # store user in session
    user_id = request.form['user']
    session['user_id'] = user_id

    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        # TODO change query dependent on active learning method eg decision function
        # THIS ORDERS RELATIONS BY THEIR DISTANCE FROM THE SEPARATING HYPERPLANE
        cursor.execute('''SELECT rel_id
                          FROM relations NATURAL JOIN predictions
                          WHERE relations.true_rel IS NULL AND
                                relations.bad_ner = 0 AND
                                predictions.user_id = (SELECT max(user_id)
                                                       FROM users
                                                       WHERE type = 'classifier') AND
                                relations.rel_id NOT IN (SELECT rel_id
                                                         FROM decisions
                                                         WHERE decisions.user_id = ?)
                           ORDER BY predictions.confidence_value;''', [user_id])

        """
        # THIS JUST TAKES EVERYTHING THAT HAS NOT BEEN HUMAN ANNOTATED
        cursor.execute('''SELECT rel_id
                          FROM relations
                          WHERE relations.true_rel IS NULL AND
                                relations.bad_ner = 0 AND
                                relations.rel_id NOT IN (SELECT rel_id
                                                         FROM decisions
                                                         WHERE decisions.user_id = ?);''', [user_id])
        """

        """
        #USE THIS TO SEE WHAT HAS BEEN CLASSIFIED AS TRUE
        cursor.execute('''SELECT rel_id
                          FROM relations NATURAL JOIN predictions
                          WHERE relations.true_rel IS NULL AND
                                relations.bad_ner = 0 AND
                                predictions.decision = 1 AND
                                relations.rel_id NOT IN (SELECT rel_id
                                                         FROM decisions
                                                         WHERE decisions.user_id = ?);''', [user_id])
        """

        # create list of relations to classify to iterate through
        rels = [c[0] for c in cursor]
        # TODO shuffling is one possible strategy, in order is another, distance from support vectors is another
        #random.shuffle(rels)
        session['rels_to_classify'] = rels
        session['number_rels'] = len(rels)
        session['next_index'] = 0

    return redirect('/classify')


@app.route('/classify')
def classify():
    """
    Display next relation to be classified
    """
    next_rel = session['rels_to_classify'][int(session['next_index'])]
    before, between, after, e1, e2, prediction, type1 = return_relation(next_rel)

    # set radio button to pre check the classifiers prediction
    if prediction:
        pred = 'True'
        true_check = True
    else:
        pred = 'False'
        true_check = False

    # set correct colouring for drugs and disorders
    if type1 == 'Drug':
        drug_first = True
    else:
        drug_first = False

    return render_template('index.html', before=before, between=between, after=after, classification=pred,
                           e1=e1, e2=e2, true_check=true_check, drug_first=drug_first)


@app.route('/save', methods=['POST'])
def record_decision():
    """
    Save annotators decision and redirect to next relation to be classified
    """
    # write decision to the database
    store_decision(request.form['class'])

    # redirect to finished page if there are no remaining relations
    if session['next_index'] == session['number_rels'] - 1:
        return render_template('finished.html')

    # increment next relation index
    session['next_index'] = int(session['next_index']) + 1

    return redirect('/classify')


def store_decision(classification):
    """
    Record the annotators classification
    """
    rel_id = session['rels_to_classify'][int(session['next_index'])]
    user_id = session['user_id']

    with sqlite3.connect(db_path) as db:
        cursor = db.cursor()
        cursor.execute('''INSERT into decisions
                          VALUES (NULL, ?, ?, ?)''',
                       (rel_id, user_id, classification))

        # set bad NER to true if annotator believes it is
        # ideally this wouldn't happen here, it would need to be a majority decision so would be in retraining
        if classification == '2':
            cursor.execute('''UPDATE relations
                              SET bad_ner = 1
                              WHERE rel_id = ?''', [rel_id])


def split_sentence(sent, start1, end1, start2, end2):
    """
    Put divs around the entities so they will be highlighted on page
    """
    return sent[:start1], sent[end1 + 1:start2], sent[end2 + 1:]


def return_relation(rel_id):
    """
    Get potential relation from database
    """
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        cursor = db.cursor()
        cursor.execute('''SELECT sentences.sentence,
                                 relations.entity1,
                                 relations.type1,
                                 relations.start1,
                                 relations.end1,
                                 relations.entity2,
                                 relations.type2,
                                 relations.start2,
                                 relations.end2,
                                 relations.rel_id,
                                 predictions.decision
                          FROM sentences NATURAL JOIN relations
                                         NATURAL JOIN predictions
                          WHERE relations.rel_id = ? AND
                                predictions.user_id = (SELECT max(user_id)
                                                       FROM users
                                                       WHERE type = 'classifier');''', [rel_id])

        row = cursor.fetchone()
        before, between, after = split_sentence(row['sentence'], row['start1'], row['end1'], row['start2'],
                                                row['end2'])

        return before, between, after, row['entity1'], row['entity2'], row['decision'], row['type1']


def get_user_list():
    """
    Return list of existing users for login page
    """
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE type = 'user'")
    return [u for u in cursor]


if __name__ == '__main__':
    # set command line arg to 1 to go live
    if int(sys.argv[1]) == 1:
        app.run(host='0.0.0.0', port=55100)
    else:
        app.run(debug=True)
