from flask import Flask
from flask import render_template
from flask import request
from flask import session
from flask import redirect

import sqlite3

app = Flask(__name__)
app.secret_key = 'blahblahblah'
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
        # TODO later need to add WHERE relations.true_rel IS NULL so we don't reclassify things
        cursor.execute('''SELECT min(rel_id) as next
                          FROM relations
                          WHERE relations.rel_id NOT IN (SELECT rel_id
                                                         FROM decisions
                                                         WHERE decisions.user_id = ?);''', [user_id])
        session['next_relation'] = cursor.fetchone()['next']

    return redirect('/classify')


@app.route('/classify')
def classify():
    """
    Display next relation to be classified
    """
    next_rel = session['next_relation']
    before, between, after, e1, e2, true_rel = return_relation(next_rel)

    # set radio button to pre check the classifiers prediction
    # TODO why does this switch to false every time???
    if true_rel:
        true_check = 'true'
        false_check = 'false'
    else:
        true_check = 'false'
        false_check = 'true'

    return render_template('index.html', before=before, between=between, after=after, classification=true_rel,
                           e1=e1, e2=e2, true_check=true_check, false_check=false_check)


@app.route('/save', methods=['POST'])
def record_decision():
    """
    Save annotators decision and redirect to next relation to be classified
    """
    # write decision to the database
    store_decision(request.form['class'])
    # TODO not sure if this will work properly, relations may not be ordered one by one?
    # may have to do a select min where rel_id not in decisions for this use
    # increment relation id
    session['next_relation'] = int(session['next_relation']) + 1

    return redirect('/classify')


def store_decision(classification):
    """
    Record the annotators classification
    """
    rel_id = session['next_relation']
    user_id = session['user_id']

    with sqlite3.connect(db_path) as db:
        cursor = db.cursor()
        cursor.execute('''INSERT into decisions
                          VALUES (NULL, ?, ?, ?)''',
                       (rel_id, user_id, classification))
        db.commit()


def split_sentences(sent, start1, end1, start2, end2):
    """
    Put divs around the entities so they will be highlighted on page
    """
    return sent[:start1], sent[end1:start2], sent[end2:]


def return_relation(rel_id):
    """
    Get relation from database - very basic for now, just select next one in list
    """
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        cursor = db.cursor()
        # TODO only alter WHERE statement instead of repeating entire query
        # no such thing as a stored procedure in sqlite... need to do everything here
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
                                 relations.true_rel
                          FROM sentences NATURAL JOIN relations
                          WHERE relations.rel_id = ?;''', [rel_id])

        row = cursor.fetchone()
        before, between, after = split_sentences(row['sentence'], row['start1'], row['end1'], row['start2'],
                                                 row['end2'])

        return before, between, after, row['entity1'], row['entity2'], row['true_rel']


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
    app.run(debug=True)
