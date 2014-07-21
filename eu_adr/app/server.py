from flask import Flask
from flask import render_template
from flask import request
from flask import session
from flask import redirect

import sqlite3

app = Flask(__name__)
app.secret_key = 'blahblahblah'

@app.route('/')
@app.route('/index')
def index():
    """
    Get the first relation to be classified and display
    """
    rel_id, sent, true_rel = next_relation()
    session['next_relation'] = rel_id
    return render_template('index.html', sentence=sent, classification=true_rel)


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


@app.route('/classify')
def classify():
    """
    Display next relation to be classified
    """
    next_rel = session['next_relation']
    sent, true_rel = next_relation(next_rel)
    return render_template('index.html', sentence=sent, classification=true_rel)


def store_decision(classification, user_id=1):
    """
    Record the annotators classification
    """
    rel_id = session['next_relation']

    with sqlite3.connect('../database/test.db') as db:
        cursor = db.cursor()
        cursor.execute('''INSERT into decisions
                          VALUES (?, ?, ?)''',
                       (rel_id, user_id, classification))
        db.commit()


def next_relation(rel_id=0):
    """
    Get relation from database - very basic for now, just select next one in list
    """
    with sqlite3.connect('../database/test.db') as db:
        db.row_factory = sqlite3.Row
        cursor = db.cursor()
        # no such thing as a stored procedure in sqlite... need to do everything here
        # relation id may be specified in which case just use it
        if rel_id:
            cursor.execute('''SELECT sentences.sentence,
                                     relations.true_rel,
                                     relations.rel_id
                              FROM sentences NATURAL JOIN relations
                              WHERE relations.rel_id = ?;''', [rel_id])
        # otherwise calculate next one to send over
        else:
            cursor.execute('''SELECT sentences.sentence,
                                     relations.true_rel,
                                     relations.rel_id
                              FROM sentences NATURAL JOIN relations
                              WHERE relations.rel_id = (SELECT min(rel_id) FROM relations);''')
        row = cursor.fetchone()

        # only return relation id if it wasn't passed in
        if rel_id:
            return row['sentence'], row['true_rel']
        else:
            return row['rel_id'], row['sentence'], row['true_rel']


if __name__ == '__main__':
    app.run(debug=True)
