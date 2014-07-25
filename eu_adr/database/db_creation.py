import sqlite3
import csv


def create_tables():
    """
    Create tables to hold the data
    """
    with sqlite3.connect('relex.db') as db:
        cursor = db.cursor()

        # drop the tables
        cursor.execute('DROP TABLE sentences;')
        cursor.execute('DROP TABLE relations;')
        cursor.execute('DROP TABLE users;')
        cursor.execute('DROP TABLE decisions;')

        # table for sentences themselves, don't know if this is necessary since the tags etc are per relation?
        cursor.execute('''CREATE TABLE sentences(sent_id INTEGER PRIMARY KEY,
                                                 pubmed_id INTEGER,
                                                 sent_num INTEGER,
                                                 sentence TEXT,
                                                 source TEXT);''')

        # table for the relations
        cursor.execute('''CREATE TABLE relations(rel_id INTEGER PRIMARY KEY,
                                                 sent_id INTEGER,
                                                 true_rel BOOL,
                                                 bad_ner BOOL,
                                                 entity1 TEXT,
                                                 type1 TEXT,
                                                 start1 INTEGER,
                                                 end1 INTEGER,
                                                 entity2 TEXT,
                                                 type2 TEXT,
                                                 start2 INTEGER,
                                                 end2 INTEGER,
                                                 before_tags TEXT,
                                                 between_tags TEXT,
                                                 after_tags TEXT,
                                                 FOREIGN KEY(sent_id) REFERENCES sentences);''')

        # table for annotators
        cursor.execute('''CREATE TABLE users(user_id INTEGER PRIMARY KEY,
                                             name TEXT,
                                             type TEXT);''')

        # table for annotators decision
        cursor.execute('''CREATE TABLE decisions(decision_id INTEGER PRIMARY KEY,
                                                 rel_id INTEGER,
                                                 user_id INTEGER,
                                                 decision INTEGER,
                                                 FOREIGN KEY(rel_id) REFERENCES relations,
                                                 FOREIGN KEY(user_id) REFERENCES users);''')


def create_temp_sentences():
    """
    Table to store relevant sentences from pubmed query before processing
    """
    # TODO rename this as temp
    with sqlite3.connect('test.db') as db:
        cursor = db.cursor()
        cursor.execute('DROP TABLE relevant_sentences')
        # table for annotators decision
        cursor.execute('''CREATE TABLE relevant_sentences(sent_id INTEGER,
                                                          entity_dict TEXT,
                                                          FOREIGN KEY(sent_id) REFERENCES sentences);''')


def create_classifier_table():
    """
    Table to store training set used for each classifier
    """
    with sqlite3.connect('test.db') as db:
        cursor = db.cursor()
        cursor.execute('DROP TABLE classifier_data')
        # table for annotators decision
        cursor.execute('''CREATE TABLE classifier_data(clsf_id INTEGER,
                                                       training_rel INTEGER,
                                                       FOREIGN KEY(clsf_id) REFERENCES users(user_id),
                                                       FOREIGN KEY(training_rel) REFERENCES relations(rel_id));''')


def populate_sentences():
    """
    Populate the sentences table with initial set of sentences from biotext and eu-adr corpora
    """
    with open('../csv/tagged_sentences_stemmed.csv', 'rb') as f_in:
        csv_reader = csv.DictReader(f_in, delimiter=',')
        pid = 0
        sent_num = 0

        with sqlite3.connect('relex.db') as db:
            cursor = db.cursor()

            for row in csv_reader:
                # this isn't a great way to do it but since the spreadsheet is ordered it will work
                if row['pid'] != pid or row['sent_num'] != sent_num:
                    # set the source, this should make it easier to query new records later
                    if eval(row['pid']) < 1000:
                        src = 'Biotext'
                    else:
                        src = 'EU-ADR'
                    cursor.execute('INSERT INTO sentences VALUES (NULL, ?, ?, ?, ?);',
                                   # sentences saved in utf-8 but sqlite wants unicode -> need to decode
                                   (row['pid'], row['sent_num'], row['sentence'].decode('utf-8'), src))
                pid = row['pid']
                sent_num = row['sent_num']


def populate_relations():
    """
    Populate the relations table with set of 'correctly' annotated relations
    """
    with open('../csv/tagged_sentences_stemmed.csv', 'rb') as f_in:
        csv_reader = csv.DictReader(f_in, delimiter=',')

        # TODO should I just be connecting once here or multiple times?
        with sqlite3.connect('relex.db') as db:
            cursor = db.cursor()

            for row in csv_reader:
                # retrieve sentence ID for this sentence
                cursor.execute('''SELECT sent_id
                                  FROM sentences
                                  WHERE pubmed_id = ? AND sent_num = ?''', (row['pid'], row['sent_num']))
                try:
                    sent_id = cursor.fetchone()[0]
                except:
                    print 'pid =', row['pid']
                    print 'sent_num =', row['sent_num']
                    return 0

                cursor.execute('INSERT INTO relations VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);',
                               (sent_id,
                                row['true_relation'],
                                0,  # this is the bad NER field, false for the original relations
                                row['e1'].decode('utf-8'),
                                row['type1'],
                                row['start1'],
                                row['end1'],
                                row['e2'].decode('utf-8'),
                                row['type2'],
                                row['start2'],
                                row['end2'],
                                row['before_tags'].decode('utf-8'),
                                row['between_tags'].decode('utf-8'),
                                row['after_tags'].decode('utf-8')))


def populate_users():
    """
    Populate the decisions table to record 'correct' annotations from the corpora
    """
    with sqlite3.connect('relex.db') as db:
        cursor = db.cursor()

        # this deals with the biotext entries, they have artificial pubmed ids all < 1000
        # biotext annotator id is 1
        cursor.execute('''INSERT INTO users
                                 VALUES (0, 'Douglas', 'testing'), (1, 'Andrew', 'user');''')


def populate_decisions():
    """
    Populate the decisions table to record 'correct' annotations from the corpora
    """
    with sqlite3.connect('relex.db') as db:
        cursor = db.cursor()

        # this deals with the biotext entries, they have artificial pubmed ids all < 1000
        # biotext annotator id is 1
        cursor.execute('''INSERT INTO decisions
                                 SELECT rel_id,
                                        1 as annotator_id,
                                        true_rel
                                 FROM relations NATURAL JOIN SENTENCES
                                 WHERE SENTENCES.source = 'Biotext';''')

        # now add eu-adr records
        cursor.execute('''INSERT INTO decisions
                                 SELECT rel_id,
                                        2 as annotator_id,
                                        true_rel
                                 FROM relations NATURAL JOIN sentences
                                 WHERE SENTENCES.source = 'EU-ADR';''')


def initial_setup():
    """
    Set up database based on preprocessed sentences from existing corpora
    """
    create_tables()
    populate_sentences()
    populate_relations()
    populate_users()
    # don't need to have the original decisions recorded here now there is source field in sentences
    #populate_decisions()

if __name__ == '__main__':
    #create_tables()
    #populate_sentences()
    #populate_relations()
    #populate_decisions()
    #initial_setup()
    #create_temp_sentences()
    create_classifier_table()
