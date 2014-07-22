import sqlite3

from reuters_NER import ontologies_api_demo as ner

db_path = 'database/test.db'


def test():
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        cursor.execute('''SELECT *
                          FROM sentences
                          WHERE source = 'pubmed';''')
        for i in xrange(20):
            for j in xrange(10):
                row = cursor.fetchone()

            print row['sentence']
            raw_xml = ner.runOntologiesSearch(row['sentence'])
            print raw_xml
            print ner.summariseXml(raw_xml)
            print '\n\n'


def relevant_into_temp():
    """
    Populate temporary table with relevant sentences from pubmed query
    NEED TO DO THIS ON THE SERVER!
    """
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        cursor.execute('''SELECT *
                          FROM sentences
                          WHERE source = 'pubmed';''')
        # need to fetch all here since we are using same cursor later for insert query
        sentences = cursor.fetchall()

        for row in sentences:
            # call the reuters ner
            raw_xml = ner.runOntologiesSearch(row['sentence'])
            entity_dict = ner.summariseXml(raw_xml)

            # need at least two entities for a relation
            if len(entity_dict) > 1:
                indication_present = False
                action_present = False

                for key in entity_dict:
                    if entity_dict[key][0] == 'Indication':
                        indication_present = True
                    elif entity_dict[key][0] == 'Action':
                        action_present = True

                    # if there is at least one action and one indication the sentence is relevant
                    if indication_present and action_present:
                        cursor.execute('''INSERT INTO relevant_sentences
                                                 VALUES (?, ?);''',
                                       (row['sent_id'], str(entity_dict)))
                        print row['sent_id']
                        # committing here will slow things down but means can break and still have SOME data
                        # slow down is negligible compared to calling NER engine
                        db.commit()
                        break


if __name__ == '__main__':
    #test()
    relevant_into_temp()
