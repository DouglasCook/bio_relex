import sqlite3

from reuters_NER import ontologies_api_demo as ner

db_path = 'database/test.db'


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
                disorder_present = False
                treatment_present = False

                # TODO may want to be certain that these are the only entities we are interested in
                for key in entity_dict:
                    if entity_dict[key][0] == 'Indication':
                        disorder_present = True
                    elif entity_dict[key][0] == 'Action' or entity_dict[key][0] == 'Drug':
                        treatment_present = True

                    # if there is at least one action and one indication the sentence is relevant
                    if disorder_present and treatment_present:
                        cursor.execute('''INSERT INTO relevant_sentences
                                                 VALUES (?, ?);''',
                                       (row['sent_id'], str(entity_dict)))
                        print row['sent_id']
                        # committing here will slow things down but means can break and still have SOME data
                        # slow down is negligible compared to calling NER engine
                        db.commit()
                        break


def tag_sentences():
    """
    Populate temporary table with relevant sentences from pubmed query
    NEED TO DO THIS ON THE SERVER!
    """
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        cursor.execute('''SELECT *
                          FROM relevant_sentences;''')
        sentences = cursor.fetchall()

        for row in sentences:
            sent_id = row['sent_id']
            print row['entity_dict']
            entity_dict = eval(row['entity_dict'])
            # create lists for disorders and treatments
            disorders = [k for k in entity_dict if k[0] == 'Indication']
            treatments = [k for k in entity_dict if k[0] in ['Action', 'Drug']]
            print disorders
            print treatments


if __name__ == '__main__':
    #test()
    #relevant_into_temp()
    tag_sentences()
