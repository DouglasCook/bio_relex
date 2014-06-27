import os
import csv
import re


def entity_indices(entities, tags):
    """
    Given entities return indices in tags where their first words are found
    Return dictionary with entities as keys and indices as values
    """

    entity_dict = {}

    for entity in entities:
        # TODO fix this method, sometimes first word makes sense but sometimes identifies incorrect entities
        # locate first word if entity is made of multiple words
        space = entity.find(' ')
        if space > 0:
            head_word = entity[:space]
        else:
            head_word = entity
        # underscores are used in the tokens so need to replace before searching
        if head_word.find('-') > 0:
            head_word = head_word.replace('-', '_')

        # add indices of head word to dict entry for this drug
        # TODO does casting everything as lower case give correct results?
        #indices = [i for i, x in enumerate(tags) if x[0].lower() == head_word.lower()]
        indices = [i for i, x in enumerate(tags) if x[0] == head_word]
        if len(indices) > 0:
            entity_dict[entity] = indices

    return entity_dict


def entities_only(filename):
    """
    Extract all named entities from the given text file
    """

    with open(filename, 'r') as f_in:
        text = f_in.read()
        # regex to match words within tags
        #entities = re.findall('<.{,12}>.+?</', text)
        # below will not match any organisations, may be better for now for generating false examples
        entities = re.findall('<.{,8}>.+?</', text)
        # strip tags and remove duplicates
        entities = set([x[x.find('>')+1:x.rfind('<')] for x in entities])

    return entities


def drug_and_company_entities():
    """
    Locate named drugs and companies, indexed by word
    """

    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters/sentences_POS.csv'))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'reuters/entities_marked.csv'))

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, delimiter=',')
            csv_writer = csv.DictWriter(csv_out, ['SOURCE_ID', 'SENT_NUM', 'SENTENCE', 'DRUGS', 'COMPANIES',
                                                  'POS_TAGS'], delimiter=',')
            csv_writer.writeheader()

            for row in csv_reader:
                tags = eval(row['POS_TAGS'])
                # find indices for drugs and companies mentioned in the row
                drug_dict = entity_indices(eval(row['DRUGS']), tags)
                comp_dict = entity_indices(eval(row['COMPANIES']), tags)
                row.update({'DRUGS': drug_dict})
                row.update({'COMPANIES': comp_dict})

                # remove this field, think pop is the only way to do it
                row.pop('NO_PUNCT')
                csv_writer.writerow(row)

    print 'Written to entities_marked.csv'


def other_entities():
    """
    Locate named entities tagged by Stanford NER tool
    Text file must be created via bash script for now, really not a good way to do it
    """

    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters/entities_marked.csv'))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'reuters/all_entities_marked.csv'))

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, delimiter=',')
            csv_writer = csv.DictWriter(csv_out, ['SOURCE_ID', 'SENT_NUM', 'SENTENCE', 'DRUGS', 'COMPANIES', 'OTHER',
                                                  'POS_TAGS'], delimiter=',')
            csv_writer.writeheader()

            for row in csv_reader:
                # extract tagged entities from preprocessed file
                ne_filepath = os.path.abspath(os.path.join(basepath, '..', 'reuters/named_entities'))
                entities = entities_only(ne_filepath + '/' + row['SOURCE_ID'] + '.txt')

                entities_dict = {}
                drug_dict = eval(row['DRUGS'])
                comp_dict = eval(row['COMPANIES'])
                tags = eval(row['POS_TAGS'])

                for entity in entities:
                    # underscores are used in the tokens so need to replace before searching
                    if entity.find('-') > 0:
                        entity = entity.replace('-', '_')

                    words = entity.split()
                    tokens = [tok for (tok, tag) in tags]
                    indices = []

                    # search for whole entity in tokens
                    for i in range(len(tokens)):
                        if tokens[i:i+len(words)] == words:
                            indices.append(i)

                    # only add to other entities if it doesn't match an existing drug or company
                    #if len(indices) > 0 and indices not in drug_dict.values() and indices not in comp_dict.values():
                    # is the line below correct? adding two lists right?
                    if len(indices) > 0 and indices not in (drug_dict.values() + comp_dict.values()):
                        entities_dict[entity] = indices

                row.update({'OTHER': entities_dict})
                csv_writer.writerow(row)

    print 'Written to all_entities_marked.csv'

if __name__ == '__main__':
    drug_and_company_entities()
    other_entities()
    #stanford_input_split_sentences()
