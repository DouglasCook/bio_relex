import csv
import os


def single_sentence_relations():
    """
    Extract all sentences containing both drug and company ie our easy true relations
    """

    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters_new/all_entities_marked.csv'))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'reuters_new/single_sentence_relations.csv'))

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, delimiter=',')
            csv_writer = csv.DictWriter(csv_out, ['DRUGS', 'COMPANIES', 'SENTENCE', 'SOURCE_ID', 'SENT_NUM', 'OTHER',
                                                  'POS_TAGS'], delimiter=',')
            csv_writer.writeheader()

            for row in csv_reader:
                # need to use eval to get dicts back, this is horrible must find better method than CSV for everything
                if len(eval(row['DRUGS'])) > 0 and len(eval(row['COMPANIES'])) > 0:
                #if len(row['DRUGS'])*len(row['COMPANIES']) > 0:
                    csv_writer.writerow(row)

    print 'Written to single_sentence_relations.csv'


def single_sentence_non_relations():
    """
    Extract all sentences containing two or more 'other' entities
    Problem going to arise when the entities have multiple names
    """

    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters_new/all_entities_marked.csv'))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'reuters_new/single_sentence_non_relations.csv'))

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, delimiter=',')
            csv_writer = csv.DictWriter(csv_out, ['OTHER', 'SENTENCE', 'SOURCE_ID', 'SENT_NUM', 'DRUGS', 'COMPANIES',
                                                  'POS_TAGS'], delimiter=',')
            csv_writer.writeheader()

            for row in csv_reader:
                # need to use eval to get dicts back, this is horrible must find better method than CSV for everything
                # TODO better generation of other entity pairs
                # for now just using those with a single pair of entities
                if len(eval(row['OTHER'])) == 2:
                    csv_writer.writerow(row)

    print 'Written to single_sentence_non_relations.csv'


def generate_attributes():
    """
    Create list of feature vectors for given entity pairs
    """

    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters_new/single_sentence_relations.csv'))

    feature_vectors = []

    with open(file_in, 'rb') as csv_in:
        csv_reader = csv.DictReader(csv_in, delimiter=',')

        for row in csv_reader:
            drug_dict = eval(row['DRUGS'])
            comp_dict = eval(row['COMPANIES'])

            # consider all drug-company pairs
            for drug in drug_dict.keys():
                for comp in comp_dict.keys():

                    # generate all pairs of indices
                    pairs = [(x, y) for x in drug_dict[drug] for y in comp_dict[comp]]
                    distance = [abs(x - y) for (x, y) in pairs]
                    #print 'nearest index', distance.index(min(distance))
                    #print 'closest pair', pairs[distance.index(min(distance))]

                    # add feature vector for this pair
                    feature_vectors.append([row['SENT_NUM'], min(distance), 'yes'])

    return feature_vectors


def generate_attributes_no_relation():
    """
    Create list of feature vectors for given entity pairs, these are the non-related for now
    """

    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters_new/single_sentence_non_relations.csv'))

    feature_vectors = []

    with open(file_in, 'rb') as csv_in:
        csv_reader = csv.DictReader(csv_in, delimiter=',')

        for row in csv_reader:
            other_dict = eval(row['OTHER'])

            # consider all pairs of entities - this is a crap way to do it
            for e1 in other_dict.keys():
                for e2 in other_dict.keys():

                    if e1 != e2:
                        # generate all pairs of indices
                        pairs = [(x, y) for x in other_dict[e1] for y in other_dict[e2]]
                        distance = [abs(x - y) for (x, y) in pairs]
                        #print 'nearest index', distance.index(min(distance))
                        #print 'closest pair', pairs[distance.index(min(distance))]

                        # add feature vector for this pair
                        feature_vectors.append([row['SENT_NUM'], min(distance), 'no'])

    return feature_vectors


if __name__ == '__main__':
    #single_sentence_non_relations()
    #generate_attributes()
    generate_attributes_no_relation()
