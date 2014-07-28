import os

from eu_adr.WEKA import feature_extraction


def create_arff(filepath, relation, attributes):
    """
    Create arff file for use in WEKA, don't worry about comments or anything for now
    Requires lists of lists for attributes and data
    Dictionary seems like it could be useful to use here but has no order so maybe better to stick with list?
    """
    with open(filepath, 'w') as f:
        # create relation title
        f.write('@RELATION ' + relation)
        f.write('\n' * 2)

        # now add attributes and their types
        for [att, att_type] in attributes:
            f.write('@ATTRIBUTE ' + att + ' ' + att_type + '\n')
        f.write('\n@DATA\n')


def add_arff_data(filepath, data):
    """
    Add data to given arff file
    """
    with open(filepath, 'a') as f:
        # add data to data section
        for vector in data:
            vector = [str(v) for v in vector]
            f.write(', '.join(vector) + '\n')


def write_file(f_name, stem=False):
    """
    Step 4 in the pipeline so far...
    Write the weka file
    """
    # want to be able to save to different files easily
    basepath = os.path.dirname(__file__)
    file_out = os.path.abspath(os.path.join(basepath, 'WEKA/' + f_name + '.arff'))

    # write the header stuff
    create_arff(file_out, 'eu-adr', [['true_relation', '{True, False}'],
                                     ['sent_num', 'NUMERIC'],
                                     ['word_gap', 'NUMERIC'],
                                     ['rel_type', '{PA, SA, NA}'],
                                     ['type1', '{Drug, Disorder, Target}'],
                                     ['type2', '{Drug, Disorder, Target}'],
                                     # before first entity
                                     ['bef_words', 'STRING'],
                                     ['bef_bigrams', 'STRING'],
                                     ['bef_pos_tags', 'STRING'],
                                     ['bef_phrase_path', 'STRING'],
                                     ['bef_combo', 'STRING'],
                                     ['bef_np_count', 'NUMERIC'],
                                     ['bef_vp_count', 'NUMERIC'],
                                     ['bef_pp_count', 'NUMERIC'],
                                     # between entities
                                     ['bet_words', 'STRING'],
                                     ['bet_bigrams', 'STRING'],
                                     ['bet_pos_tags', 'STRING'],
                                     ['bet_phrase_path', 'STRING'],
                                     ['bet_combo', 'STRING'],
                                     ['bet_np_count', 'NUMERIC'],
                                     ['bet_vp_count', 'NUMERIC'],
                                     ['bet_pp_count', 'NUMERIC'],
                                     # after second entity
                                     ['aft_words', 'STRING'],
                                     ['aft_bigrams', 'STRING'],
                                     ['aft_pos_tags', 'STRING'],
                                     ['aft_phrase_path', 'STRING'],
                                     ['aft_combo', 'STRING'],
                                     ['aft_np_count', 'NUMERIC'],
                                     ['aft_vp_count', 'NUMERIC'],
                                     ['aft_pp_count', 'NUMERIC']])
    # add the data
    if stem:
        add_arff_data(file_out, feature_extraction.generate_features(True))
    else:
        add_arff_data(file_out, feature_extraction.generate_features())


if __name__ == '__main__':
    #file_name = raw_input('Enter file name ')
    write_file('test')
    write_file('test_stemmed', True)
