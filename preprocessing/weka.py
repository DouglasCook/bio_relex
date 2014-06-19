def create_arff(filepath, relation, attributes, data):
    """
    Create arff file for use in WEKA, don't worry about comments or anything for now
    Requires lists of lists for attributes and data
    """

    with open(filepath, 'w') as f:
        # create relation title
        f.write('@RELATION ' + relation)
        f.write('\n'*2)

        # now add attributes and their types
        for [att, att_type] in attributes:
            f.write('@ATTRIBUTE ' + att + ' ' + att_type + '\n')
        f.write('\n')

        # finally add data section
        f.write('@DATA\n')
        for vector in data:
            vector = [str(v) for v in vector]
            f.write(', '.join(vector) + '\n')


def test():
    with open('test.txt', 'w') as f:
        f.write('a' + 'b' + 'c')
        f.write('\n'*2)
        f.write('newline?')

if __name__ == '__main__':
    create_arff('test.arff', 'testing the water', [['test1', 'STRING'],['test2', 'NUMERIC']], [['testing', 2],['no', 3]])
