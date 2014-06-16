import nltk
import csv
import ast  # ast is needed to convert list from string to proper list


def chunk():
    """ Chunk the POS tagged sentences using basic regex grammar """

    # TODO improve the grammar! should split into multiple rules
    # first need a grammar to define the chunks
    # can use nltk.app.chunkparser() to evaluate your grammar
    # + matches one or more
    # * matches zero or more
    # . matches anything
    # ? means optional
    grammar = r"""
        NP: {(<DT>|<PRP.>|<POS>)?<CD>*<JJ.*>*<CD>*<NN.*>+}
            {<PRP>}
        """

    cp = nltk.RegexpParser(grammar)

    with open('sentences_POS.csv', 'rb') as csv_in:
        csv_reader = csv.reader(csv_in, delimiter=',')
        # skip column headers
        csv_reader.next()

        for i in xrange(10):
            row = csv_reader.next()
            # evaluating the string converts it back to list
            l = ast.literal_eval(row[2])
            print cp.parse(l)
            # can also draw the chunk tree
            #cp.parse(l).draw()


if __name__ == '__main__':
    chunk()
