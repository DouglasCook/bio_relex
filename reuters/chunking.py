import nltk
import csv          # for reading and writing CSV files
import ast          # ast is needed to convert list from string to proper list
import re           # regular expressions


def chunk():
    """ Chunk the POS tagged sentences using basic regex grammar
        The IOB tags include POS tags so should replace existing field in CSV """

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
        with open('sentences_chunk.csv', 'wb') as csv_out:
            csv_reader = csv.reader(csv_in, delimiter=',')
            csv_writer = csv.writer(csv_out, delimiter=',')

            # write column headers
            row = csv_reader.next()
            row[-1] = 'CHUNKS'
            #row.append('CHUNKS')
            csv_writer.writerow(row)

            for row in csv_reader:
                # evaluating the string converts it back to list
                l = ast.literal_eval(row[-1])
                result = cp.parse(l)
                # can also draw the tree with result.draw()
                # convert to IOB tags
                result = nltk.chunk.util.tree2conlltags(result)

                # strip out any punctuation at this point, chunking works better if some punctuation is left in
                result = [(x, y, z) for (x, y, z) in result if re.search(r'\w+', y)]

                # write row
                row[-1] = result
                csv_writer.writerow(row)


if __name__ == '__main__':
    chunk()
