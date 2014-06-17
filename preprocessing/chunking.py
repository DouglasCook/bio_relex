import nltk
import csv          # for reading and writing CSV files
import ast          # ast is needed to convert list from string to proper list
import re           # regular expressions
import os           # for filepath stuff


def chunk(f_in, f_out):
    """
    Chunk the POS tagged sentences using basic regex grammar
    The IOB tags include POS tags so should replace existing field in CSV
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    # TODO work out what directory structure to use...
    filepath = os.path.abspath(os.path.join(basepath, '..', f_in))
    file_out = os.path.abspath(os.path.join(basepath, '..', f_out))

    # TODO improve the grammar! should split into multiple rules
    # first need a grammar to define the chunks
    # can use nltk.app.chunkparser() to evaluate your grammar

    # don't know how this should be formatted?
    grammar = r"""
                NP: {(<DT>|<PRP.>|<POS>)?<CD>*<JJ.*>*<CD>*<NN.*>+}
                    {<PRP>}
               """
    cp = nltk.RegexpParser(grammar)

    with open(filepath, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
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

                # TODO clean up text more, remove stop words
                # strip out any punctuation at this point, chunking works better if some punctuation is left in
                result = [(x, y, z) for (x, y, z) in result if re.search(r'\w+', y)]

                # write row
                row[-1] = result
                csv_writer.writerow(row)

if __name__ == '__main__':
    chunk('data/sentences_POS.csv', 'data/sentences_chunk.csv')
