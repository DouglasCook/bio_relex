import nltk
import csv              # used for accessing data held in CSV format
import os.path          # need this to use relative filepaths

def test():
    # in order to access a file relative to current directory need to build a path like this...
    basepath = os.path.dirname(__file__)
    filepath = 'data/reuters/press_releases/PR_drug_company_500.csv'
    filepath = os.path.abspath(os.path.join(basepath, '..', '..', filepath))

    # using with ... as here insures that the file will be closed at the end
    # it is a sort of shorthand for a try ... finally block
    with open(filepath, 'rb') as csvfile:

        # the reader is an iterable object
        csvreader = csv.reader(csvfile, delimiter=',')
        
        # skip first row since it just contains names of columns
        csvreader.next()
        record = csvreader.next()

        # second field is the text, clean_html removes HTML tags
        raw = nltk.clean_html(record[1])
        tokens = nltk.word_tokenize(raw)
        text = nltk.Text(tokens)

        # part-of-speech tagging for all tokens
        tagged = nltk.pos_tag(tokens) 
        tag_fd = nltk.FreqDist(tag for (word, tag) in tagged)
        tag_fd.plot()

        # returns words that are in similar context to the given word
        #print text.similar('repifermin')


def preprocess():
    """ Reduce dataset to only those where text contains mentions of both entities """

    # set filepath to input
    basepath = os.path.dirname(__file__)
    filepath = 'data/reuters/press_releases/PR_drug_company_500.csv'
    filepath = os.path.abspath(os.path.join(basepath, '..', '..', filepath))

    with open(filepath, 'rb') as csv_in:

        with open('out.csv', 'wb') as csv_out:
            csv_reader = csv.reader(csv_in, delimiter=',')
            csv_writer = csv.writer(csv_out, delimiter=',')

            # write column headers on first row
            csv_writer.writerow(csv_reader.next())

            for row in csv_reader:
                add = True
                # remove HTML tags
                clean_text = nltk.clean_html(row[1])

                # check if the drug is mentioned in the text
                if row[3] not in clean_text:
                    print row[0], row[3]
                    add = False
                # check if company is mentioned in text
                if row[5] not in clean_text:
                    print row[0], row[5]
                    add = False
                
                if add:
                    # write new row to file
                    out_row = [row[0], clean_text, row[2], row[3], row[4], row[5]]
                    csv_writer.writerow(out_row)

if __name__ == '__main__':
    preprocess()
