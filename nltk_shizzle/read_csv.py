import nltk
import csv              # used for accessing data held in CSV format
import os.path          # need this to use relative filepaths

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
