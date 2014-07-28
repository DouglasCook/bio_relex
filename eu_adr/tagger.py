import re
import nltk


class TaggerChunker(nltk.ChunkParserI):
    """
    Bigram based chunker
    Should give reasonable results and require much less work than regex based one
    Actually doesn't work particularly well... maybe should use Stanford instead?
    """
    def __init__(self):
        """
        The constructor takes a training data set and trains the classifier
        """
        train_sents = nltk.corpus.conll2000.chunked_sents('train.txt')
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]

        self.tagger = nltk.BigramTagger(train_data)
        self.stemmer = nltk.SnowballStemmer('english')

    def parse(self, sentence):
        """
        Takes POS tagged sentence and returns a chunk tree
        """
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag) in zip(sentence, chunktags)]
        return conlltags

    def pos_and_chunk_tags(self, text):
        """
        Return word, pos tag, chunk triples
        """
        # TODO remove undesirable words or tokens here? also remove CONCLUSION, METHODS etc since they aren't important
        # TODO want to chunk whole sentence before splitting into parts, should give better results
        text = nltk.word_tokenize(text)
        # text = [b for b in between if b not in stopwords]
        tags = nltk.pos_tag(text)
        chunks = self.parse(tags)

        # now want to remove any punctuation - maybe don't want to remove absolutely all punctuation?
        # match returns true without needing to match whole string
        chunks = [c for c in chunks if not re.match('\W', c[0])]

        # stemming - not sure about the encode decode nonsense...
        chunks = [(self.stemmer.stem(c[0].decode('utf-8')), c[1], c[2]) for c in chunks]
        chunks = [(c[0].encode('utf-8'), c[1], c[2]) for c in chunks]

        return chunks
