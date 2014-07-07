import os
import urllib2


def scrape_pubmed():
    """
    Download XML version of abstracts used in EU-ADR data set
    """
    basepath = os.path.dirname(__file__)
    files = os.listdir('/Users/Dug/Imperial/individual_project/data/euadr_corpus')

    for f in files:
        pubmed_id = f.split('.')[0]
        # avoid random crap files eg ds_store
        if len(pubmed_id) > 0:
            # want xml version for easier processing
            url = 'http://www.ncbi.nlm.nih.gov/pubmed/?term=' + pubmed_id + '&report=xml&format=text'
            fp = os.path.abspath(os.path.join(basepath, 'abstracts', pubmed_id + '.xml'))

            raw = urllib2.urlopen(url).read()
            # replace HTML literals
            raw = raw.replace('&lt;', '<')
            raw = raw.replace('&gt;', '>')

            with open(fp, 'wb') as f_out:
                f_out.write(raw)


if __name__ == '__main__':
    scrape_pubmed()
