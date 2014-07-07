import os
import csv
import xml.etree.cElementTree as ET     # python XML manipulation library, C version because it's way faster!


def convert_to_csv():
    """
    Create one record per text
    """
    # TODO parse the HTML properly, maybe use beautiful soup?
    basepath = os.path.dirname(__file__)
    file_path = os.path.abspath(os.path.join(basepath, 'abstracts'))
    files = os.listdir(file_path)

    # need to deal with the fucking DS store file...
    files = [os.path.abspath(os.path.join(basepath, 'abstracts', f)) for f in files if f != '.DS_Store']
    f_out = os.path.abspath(os.path.join(basepath, 'csv', 'texts.csv'))

    with open(f_out, 'wb') as csv_out:
        csv_writer = csv.DictWriter(csv_out, ['id', 'text'], delimiter=',')
        csv_writer.writeheader()

        for f in files:
            # parse the tree
            tree = ET.parse(f)

            # use xpath paths here to search entire tree
            pubmed_id = tree.findtext('.//PMID')
            title = tree.findtext('.//ArticleTitle')
            abstract = ' '.join([e.text for e in tree.findall('.//AbstractText')])

            # dict comprehension here to hack the unicode into csv writer
            dict_row = {'id': pubmed_id, 'text': title + ' ' + abstract}
            row = {key: value.encode('utf-8') for key, value in dict_row.items()}
            csv_writer.writerow(row)

if __name__ == '__main__':
    convert_to_csv()
