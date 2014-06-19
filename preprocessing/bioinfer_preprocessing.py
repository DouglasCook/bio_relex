import csv
import os
import xml.etree.cElementTree as ET        # python XML manipulation library, C version because it's way faster!


def remove_extra_crap():
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', '..','data/bioinfer/BioInfer_1.2_b.xml'))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'bioinfer/bioinfer_1.2.xml'))

    # first parse the tree
    tree = ET.parse(file_in)
    root = tree.getroot()

    # remove ontology tags, don't need them
    rubbish = tree.findall('ontology')
    for r in rubbish:
        root.remove(r)

    # remove all the linkage info, not sure how the format works
    for child in tree.find('sentences'):
        links = child.find('linkages')
        child.remove(links)

    # write stripped version to file
    tree.write(file_out)

if __name__ == '__main__':
    remove_extra_crap()
