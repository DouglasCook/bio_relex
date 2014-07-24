import os


def build_filepath(caller, f_path):
    """
    Build full filepath based on location of calling function and relative filepath
    This means that the script can be called from anywhere and will still be able to find other directories?
    """
    basepath = os.path.dirname(caller)
    return os.path.abspath(os.path.join(basepath, f_path))


# TODO until I can work out the package bullshit this cannot be called from server etc
# so will need to stay duplicated in the code
def split_sentence(sent, start1, end1, start2, end2):
    """
    Split a sentence into before, between and after sections
    """
    return sent[:start1], sent[end1:start2], sent[end2:]
