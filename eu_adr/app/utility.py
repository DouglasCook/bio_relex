import os


def build_filepath(caller, f_path):
    """
    Build full filepath based on location of calling function and relative filepath
    This means that the script can be called from anywhere and will still be able to find other directories?
    """
    basepath = os.path.dirname(caller)
    return os.path.abspath(os.path.join(basepath, f_path))
