import tagging
import chunking


def boom():
    tagging.clean_and_tag()
    chunking.chunk()

if __name__ == '__main__':
    boom()