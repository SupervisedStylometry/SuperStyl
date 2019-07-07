from builtins import sum
import unidecode
import re
import nltk.tokenize

def count_words(text, feats = "words", n = 1, relFreqs = False):
    """
    Get word counts from  a text
    :param text:
    :return:
    """

    text = normalise(text)

    if feats == "words":
        tokens = nltk.tokenize.wordpunct_tokenize(text)

    counts = {}

    for t in tokens:
        if not t in counts.keys():
            counts[t] = 1

        else:
            counts[t] = counts[t] + 1

    if relFreqs:
        total = sum(counts.values())
        for t in counts.keys():
            counts[t] = counts[t] / total

    return counts


def normalise(text):
    # Remove all but word chars, remove accents, and normalise space
    # and then normalise unicode

    return unidecode.unidecode(re.sub("\s+", " ", re.sub("[\W0-9]+", " ", text.lower()).strip()))
