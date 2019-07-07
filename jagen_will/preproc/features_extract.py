from builtins import sum
import unidecode
import re
import nltk.tokenize
from nltk.util import ngrams

def count_words(text, feats = "words", n = 1, relFreqs = False):
    """
    Get word counts from  a text
    :param text: the source text
    :param feats: the type of feats (words, chars, etc.)
    :param n: the length of n-grams
    :param relFreqs: whether or not to compute relative freqs
    :return: feature frequencies in text
    """

    if feats == "words":
        tokens = nltk.tokenize.wordpunct_tokenize(text)

        if n > 1:
            tokens = ["_".join(t) for t in list(ngrams(tokens, n))]

    if feats == "chars":
        tokens = [c for c in text.replace(' ', '_')]
        if n > 1:
            tokens = ["".join(t) for t in list(ngrams(tokens, n))]

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
