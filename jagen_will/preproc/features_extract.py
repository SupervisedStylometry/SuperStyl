from builtins import sum

import nltk.tokenize

def count_words(text, relFreqs = False):
    """
    Get word counts from  a text
    :param text:
    :return:
    """

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
