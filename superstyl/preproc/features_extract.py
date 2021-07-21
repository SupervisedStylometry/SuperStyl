from builtins import sum
from collections import Counter
import nltk.tokenize
import nltk


def count_words(text, feat_list=None, feats = "words", n = 1, relFreqs = False):
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
            tokens = ["_".join(t) for t in list(nltk.ngrams(tokens, n))]

    if feats == "chars":
        tokens = list(text.replace(' ', '_'))
        if n > 1:
            tokens = ["".join(t) for t in list(nltk.ngrams(tokens, n))]

    counts = {}

    for t in tokens:
        if t not in counts.keys():
            counts[t] = 1

        else:
            counts[t] = counts[t] + 1

    if relFreqs:
        total = sum(counts.values())
        for t in counts.keys():
            if counts[t] > 0:
                counts[t] = counts[t] / total
            else:
                counts[t] = 0

    if feat_list:
        # and keep only the ones in the feature list
        counts = {f: counts[f] for f in feat_list if f in counts.keys()}

    return counts


def get_feature_list(myTexts, feats="words", n=1, relFreqs=True):
    """

    :param myTexts: a 'myTexts' object, containing documents to be processed
    :param feats: type of feats (words, chars)
    :param n: n-grams length
    :return: list of features, with total frequency
    """
    my_feats = Counter()

    for text in myTexts:
        counts = count_words(text["text"], feats=feats, n=n, relFreqs=relFreqs)

        my_feats.update(counts)

    # sort them

    my_feats = [(i, my_feats[i]) for i in sorted(my_feats, key=my_feats.get, reverse=True)]

    return my_feats


def get_counts(myTexts, feat_list, feats = "words", n = 1, relFreqs = False):
    """
    Get counts for a collection of texts
    :param myTexts: the document collection
    :param feats: the type of feats (words, chars, etc.)
    :param n: the length of n-grams
    :param relFreqs: whether or not to compute relative freqs
    :return: the collection with, for each text, a 'wordCounts' dictionary
    """

    for i in enumerate(myTexts):
        myTexts[i[0]]["wordCounts"] = count_words(
            myTexts[i[0]]["text"], feat_list=feat_list, feats=feats, n=n, relFreqs=relFreqs)

    return myTexts
