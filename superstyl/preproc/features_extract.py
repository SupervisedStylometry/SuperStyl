from builtins import sum
from collections import Counter
import nltk.tokenize
import nltk


def count_words(text, feats = "words", n = 1):
    """
    Get word counts from  a text
    :param text: the source text
    :param feats: the type of feats (words, chars, etc.)
    :param n: the length of n-grams
    :return: features absolute frequencies in text as a counter
    """

    if feats == "words":
        tokens = nltk.tokenize.wordpunct_tokenize(text)

        if n > 1:
            tokens = ["_".join(t) for t in list(nltk.ngrams(tokens, n))]

    if feats == "chars":
        tokens = list(text.replace(' ', '_'))
        if n > 1:
            tokens = ["".join(t) for t in list(nltk.ngrams(tokens, n))]

    counts = Counter()
    counts.update(tokens)

    return counts

def relative_frequencies(wordCounts):
    """
    For a counter of word counts, return the relative frequencies
    :param wordCounts: a dictionary of word counts
    :return a counter of word relative frequencies
    """

    total = sum(wordCounts.values())
    for t in wordCounts.keys():
        wordCounts[t] = wordCounts[t] / total

    return wordCounts


def get_feature_list(myTexts, feats="words", n=1, relFreqs=True):
    """
    :param myTexts: a 'myTexts' object, containing documents to be processed
    :param feat_list: a list of features to be selected
    :param feats: type of feats (words, chars)
    :param n: n-grams length
    :return: list of features, with total frequency
    """
    my_feats = Counter()

    for text in myTexts:
        counts = count_words(text["text"], feats=feats, n=n)

        my_feats.update(counts)

    if relFreqs:
        my_feats = relative_frequencies(my_feats)

    # sort them
    my_feats = [(i, my_feats[i]) for i in sorted(my_feats, key=my_feats.get, reverse=True)]

    return my_feats


def get_counts(myTexts, feat_list=None, feats = "words", n = 1, relFreqs = False):
    """
    Get counts for a collection of texts
    :param myTexts: the document collection
    :param feat_list: a list of features to be selected (None for all)
    :param feats: the type of feats (words, chars, etc.)
    :param n: the length of n-grams
    :param relFreqs: whether to compute relative freqs
    :return: the collection with, for each text, a 'wordCounts' dictionary
    """

    for i in enumerate(myTexts):

        counts = count_words(myTexts[i[0]]["text"], feats=feats, n=n)

        if relFreqs:
            counts = relative_frequencies(counts)

        if feat_list:
            # and keep only the ones in the feature list
            counts = {f: counts[f] for f in feat_list if f in counts.keys()}

        myTexts[i[0]]["wordCounts"] = counts

    return myTexts
