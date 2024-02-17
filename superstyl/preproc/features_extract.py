from builtins import sum
from collections import Counter
import nltk.tokenize
import nltk


def count_words(text, feat_list=None, feats = "words", n = 1, relFreqs = False):
    """
    Get word counts from  a text
    :param text: the source text
    :param feat_list: a list of features to be selected
    :param feats: the type of feats: words, chars, POS (supported only for English)
    :param n: the length of n-grams
    :param relFreqs: whether to compute relative freqs
    :return: feature frequencies in text
    """
    # Should this be called count_words ? It counts other features as well... count_features ? It's just a grep and replace away.
    # Same for the first sentence of the paragraph that I find confusing.

    if feats == "words":
        tokens = nltk.tokenize.wordpunct_tokenize(text)

        if n > 1:
            tokens = ["_".join(t) for t in list(nltk.ngrams(tokens, n))]

    elif feats == "chars":
        tokens = list(text.replace(' ', '_'))
        if n > 1:
            tokens = ["".join(t) for t in list(nltk.ngrams(tokens, n))]

    #Adding POS for English language with NLTK
    
    elif feats == "pos":
        words = nltk.tokenize.word_tokenize(text)
        pos_tags = [pos for word, pos in nltk.pos_tag(words)]
        if n > 1:
            tokens = ["_".join(t) for t in list(nltk.ngrams(pos_tags, n))]
        else:
            tokens = pos_tags

    #Adding an error message in case some distracted guy like me would enter something wrong:
    else:
        raise ValueError("Unsupported feature type. Choose from 'words', 'chars', or 'pos'.")


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
    :param feat_list: a list of features to be selected
    :param feats: type of feats (words, chars, POS)
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
    :param feat_list: a list of features to be selected
    :param feats: the type of feats (words, chars, etc.)
    :param n: the length of n-grams
    :param relFreqs: whether to compute relative freqs
    :return: the collection with, for each text, a 'wordCounts' dictionary
    """

    for i in enumerate(myTexts):
        myTexts[i[0]]["wordCounts"] = count_words(
            myTexts[i[0]]["text"], feat_list=feat_list, feats=feats, n=n, relFreqs=relFreqs)

    return myTexts
