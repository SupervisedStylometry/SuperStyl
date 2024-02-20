from builtins import sum
from collections import Counter
import nltk.tokenize
import nltk


def count_words(text, feats = "words", n = 1):
    """
    Get feature counts from  a text (words, chars or POS n-grams)
    :param text: the source text
    :param feats: the type of feats: words, chars, POS (supported only for English)
    :param n: the length of n-grams
    :return: features absolute frequencies in text as a counter
    """
    # Should this be called count_words ? It counts other features as well... count_features ? It's just a grep and replace away.

    if feats == "words":
        tokens = nltk.tokenize.wordpunct_tokenize(text)
        if n > 1:
            tokens = ["_".join(t) for t in list(nltk.ngrams(tokens, n))]

    elif feats == "chars":
        tokens = list(text.replace(' ', '_'))
        if n > 1:
            tokens = ["".join(t) for t in list(nltk.ngrams(tokens, n))]

    #POS in english with NLTK - need to propose spacy later on
    elif feats == "pos":
        words = nltk.tokenize.word_tokenize(text)
        pos_tags = [pos for word, pos in nltk.pos_tag(words)]
        if n > 1:
            tokens = ["_".join(t) for t in list(nltk.ngrams(pos_tags, n))]
        else:
            tokens = pos_tags

    # Adding sentence length ; still commented as it is a work in progress, an integer won't do, a quantile would be better
    #elif feats == "sentenceLength":
    #    sentences = nltk.tokenize.sent_tokenize(text)
    #    tokens = [str(len(nltk.tokenize.word_tokenize(sentence))) for sentence in sentences]

    #Adding an error message in case some distracted guy like me would enter something wrong:
    else:
        raise ValueError("Unsupported feature type. Choose from 'words', 'chars', or 'pos'.")

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
    :param feats: type of feats (words, chars, POS)
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
