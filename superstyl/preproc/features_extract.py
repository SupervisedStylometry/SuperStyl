# first line only necessary for older Python versions
from builtins import sum
from collections import Counter
import nltk.tokenize
from nltk.util import ngrams
import nltk
import regex as re
#the download is a one time operation: should be placed elsewhere but I put it here for now
nltk.download('averaged_perceptron_tagger')

# Defining a function for n-gram generation
def generate_ngrams(tokens, n):
    """
    Generate n-grams from a list of tokens.
    """
    if n == 1:
        return tokens
    else:
        return ["_".join(t) for t in list(ngrams(tokens, n))]


def count_features(text, feats ="words", n = 1):
    """
    Get feature counts from  a text (words, chars or POS n-grams, or affixes(+punct if keep_punct),
    following Sapkota et al., NAACL 2015
    :param text: the source text
    :param feats: the type of feats: words, chars, POS (supported only for English), or affixes
    :param n: the length of n-grams
    :return: features absolute frequencies in text as a counter, and the total of frequencies
    """

    if not isinstance(text, str):
        raise ValueError("Text must be a string.")
    if not text:
        raise ValueError("Text cannot be empty.")
    if n < 1 or not isinstance(n, int):
        raise ValueError("n must be a positive integer.")
    if feats not in ["words", "chars", "affixes", "pos"]:
        raise ValueError("Unsupported feature type. Choose from 'words', 'chars', 'affixes', or 'pos'.")

    tokens = []
    if feats == "words":
        tokens = nltk.tokenize.wordpunct_tokenize(text)
        tokens = generate_ngrams(tokens, n)
    
    elif feats == "chars":
        # Directly generating character n-grams without regex 
        tokens = [text[i:i+n] for i in range(len(text)-n+1)]
    
    elif feats == "affixes":
        words = nltk.tokenize.wordpunct_tokenize(text)
        # Directly extract affixes based on n
        affixes = [word[:n] for word in words if len(word) > n] + [word[-n:] for word in words if len(word) > n]
        tokens.extend(affixes)
    
    elif feats == "pos":
        words = nltk.tokenize.word_tokenize(text)
        pos_tags = [pos for _, pos in nltk.pos_tag(words)]
        tokens = generate_ngrams(pos_tags, n)

    # Adding sentence length ; still commented as it is a work in progress, an integer won't do, a quantile would be better
    #elif feats == "sentenceLength":
    #    sentences = nltk.tokenize.sent_tokenize(text)
    #    tokens = [str(len(nltk.tokenize.word_tokenize(sentence))) for sentence in sentences]

    counts = Counter(tokens)
    total = sum(counts.values())

    return counts, total


def relative_frequencies(wordCounts, total):
    """
    For a counter of word counts, return the relative frequencies
    :param wordCounts: a dictionary of word counts
    :param total, the total number of features
    :return a counter of word relative frequencies
    """
    # Validate input types
    if not isinstance(wordCounts, Counter):
        raise TypeError("wordCounts must be a Counter object.")
    if not (isinstance(total, int) or isinstance(total, float)):
        raise TypeError("total must be an integer or float.")
    if total < 0:
        raise ValueError("total must not be negative.")
    
    if total > 0:  # Avoid division by zero
        for t in wordCounts.keys():
            wordCounts[t] = wordCounts[t] / total
    else:
        print("Warning: Total count is 0. Relative frequencies have not been calculated.")
    
    return wordCounts


def get_feature_list(myTexts, feats="words", n=1, relFreqs=True):
    """
    :param myTexts: a 'myTexts' object, containing documents to be processed
    :param feat_list: a list of features to be selected
    :param feats: type of feats (words, chars, affixes or POS)
    :param n: n-grams length
    :return: list of features, with total frequency
    """
    my_feats = Counter()
    total = 0

    for text in myTexts:
        counts, text_total = count_features(text["text"], feats=feats, n=n)

        my_feats.update(counts)
        total = total + text_total

    if relFreqs:
        my_feats = relative_frequencies(my_feats, total)

    # sort them
    my_feats = [(i, my_feats[i]) for i in sorted(my_feats, key=my_feats.get, reverse=True)]

    return my_feats


def get_counts(myTexts, feat_list=None, feats = "words", n = 1, relFreqs = False):
    """
    Get counts for a collection of texts
    :param myTexts: the document collection
    :param feat_list: a list of features to be selected (None for all)
    :param feats: the type of feats (words, chars, affixes, POS)
    :param n: the length of n-grams
    :param relFreqs: whether to compute relative freqs
    :return: the collection with, for each text, a 'wordCounts' dictionary
    """

    for i in enumerate(myTexts):

        counts, total = count_features(myTexts[i[0]]["text"], feats=feats, n=n)

        if relFreqs:
            counts = relative_frequencies(counts, total)

        if feat_list:
            # and keep only the ones in the feature list
            counts = {f: counts[f] for f in feat_list if f in counts.keys()}

        myTexts[i[0]]["wordCounts"] = counts

    return myTexts
