# first line only necessary for older Python versions
from builtins import sum
from collections import Counter
import nltk.tokenize
import nltk
import regex as re

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
    if feats not in ["words", "chars", "affixes", "pos", "met_line", "met_syll"]:
        raise ValueError("Unsupported feature type. Choose from 'words', 'chars', 'affixes', 'met_line', 'met_syll', or 'pos'.")
    if feats == "words":
        tokens = nltk.tokenize.wordpunct_tokenize(text)
        if n > 1:
            tokens = ["_".join(t) for t in list(nltk.ngrams(tokens, n))]
        total = len(tokens)

    elif feats in ["chars", "met_syll"]:
        tokens = [re.sub(r'\p{Z}', '_', ''.join(ngram)) for ngram in nltk.ngrams(text, n)]
        total = len(tokens)

    elif feats == "affixes":
        words = nltk.tokenize.wordpunct_tokenize(text)
        ngrams = [''.join(ngram) for ngram in nltk.ngrams(text, n)]
        # relative frequencies should be computed from all existing n-grams
        total = len(ngrams)
        # and now get all types from Sapkota et al.
        affs = [w[:3] for w in words if len(w) > n] + [w[-3:] for w in words if len(w) > n]
        # space affixes (and punct affixes if keep_punct has been enabled)
        space_affs_and_punct = [re.sub(r'\p{Z}', '_', ngram)
                                for ngram in ngrams
                                if re.search(r'(^\p{Z})|(\p{Z}$)|(\p{P})', ngram)
                                ]
        tokens = affs + space_affs_and_punct

    #POS in english with NLTK - need to propose spacy later on
    elif feats == "pos":
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except:
            nltk.download('averaged_perceptron_tagger_eng')
        words = nltk.tokenize.wordpunct_tokenize(text)
        pos_tags = [pos for word, pos in nltk.pos_tag(words)]
        if n > 1:
            tokens = ["_".join(t) for t in list(nltk.ngrams(pos_tags, n))]
        else:
            tokens = pos_tags
        total = len(tokens)

    elif feats == "met_line":
        tokens = text.split()
        if n > 1:
            tokens = ["_".join(t) for t in list(nltk.ngrams(tokens, n))]
        total = len(tokens)

    # Adding sentence length ; still commented as it is a work in progress, an integer won't do, a quantile would be better
    #elif feats == "sentenceLength":
    #    sentences = nltk.tokenize.sent_tokenize(text)
    #    tokens = [str(len(nltk.tokenize.word_tokenize(sentence))) for sentence in sentences]

    #Adding an error message in case some distracted guy like me would enter something wrong:
    else:
        raise ValueError("Unsupported feature type. Choose from 'words', 'chars', 'affixes', 'met_line', 'met_syll' or 'pos'.")

    counts = Counter()
    counts.update(tokens)

    return counts, total

def relative_frequencies(wordCounts, total):
    """
    For a counter of word counts, return the relative frequencies
    :param wordCounts: a dictionary of word counts
    :param total, the total number of features
    :return a counter of word relative frequencies
    """
    for t in wordCounts.keys():
        wordCounts[t] = wordCounts[t] / total
    return wordCounts

def bin_frequencies(wordCounts):
    """
    For a counter of word counts, return the binarised frequencies
    :param wordCounts: a dictionary of word counts
    :return a counter of word relative frequencies
    """
    for t in wordCounts.keys():
        if wordCounts[t] > 0:
            wordCounts[t] = 1

    return wordCounts


def get_feature_list(myTexts, feats="words", n=1, freqsType="relative"):
    """
    :param myTexts: a 'myTexts' object, containing documents to be processed
    :param feat_list: a list of features to be selected
    :param feats: type of feats (words, chars, affixes, POS, met_line, or met_syll)
    :param freqsType: "relative", "absolute" or "binary" frequencies
    :param n: n-grams length
    :return: list of features, with total frequency
    """
    my_feats = Counter()
    total = 0

    for text in myTexts:
        counts, text_total = count_features(text["text"], feats=feats, n=n)

        my_feats.update(counts)
        total = total + text_total

    if freqsType == "relative":
        my_feats = relative_frequencies(my_feats, total)
    elif freqsType == "binary":
        my_feats = bin_frequencies(my_feats)

    # sort them
    my_feats = [(i, my_feats[i]) for i in sorted(my_feats, key=my_feats.get, reverse=True)]

    return my_feats

def get_doc_frequency(myTexts):
    """
    Compute the document frequency of features (i.e., in how many documents it appears).
    """
    feats_doc_freq = Counter()
    feats_doc_freq.update([k for text in myTexts for k in text["wordCounts"].keys()])

    return feats_doc_freq




def get_counts(myTexts, feat_list=None, feats = "words", n = 1, freqsType = "relative"):
    """
    Get counts for a collection of texts
    :param myTexts: the document collection
    :param feat_list: a list of features to be selected (None for all)
    :param feats: the type of feats (words, chars, affixes, POS)
    :param n: the length of n-grams
    :param freqsType: relative, absolute or binarised freqs
    :return: the collection with, for each text, a 'wordCounts' dictionary
    """

    if freqsType not in ["relative", "absolute", "binary"]:
        raise ValueError("Unsupported frequency type. Choose from 'relative', 'absolute', or 'binary'.")

    for i in enumerate(myTexts):

        counts, total = count_features(myTexts[i[0]]["text"], feats=feats, n=n)

        if freqsType == "relative":
            counts = relative_frequencies(counts, total)

        elif freqsType == "binary":
            counts = bin_frequencies(counts)

        if feat_list:
            # and keep only the ones in the feature list
            counts = {f: counts[f] for f in feat_list if f in counts.keys()}

        myTexts[i[0]]["wordCounts"] = counts

    return myTexts


