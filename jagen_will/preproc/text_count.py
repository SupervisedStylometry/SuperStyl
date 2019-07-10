import nltk.tokenize

def count_process(args):
    text, feat_list = args
    local_freqs = list([0] * len(feat_list))
    for word, value in text["wordCounts"].items():
        if word in feat_list:
            local_freqs[feat_list.index(word)] = value
    return text, local_freqs


def encode_texts(text, feat_map, n, feats="chars"):

    if feats == "words":
        tokens = nltk.tokenize.wordpunct_tokenize(text["text"])

        if n > 1:
            tokens = ["_".join(t) for t in list(nltk.ngrams(tokens, n))]

    if feats == "chars":
        tokens = list(text["text"].replace(' ', '_'))
        if n > 1:
            tokens = ["".join(t) for t in list(nltk.ngrams(tokens, n))]

    return text, list([feat_map[ngram] for ngram in tokens])
