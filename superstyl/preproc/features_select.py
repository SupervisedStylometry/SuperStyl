import regex as re

def filter_ngrams(feat_list, affixes=True, punct=True):
    """
    Filter a list of features in input to yield a selection of n-grams, according to the parameters,
    following Sapktota et al.
    feat_list: the feature list (typically, coming of main.py and loaded)
     affixes: affixes (n-grams beginning or ending by space)
     punct: n-grams containing punctuation
    """

    out = []

    if affixes:
        out = out + [f for f in feat_list if f[0].startswith('_') or f[0].endswith('_')]
        switch = True
        seen = set([f[0] for f in out])

    if punct:
        # a bit trickier: need to remove underscore not to include n-grams with just
        # underscore as punctuation
        if switch:
            out = out + [f for f in feat_list if re.match(r"\p{P}", re.sub('_', '', f[0]))
                         and f[0] not in seen]

        else:
            out = out + [f for f in feat_list if re.match(r"\p{P}", re.sub('_', '', f[0]))]

    return out







