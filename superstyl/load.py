import superstyl.preproc.pipe as pipe
import superstyl.preproc.features_extract as fex
from superstyl.preproc.text_count import count_process
import superstyl.preproc.embedding as embed
import tqdm
import pandas


def load_corpus(data_paths, feat_list=None, feats="words", n=1, k=5000, freqsType="relative", format="txt", sampling=False,
                units="words", size=3000, step=None, max_samples=None, samples_random=False, keep_punct=False, keep_sym=False,
                no_ascii=False,
                identify_lang=False, embedding=False, neighbouring_size=10, culling=0):
    """
    Main function to load a corpus from a collection of file, and an optional list of features to extract.
    :param data_paths: paths to the source files
    :param feat_list: an optional list of features (as created by load_corpus), default None
    :param feats: the type of features, one of 'words', 'chars', 'affixes, and 'POS'. Affixes are inspired by
    Sapkota et al. 2015, and include space_prefix, space_suffix, prefix, suffix, and, if keep_pos, punctuation n-grams.
    POS are currently only implemented for Modern English
    :param n: n grams lengths (default 1)
    :param k: How many most frequent? The function takes the rank of k (if k is smaller than the total number of features),
    gets its frequencies, and only include features of superior or equal total frequencies.
    :param freqsType: return relative, absolute or binarised frequencies (default: relative)
    :param format: one of txt, xml or tei. /!\ only txt is fully implemented.
    :param sampling: whether to sample the texts, by cutting it into slices of a given length, until the last possible
      slice of this length, which means that often the end of the text will be eliminated (default False)
    :param units: units of length for sampling, one of 'words', 'verses' (default: words). 'verses' is only implemented
    for the 'tei' format
    :param size: the size of the samples (in units)
    :param step: step for sampling with overlap (default is step = size, which means no overlap).
    Reduce for overlapping slices
    :param max_samples: Maximum number of (randomly selected) samples per author/class (default is all)
    :param samples_random: Should random sampling with replacement be performed instead of continuous sampling (default: false)
    :param keep_punct: whether to keep punctuation and caps (default is False)
    :param keep_sym: same as keep_punct, and numbers are kept as well (default is False). /!\ does not
    actually keep symbols
    :param no_ascii: disables conversion to ASCII (default is conversion)
    :param identify_lang: if true, the language of each text will be guessed, using langdetect (default is False)
    :param embedding: optional path to a word2vec embedding in txt format to compute frequencies among a set of
    semantic neighbourgs (i.e., pseudo-paronyms)
    :param neighbouring_size: size of semantic neighbouring in the embedding (as per gensim most_similar,
    with topn=neighbouring_size)
    :param culling percentage value for culling, meaning in what percentage of samples should a feature be present to be retained (default is 0, meaning no culling)
    :return a pandas dataFrame of text metadata and feature frequencies; a global list of features with their frequencies
    """

    embeddedFreqs = False
    if embedding:
        print(".......loading embedding.......")
        relFreqs = False  # we need absolute freqs as a basis for embedded frequencies
        model = embed.load_embeddings(embedding)
        embeddedFreqs = True
        freqsType = "absolute" #absolute freqs are required for embedding

    print(".......loading texts.......")

    if sampling:
        myTexts = pipe.docs_to_samples(data_paths, feats=feats, format=format, units=units, size=size, step=step,
                                       max_samples=max_samples, samples_random=samples_random,
                                       keep_punct=keep_punct, keep_sym=keep_sym, no_ascii=no_ascii,
                                       identify_lang = identify_lang)

    else:
        myTexts = pipe.load_texts(data_paths, format=format, max_samples=max_samples, keep_punct=keep_punct,
                                  keep_sym=keep_sym, no_ascii=no_ascii, identify_lang=identify_lang)

    print(".......getting features.......")

    if feat_list is None:
        feat_list = fex.get_feature_list(myTexts, feats=feats, n=n, freqsType=freqsType)
        if k > len(feat_list):
            print("K Limit ignored because the size of the list is lower ({} < {})".format(len(feat_list), k))
        else:
            # and now, cut at around rank k
            val = feat_list[k-1][1]
            feat_list = [m for m in feat_list if m[1] >= val]


    print(".......getting counts.......")

    my_feats = [m[0] for m in feat_list] # keeping only the features without the frequencies
    myTexts = fex.get_counts(myTexts, feat_list=my_feats, feats=feats, n=n, freqsType=freqsType)

    if embedding:
        print(".......embedding counts.......")
        myTexts, my_feats = embed.get_embedded_counts(myTexts, my_feats, model, topn=neighbouring_size)
        feat_list = [f for f in feat_list if f[0] in my_feats]

    unique_texts = [text["name"] for text in myTexts]

    if culling > 0:
        print(".......Culling at " + str(culling) + "%.......")
        # Counting in how many sample the feat appear
        feats_doc_freq = fex.get_doc_frequency(myTexts)
        # Now selecting
        my_feats = [f for f in my_feats if (feats_doc_freq[f] / len(myTexts) * 100) > culling]
        feat_list = [f for f in feat_list if f[0] in my_feats]

    print(".......feeding data frame.......")

    loc = {}

    for t in tqdm.tqdm(myTexts):
        text, local_freqs = count_process((t, my_feats), embeddedFreqs=embeddedFreqs)
        loc[text["name"]] = local_freqs

    # Saving metadata for later
    metadata = pandas.DataFrame(columns=['author', 'lang'], index=unique_texts, data=
    [[t["aut"], t["lang"]] for t in myTexts])

    # Free some space before doing this...
    del myTexts

    # frequence based selection
    # WOW, pandas is a great tool, almost as good as using R
    # But confusing as well: boolean selection works on rows by default
    # were elsewhere it works on columns
    # take only rows where the number of values above 0 is superior to two
    # (i.e. appears in at least two texts)
    #feats = feats.loc[:, feats[feats > 0].count() > 2]

    feats = pandas.DataFrame.from_dict(loc, columns=list(my_feats), orient="index")

    # Free some more
    del loc

    corpus = pandas.concat([metadata, feats], axis=1)

    return corpus, feat_list