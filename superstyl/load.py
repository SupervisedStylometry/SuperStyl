import superstyl.preproc.pipe as pipe
import superstyl.preproc.features_extract as fex
from superstyl.preproc.text_count import count_process
import superstyl.preproc.embedding as embed
import json
import tqdm
import pandas

def load_corpus(data_paths, feat_list=None, feats="words", n=1, k=5000, relFreqs=True, format="txt", sampling=False,
                units="words", size=3000, step=None, max_samples=None, keep_punct=False, keep_sym=False,
                identify_lang=False, embedding=False, neighbouring_size=10):
    """
    Main function to load a corpus from a collection of file, and an optional list of features to extract.
    :param #TODO, document all params
    :return a pandas dataFrame of text metadata and feature frequencies; a global list of features with their frequencies
    """

    embeddedFreqs = False
    if embedding:
        print(".......loading embedding.......")
        relFreqs = False  # we need absolute freqs as a basis for embedded frequencies
        model = embed.load_embeddings(embedding)
        embeddedFreqs = True

    print(".......loading texts.......")

    if sampling:
        myTexts = pipe.docs_to_samples(data_paths, format=format, units=units, size=size, step=step,
                                       max_samples=max_samples, keep_punct=keep_punct, keep_sym=keep_sym,
                                       identify_lang = identify_lang
                                       )

    else:
        myTexts = pipe.load_texts(data_paths, format=format, max_samples=max_samples, keep_punct=keep_punct,
                                  keep_sym=keep_sym, identify_lang=identify_lang)

    print(".......getting features.......")

    if not feat_list:
        feat_list = fex.get_feature_list(myTexts, feats=feats, n=n, relFreqs=relFreqs)
        if k > len(feat_list):
            print("K Limit ignored because the size of the list is lower ({} < {})".format(len(feat_list), k))
        else:
            # and now, cut at around rank k
            val = feat_list[k][1]
            feat_list = [m for m in feat_list if m[1] >= val]


    print(".......getting counts.......")

    my_feats = [m[0] for m in feat_list] # keeping only the features without the frequencies
    myTexts = fex.get_counts(myTexts, feat_list=my_feats, feats=feats, n=n, relFreqs=relFreqs)

    if embedding:
        print(".......embedding counts.......")
        myTexts = embed.get_embedded_counts(myTexts, my_feats, model, topn=neighbouring_size)

    unique_texts = [text["name"] for text in myTexts]

    print(".......feeding data frame.......")

    loc = {}

    for t in tqdm.tqdm(myTexts):
        text, local_freqs = count_process((t, feat_list), embeddedFreqs=embeddedFreqs)
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

    feats = pandas.DataFrame.from_dict(loc, columns=list(feat_list), orient="index")

    # Free some more
    del loc

    corpus = pandas.concat([metadata, feats], axis=1)

    return corpus, feat_list