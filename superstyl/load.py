import superstyl.preproc.pipe as pipe
import superstyl.preproc.features_extract as fex
from superstyl.preproc.text_count import count_process
import superstyl.preproc.embedding as embed
import tqdm
import pandas

from typing import Optional, List, Tuple, Union
from superstyl.config import Config


def load_corpus(
    data_paths: Union[List[str], None] = None,
    feat_list: Optional[List] = None,
    feats: str = "words",
    n: int = 1,
    k: int = 5000,
    freqsType: str = "relative",
    format: str = "txt",
    sampling: bool = False,
    units: str = "words",
    size: int = 3000,
    step: Optional[int] = None,
    max_samples: Optional[int] = None,
    samples_random: bool = False,
    keep_punct: bool = False,
    keep_sym: bool = False,
    no_ascii: bool = False,
    identify_lang: bool = False,
    embedding: Union[str, bool] = False,
    neighbouring_size: int = 10,
    culling: float = 0,
    config: Optional[Config] = None
) -> Tuple[pandas.DataFrame, List]:
    """
    Main function to load a corpus from a collection of files.
    
    Can be called either with individual parameters (backward compatible)
    or with a Config object.
    
    :param data_paths: paths to the source files
    :param feat_list: an optional list of features (as created by load_corpus), default None
    :param feats: the type of features, one of 'words', 'chars', 'affixes, 'lemma', 'pos', 'met_line' and 'met_syll'.
    Affixes are inspired by Sapkota et al. 2015, and include space_prefix, space_suffix, prefix, suffix, and,
    if keep_punct, punctuation n-grams. From TEI, pos, lemma, met_line or met_syll can
    be extracted; met_line is the prosodic (stress) annotation of a full verse; met_syll is a char n-gram of prosodic
    annotation
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

    if feats in ('lemma', 'pos', 'met_line', 'met_syll') and format != 'tei':
        raise ValueError(
            f"{feats} features are only possible with adequate TEI format (@lemma, @pos, @met)"
        )

    if feats in ('met_line', 'met_syll') and units != 'lines':
        raise ValueError(
            f"{feats} features are only possible with TEI format that includes lines and @met"
        )

    embeddedFreqs = False
    if embedding:
        print(".......loading embedding.......")
        model = embed.load_embeddings(embedding)
        embeddedFreqs = True
        freqsType = "absolute"  # absolute freqs are required for embedding

    print(".......loading texts.......")

    # Create normalization config for pipe functions
    norm_params = {
        "keep_punct": keep_punct,
        "keep_sym": keep_sym,
        "no_ascii": no_ascii
    }

    if sampling:
        myTexts = pipe.docs_to_samples(
            data_paths,
            feats=feats,
            format=format,
            units=units,
            size=size,
            step=step,
            max_samples=max_samples,
            samples_random=samples_random,
            identify_lang=identify_lang,
            **norm_params
        )
    else:
        myTexts = pipe.load_texts(
            data_paths,
            feats=feats,
            format=format,
            max_samples=max_samples,
            identify_lang=identify_lang,
            **norm_params
        )

    print(".......getting features.......")

    if feat_list is None:
        feat_list = fex.get_feature_list(myTexts, feats=feats, n=n, freqsType=freqsType)
        if k > len(feat_list):
            print(f"K Limit ignored because the size of the list is lower ({len(feat_list)} < {k})")
        else:
            # Cut at around rank k
            val = feat_list[k-1][1]
            feat_list = [m for m in feat_list if m[1] >= val]

    print(".......getting counts.......")

    my_feats = [m[0] for m in feat_list]  # keeping only the features without the frequencies
    myTexts = fex.get_counts(myTexts, feat_list=my_feats, feats=feats, n=n, freqsType=freqsType)
    
    if embedding:
        print(".......embedding counts.......")
        myTexts, my_feats = embed.get_embedded_counts(myTexts, my_feats, model, topn=neighbouring_size)
        feat_list = [f for f in feat_list if f[0] in my_feats]

    unique_texts = [text["name"] for text in myTexts]

    if culling > 0:
        print(f".......Culling at {culling}%.......")
        # Counting in how many samples the feat appears
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
    metadata = pandas.DataFrame(
        columns=['author', 'lang'],
        index=unique_texts,
        data=[[t["aut"], t["lang"]] for t in myTexts]
    )

    # Free some space before doing this...
    del myTexts

    feats_df = pandas.DataFrame.from_dict(loc, columns=list(my_feats), orient="index")

    # Free some more
    del loc

    corpus = pandas.concat([metadata, feats_df], axis=1)

    return corpus, feat_list


def load_corpus_with_config(config: Config, feat_list: Optional[List] = None) -> Tuple[pandas.DataFrame, List]:
    """
    Load a corpus using a Config object.
    
    This is a convenience wrapper around load_corpus that uses the Config directly.
    
    :param config: Configuration object
    :param feat_list: Optional pre-existing feature list
    :return: a pandas dataFrame and a global list of features
    """
    return load_corpus(config=config, feat_list=feat_list)