import superstyl.preproc.pipe as pipe
import superstyl.preproc.features_extract as fex
from superstyl.preproc.text_count import count_process
import superstyl.preproc.embedding as embed
import tqdm
import pandas
from typing import Optional, List, Tuple, Union

from superstyl.config import Config, FeatureConfig, NormalizationConfig


def _load_single_feature(
    myTexts: List[dict],
    feat_config: FeatureConfig,
    norm_config: NormalizationConfig,
    use_provided_feat_list: bool = False,
) -> Tuple[pandas.DataFrame, List]:
    """
    Extract features for a single FeatureConfig.
    Internal function used by load_corpus.
    
    Args:
        use_provided_feat_list: If True and feat_config.feat_list is provided,
            return that list instead of the computed one. Used for test sets
            to ensure same features as training set.
    """
    feats = feat_config.type
    n = feat_config.n
    k = feat_config.k
    freqsType = feat_config.freq_type
    provided_feat_list = feat_config.feat_list
    embedding = feat_config.embedding
    neighbouring_size = feat_config.neighbouring_size
    culling = feat_config.culling

    embeddedFreqs = False
    if embedding:
        print(".......loading embedding.......")
        model = embed.load_embeddings(embedding)
        embeddedFreqs = True
        freqsType = "absolute"

    print(f".......getting features ({feats}, n={n}).......")

    if provided_feat_list is None:
        feat_list = fex.get_feature_list(myTexts, feats=feats, n=n, freqsType=freqsType)
        if k > len(feat_list):
            print(f"K limit ignored ({len(feat_list)} < {k})")
        else:
            val = feat_list[k-1][1]
            feat_list = [m for m in feat_list if m[1] >= val]
    else:
        feat_list = provided_feat_list

    print(".......getting counts.......")

    my_feats = [m[0] for m in feat_list]
    # Copy myTexts to avoid mutating original for multi-feature
    texts_copy = [dict(t) for t in myTexts]
    texts_copy = fex.get_counts(texts_copy, feat_list=my_feats, feats=feats, n=n, freqsType=freqsType)

    if embedding:
        print(".......embedding counts.......")
        texts_copy, my_feats = embed.get_embedded_counts(texts_copy, my_feats, model, topn=neighbouring_size)
        feat_list = [f for f in feat_list if f[0] in my_feats]

    if culling > 0:
        print(f".......Culling at {culling}%.......")
        feats_doc_freq = fex.get_doc_frequency(texts_copy)
        my_feats = [f for f in my_feats if (feats_doc_freq[f] / len(texts_copy) * 100) > culling]
        feat_list = [f for f in feat_list if f[0] in my_feats]

    print(".......feeding data frame.......")

    loc = {}
    for t in tqdm.tqdm(texts_copy):
        text, local_freqs = count_process((t, my_feats), embeddedFreqs=embeddedFreqs)
        loc[text["name"]] = local_freqs

    feats_df = pandas.DataFrame.from_dict(loc, columns=list(my_feats), orient="index")

    # For test sets: return the provided feat_list unchanged
    if use_provided_feat_list and provided_feat_list is not None:
        return feats_df, provided_feat_list
    
    return feats_df, feat_list


def load_corpus(
    config: Optional[Config] = None,
    use_provided_feat_list: bool = False,
    **kwargs
) -> Tuple[pandas.DataFrame, Union[List, List[List]]]:
    """
    Load a corpus and extract features.
    
    Can be called with:
    1. A Config object: load_corpus(config=my_config)
    2. Individual parameters (backward compatible): 
       load_corpus(data_paths=paths, feats="chars", n=3)
    
    Args:
        config: Configuration object. If None, built from kwargs.
        use_provided_feat_list: If True and feat_list provided, return it unchanged.
                               Use for test sets to match training features.
        **kwargs: Individual parameters for backward compatibility.
                  Supported: data_paths, feat_list, feats, n, k, freqsType,
                  format, sampling, units, size, step, max_samples, samples_random,
                  keep_punct, keep_sym, no_ascii, identify_lang, embedding,
                  neighbouring_size, culling
    
    Returns:
        - If single feature: (DataFrame, feat_list)
        - If multiple features: (DataFrame with prefixed columns, list of feat_lists)
    """
    # Build config from kwargs if not provided
    if config is None:
        config = Config.from_kwargs(**kwargs)
    
    # Validate configuration
    config.validate()
    data_paths = config.corpus.paths
        
    # Handle string paths (single file or glob pattern)
    if isinstance(data_paths, str):
        import glob
        # If it's a glob pattern, expand it
        if '*' in data_paths or '?' in data_paths:
            data_paths = sorted(glob.glob(data_paths))
        else:
            # Single file path - wrap in list
            data_paths = [data_paths]

    # Validate
    for feat_config in config.features:
        if feat_config.type in ('lemma', 'pos', 'met_line', 'met_syll') and config.corpus.format != 'tei':
            raise ValueError(f"{feat_config.type} requires TEI format.")
        if feat_config.type in ('met_line', 'met_syll') and config.sampling.units != 'verses':
            raise ValueError(f"{feat_config.type} verses lines units.")
    data_paths = config.corpus.paths
        
    # Handle string paths (single file or glob pattern)
    if isinstance(data_paths, str):
        import glob
        # If it's a glob pattern, expand it
        if '*' in data_paths or '?' in data_paths:
            data_paths = sorted(glob.glob(data_paths))
        else:
            # Single file path - wrap in list
            data_paths = [data_paths]

    # Validate
    for feat_config in config.features:
        if feat_config.type in ('lemma', 'pos', 'met_line', 'met_syll') and config.corpus.format != 'tei':
            raise ValueError(f"{feat_config.type} requires TEI format.")
        if feat_config.type in ('met_line', 'met_syll') and config.sampling.units != 'verses':
            raise ValueError(f"{feat_config.type} requires verses units.")

    # Load texts once
    print(".......loading texts.......")

    if config.sampling.enabled:
        myTexts = pipe.docs_to_samples(
            data_paths,
            config=config
        )
    else:
        myTexts = pipe.load_texts(
            data_paths,
            config=config
        )

    unique_texts = [text["name"] for text in myTexts]
    
    # Build metadata
    metadata = pandas.DataFrame(
        columns=['author', 'lang'],
        index=unique_texts,
        data=[[t["aut"], t["lang"]] for t in myTexts]
    )

    # Single feature case
    if len(config.features) == 1:
        feat_config = config.features[0]
        feats_df, feat_list = _load_single_feature(
            myTexts, feat_config, config.normalization, use_provided_feat_list
        )
        corpus = pandas.concat([metadata, feats_df], axis=1)
        return corpus, feat_list

    # Multiple features case
    print(f".......extracting {len(config.features)} feature sets.......")
    
    all_feat_lists = []
    merged_feats = metadata.copy()

    for i, feat_config in enumerate(config.features):
        prefix = feat_config.name or f"f{i+1}"
        print(f".......processing {prefix}.......")
        
        feats_df, feat_list = _load_single_feature(
            myTexts, feat_config, config.normalization, use_provided_feat_list
        )
        
        # Prefix columns to avoid collisions
        feats_df = feats_df.rename(columns={col: f"{prefix}_{col}" for col in feats_df.columns})
        
        merged_feats = pandas.concat([merged_feats, feats_df], axis=1)
        all_feat_lists.append(feat_list)

    return merged_feats, all_feat_lists