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
) -> Tuple[pandas.DataFrame, List]:
    """
    Extract features for a single FeatureConfig.
    Internal function used by load_corpus.
    """
    feats = feat_config.type
    n = feat_config.n
    k = feat_config.k
    freqsType = feat_config.freq_type
    feat_list = feat_config.feat_list
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

    if feat_list is None:
        feat_list = fex.get_feature_list(myTexts, feats=feats, n=n, freqsType=freqsType)
        if k > len(feat_list):
            print(f"K limit ignored ({len(feat_list)} < {k})")
        else:
            val = feat_list[k-1][1]
            feat_list = [m for m in feat_list if m[1] >= val]

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

    return feats_df, feat_list


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
) -> Tuple[pandas.DataFrame, Union[List, List[List]]]:
    """
    Load a corpus and extract features.
    
    Can be called with individual parameters (single feature) or with a Config object
    (single or multiple features).
    
    Returns:
        - If single feature: (DataFrame, feat_list)
        - If multiple features via Config: (DataFrame with prefixed columns, list of feat_lists)
    """
    
    # Build Config from individual parameters if not provided
    if config is None:
        config = Config()
        config.corpus.paths = data_paths or []
        config.corpus.format = format
        config.corpus.identify_lang = identify_lang
        config.features = [FeatureConfig(
            type=feats,
            n=n,
            k=k,
            freq_type=freqsType,
            feat_list=feat_list,
            embedding=embedding if embedding else None,
            neighbouring_size=neighbouring_size,
            culling=culling,
        )]
        config.sampling.enabled = sampling
        config.sampling.units = units
        config.sampling.size = size
        config.sampling.step = step
        config.sampling.max_samples = max_samples
        config.sampling.random = samples_random
        config.normalization.keep_punct = keep_punct
        config.normalization.keep_sym = keep_sym
        config.normalization.no_ascii = no_ascii
    
    # Use paths from config if data_paths not provided directly
    if data_paths is None:
        data_paths = config.corpus.paths

    # Validate
    for feat_config in config.features:
        if feat_config.type in ('lemma', 'pos', 'met_line', 'met_syll') and config.corpus.format != 'tei':
            raise ValueError(f"{feat_config.type} requires TEI format.")
        if feat_config.type in ('met_line', 'met_syll') and config.sampling.units != 'lines':
            raise ValueError(f"{feat_config.type} requires lines units.")

    # Load texts once
    print(".......loading texts.......")
    
    norm_params = {
        "keep_punct": config.normalization.keep_punct,
        "keep_sym": config.normalization.keep_sym,
        "no_ascii": config.normalization.no_ascii,
    }

    if config.sampling.enabled:
        myTexts = pipe.docs_to_samples(
            data_paths,
            feats=config.features[0].type,  # Use first feature type for loading
            format=config.corpus.format,
            units=config.sampling.units,
            size=config.sampling.size,
            step=config.sampling.step,
            max_samples=config.sampling.max_samples,
            samples_random=config.sampling.random,
            identify_lang=config.corpus.identify_lang,
            **norm_params
        )
    else:
        myTexts = pipe.load_texts(
            data_paths,
            feats=config.features[0].type,
            format=config.corpus.format,
            max_samples=config.sampling.max_samples,
            identify_lang=config.corpus.identify_lang,
            **norm_params
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
        feats_df, feat_list = _load_single_feature(myTexts, feat_config, config.normalization)
        corpus = pandas.concat([metadata, feats_df], axis=1)
        return corpus, feat_list

    # Multiple features case - extract and merge
    print(f".......extracting {len(config.features)} feature sets.......")
    
    all_feat_lists = []
    merged_feats = metadata.copy()

    for i, feat_config in enumerate(config.features):
        prefix = feat_config.name or f"f{i+1}"
        print(f".......processing {prefix}.......")
        
        feats_df, feat_list = _load_single_feature(myTexts, feat_config, config.normalization)
        
        # Prefix columns to avoid collisions
        feats_df = feats_df.rename(columns={col: f"{prefix}_{col}" for col in feats_df.columns})
        
        merged_feats = pandas.concat([merged_feats, feats_df], axis=1)
        all_feat_lists.append(feat_list)

    return merged_feats, all_feat_lists