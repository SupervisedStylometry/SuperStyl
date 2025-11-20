import json
import pandas as pd
import glob

from superstyl.load import load_corpus

def load_corpus_from_config(config_path, is_test=False):
    """
    Load a corpus based on a JSON configuration file.
    
    Parameters:
    -----------
    config_path : str
        Path to the JSON configuration file
        
    Returns:
    --------
    tuple: (corpus, feat_list) - Same format as load_corpus function
           If multiple features are defined, returns the merged corpus and the combined feature list
           If only one feature is defined, returns that corpus and its feature list
    """
    # Load configuration
    if not config_path.endswith('.json'):
        raise ValueError(f"Unsupported configuration file format: {config_path}. Only JSON format is supported.")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get corpus paths

    if 'paths' in config:
        if isinstance(config['paths'], list):
            paths = []
            for path in config['paths']:
                if '*' in path or '?' in path or '[' in path:
                    expanded_paths = glob.glob(path)
                    if not expanded_paths:
                        print(f"Warning: No files found for pattern '{path}'")
                    paths.extend(expanded_paths)
                else:
                    paths.append(path)
        elif isinstance(config['paths'], str):
            if '*' in config['paths'] or '?' in config['paths'] or '[' in config['paths']:
                paths = glob.glob(config['paths'])
                if not paths:
                    raise ValueError(f"No files found for glob pattern '{config['paths']}'")
            else:
                paths = [config['paths']]
        else:
            raise ValueError("Paths in config must be either a list or a glob pattern string")
    else:
        raise ValueError("No paths provided and no paths found in config")
    
    # Get sampling parameters
    sampling_params = config.get('sampling', {})

    # Use the first feature to create the base corpus with sampling
    feature_configs = config.get('features', [])
    if not feature_configs:
        raise ValueError("No features specified in the configuration")
    
    # If there's only one feature, we can simply return the result of load_corpus
    if len(feature_configs) == 1:
        feature_config = feature_configs[0]
        feature_name = feature_config.get('name', "f1")
        
        # Check for feature list file
        feat_list = None
        feat_list_path = feature_config.get('feat_list')
        if feat_list_path:
            if feat_list_path.endswith('.json'):
                with open(feat_list_path, 'r') as f:
                    feat_list = json.load(f)
            elif feat_list_path.endswith('.txt'):
                with open(feat_list_path, 'r') as f:
                    feat_list = [[feat.strip(), 0] for feat in f.readlines()]
        
        # Set up other parameters
        params = {
            'feats': feature_config.get('type', 'words'),
            'n': feature_config.get('n', 1),
            'k': feature_config.get('k', 5000),
            'freqsType': feature_config.get('freq_type', 'relative'),
            'format': config.get('format', 'txt'),
            'sampling': sampling_params.get('enabled', False),
            'units': sampling_params.get('units', 'words'),
            'size': sampling_params.get('sample_size', 3000),
            'step': sampling_params.get('step', None),
            'max_samples': sampling_params.get('max_samples', None),
            'samples_random': sampling_params.get('samples_random', False),
            'keep_punct': feature_config.get('keep_punct', False),
            'keep_sym': feature_config.get('keep_sym', False),
            'no_ascii': feature_config.get('no_ascii', False),
            'identify_lang': feature_config.get('identify_lang', False),
            'embedding': feature_config.get('embedding', None),
            'neighbouring_size': feature_config.get('neighbouring_size', 10),
            'culling': feature_config.get('culling', 0)
        }

        print(f"Loading corpus with {feature_name}...")
        corpus, features = load_corpus(paths, feat_list=feat_list, **params)
        
        return corpus, features
    
    # For multiple features, we need to process each one and merge the results
    corpora = {}
    feature_lists = {}
    
    # Process each feature configuration
    for i, feature_config in enumerate(feature_configs):
        feature_name = feature_config.get('name', f"f{i+1}")

        # Check for feature list file
        feat_list = None
        feat_list_path = feature_config.get('feat_list')
        print(feat_list_path)
        if feat_list_path:
            if feat_list_path.endswith('.json'):
                with open(feat_list_path, 'r') as f:
                    feat_list = json.load(f)
            elif feat_list_path.endswith('.txt'):
                with open(feat_list_path, 'r') as f:
                    feat_list = [[feat.strip(), 0] for feat in f.readlines()]
        
        # Set up other parameters
        params = {
            'feats': feature_config.get('type', 'words'),
            'n': feature_config.get('n', 1),
            'k': feature_config.get('k', 5000),
            'freqsType': feature_config.get('freq_type', 'relative'),
            'format': config.get('format', 'txt'),
            'sampling': sampling_params.get('enabled', False),
            'units': sampling_params.get('units', 'words'),
            'size': sampling_params.get('sample_size', 3000),
            'step': sampling_params.get('step', None),
            'max_samples': sampling_params.get('max_samples', None),
            'samples_random': sampling_params.get('samples_random', False),
            'keep_punct': config.get('keep_punct', False),
            'keep_sym': config.get('keep_sym', False),
            'no_ascii': config.get('no_ascii', False),
            'identify_lang': config.get('identify_lang', False),
            'embedding': feature_config.get('embedding', None),
            'neighbouring_size': feature_config.get('neighbouring_size', 10),
            'culling': feature_config.get('culling', 0)
        }
        
        print(f"Loading {feature_name}...")

        corpus, features = load_corpus(paths, feat_list=feat_list, **params)
        
        # Store corpus and features
        corpora[feature_name] = corpus

        if feat_list is not None and is_test:
            feature_lists[feature_name] = feat_list
        else:
            feature_lists[feature_name] = features
        
    
    # Create a merged dataset
    print("Creating merged dataset...")
    first_corpus_name = next(iter(corpora))
    
    # Start with metadata from the first corpus
    metadata = corpora[first_corpus_name][['author', 'lang']]
    
    # Create an empty DataFrame for the merged corpus
    merged = pd.DataFrame(index=metadata.index)
    
    # Add metadata
    merged = pd.concat([metadata, merged], axis=1)
    
    # Combine all features with prefixes to avoid name collisions
    all_features = []
    
    # Add features from each corpus
    for name, corpus in corpora.items():
        single_feature = []

        feature_cols = [col for col in corpus.columns if col not in ['author', 'lang']]
        
        # Rename columns to avoid duplicates
        renamed_cols = {col: col for col in feature_cols}
        feature_df = corpus[feature_cols].rename(columns=renamed_cols)
        
        # Merge with the main DataFrame
        merged = pd.concat([merged, feature_df], axis=1)
        
        # Add features to the combined list with prefixes
        for feature in feature_lists[name]:
            single_feature.append((feature[0], feature[1]))
    
        all_features.append(single_feature)
    # Return the merged corpus and combined feature list
    return merged, all_features

