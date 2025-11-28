import warnings
from superstyl.config import Config
from superstyl.load import load_corpus


def load_corpus_from_config(config_path: str):
    """
    Load corpus from JSON config file.
    
    DEPRECATED: Use instead:
        config = Config.from_json(config_path)
        corpus, features = load_corpus(config=config)
    """
    warnings.warn(
        "load_corpus_from_config is deprecated. "
        "Use Config.from_json() + load_corpus(config=config) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if not config_path.endswith('.json'):
        raise ValueError(f"Only JSON config files are supported: {config_path}")
    
    config = Config.from_json(config_path)
    return load_corpus(config=config)