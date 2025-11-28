from superstyl.load import load_corpus, load_corpus_with_config
from superstyl.svm import train_svm, train_svm_with_config, plot_rolling, plot_coefficients
from superstyl.config import (
    Config,
    CorpusConfig,
    FeatureConfig,
    SamplingConfig,
    NormalizationConfig,
    SVMConfig,
    get_config,
    set_config,
    reset_config
)

__all__ = [
    # Main functions
    'load_corpus',
    'load_corpus_with_config',
    'train_svm',
    'train_svm_with_config',
    'plot_rolling',
    'plot_coefficients',
    
    # Configuration classes
    'Config',
    'CorpusConfig',
    'FeatureConfig',
    'SamplingConfig',
    'NormalizationConfig',
    'SVMConfig',
    
    # Configuration management
    'get_config',
    'set_config',
    'reset_config',
]