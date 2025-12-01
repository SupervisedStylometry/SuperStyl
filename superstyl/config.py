from dataclasses import dataclass, field, fields
from typing import Optional, List, Any, Dict, Type, TypeVar
import json


T = TypeVar('T', bound='BaseConfig')


@dataclass
class BaseConfig:
    """
    Base configuration class providing common functionality.
    
    All configuration classes inherit from this to share:
    - to_dict() serialization
    - from_dict() deserialization  
    - Validation hooks
    """
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, BaseConfig):
                result[f.name] = value.to_dict()
            elif isinstance(value, list) and value and isinstance(value[0], BaseConfig):
                result[f.name] = [v.to_dict() for v in value]
            else:
                result[f.name] = value
        return result
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        return cls(**data)
    
    def validate(self) -> None:
        pass


@dataclass
class NormalizationConfig(BaseConfig):
    """
    Configuration for text normalization.
    """
    keep_punct: bool = False
    keep_sym: bool = False
    no_ascii: bool = False


@dataclass
class SamplingConfig(BaseConfig):
    """
    Configuration for text sampling.
    """
    enabled: bool = False
    units: str = "words"
    size: int = 3000
    step: Optional[int] = None
    max_samples: Optional[int] = None
    random: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if self.units not in ["words", "verses"]:
            raise ValueError(f"Invalid sampling units: {self.units}.")
        if self.random and self.step is not None:
            raise ValueError("Random sampling is not compatible with step.")
        if self.random and self.max_samples is None:
            raise ValueError("Random sampling needs max_samples.")


@dataclass
class FeatureConfig(BaseConfig):
    """
    Configuration for feature extraction.
    """
    name: Optional[str] = None  # For multi-feature identification
    type: str = "words"
    n: int = 1
    k: int = 5000
    freq_type: str = "relative"
    feat_list: Optional[List] = None
    feat_list_path: Optional[str] = None  # Path to load feat_list from
    embedding: Optional[str] = None
    neighbouring_size: int = 10
    culling: float = 0

    VALID_TYPES = ["words", "chars", "affixes", "lemma", "pos", "met_line", "met_syll"]
    VALID_FREQ_TYPES = ["relative", "absolute", "binary"]

    def __post_init__(self):
        self.validate()
        self._load_feat_list_if_needed()

    def validate(self) -> None:
        if self.type not in self.VALID_TYPES:
            raise ValueError(f"Invalid feature type: {self.type}.")
        if self.freq_type not in self.VALID_FREQ_TYPES:
            raise ValueError(f"Invalid frequency type: {self.freq_type}.")
        if self.n < 1:
            raise ValueError("n must be a positive integer.")

    def _load_feat_list_if_needed(self) -> None:
        """
        Load feature list from file if path is specified.
        """
        if self.feat_list_path and self.feat_list is None:
            with open(self.feat_list_path, 'r') as f:
                if self.feat_list_path.endswith('.json'):
                    self.feat_list = json.load(f)
                elif self.feat_list_path.endswith('.txt'):
                    self.feat_list = [[feat.strip(), 0] for feat in f.readlines()]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureConfig':
        # Filter out unknown keys for backward compatibility
        valid_keys = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


@dataclass
class CorpusConfig(BaseConfig):
    """
    Configuration for corpus loading.
    """
    paths: List[str] = field(default_factory=list)
    format: str = "txt"
    identify_lang: bool = False

    VALID_FORMATS = ["txt", "xml", "tei", "txm"]

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if self.format not in self.VALID_FORMATS:
            raise ValueError(f"Invalid format: {self.format}.")


@dataclass
class SVMConfig(BaseConfig):
    """
    Configuration for SVM training.
    """
    cross_validate: Optional[str] = None
    k: int = 0
    dim_reduc: Optional[str] = None
    norms: bool = True
    balance: Optional[str] = None
    class_weights: bool = False
    kernel: str = "LinearSVC"
    final_pred: bool = False
    get_coefs: bool = False
    plot_rolling: bool = False
    plot_smoothing: int = 3

    VALID_CV = [None, "leave-one-out", "k-fold", "group-k-fold"]
    VALID_DIM_REDUC = [None, "pca"]
    VALID_BALANCE = [None, "downsampling", "Tomek", "upsampling", "SMOTE", "SMOTETomek"]
    VALID_KERNELS = ["LinearSVC", "linear", "sigmoid", "rbf", "poly"]

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if self.cross_validate not in self.VALID_CV:
            raise ValueError(f"Invalid cross_validate: {self.cross_validate}.")
        if self.dim_reduc not in self.VALID_DIM_REDUC:
            raise ValueError(f"Invalid dim_reduc: {self.dim_reduc}.")
        if self.balance not in self.VALID_BALANCE:
            raise ValueError(f"Invalid balance: {self.balance}.")
        if self.kernel not in self.VALID_KERNELS:
            raise ValueError(f"Invalid kernel: {self.kernel}.")


@dataclass  
class Config(BaseConfig):
    """
    Main configuration class for SuperStyl.
    
    Aggregates all sub-configurations and provides factory methods
    to create configurations from various sources (CLI, JSON, dict).
    """
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    features: List[FeatureConfig] = field(default_factory=lambda: [FeatureConfig()])
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    svm: SVMConfig = field(default_factory=SVMConfig)
    output_prefix: Optional[str] = None

    # Mapping from flat kwargs to nested config structure
    # Format: 'kwarg_name': ('section', 'attr', optional_transform)
    KWARGS_MAPPING = {
        # Corpus
        'data_paths': ('corpus', 'paths', lambda x: x if isinstance(x, list) else [x]),
        'format': ('corpus', 'format', None),
        'identify_lang': ('corpus', 'identify_lang', None),
        
        # Features (single feature mode)
        'feats': ('features', 'type', None),
        'n': ('features', 'n', None),
        'k': ('features', 'k', None),
        'freqsType': ('features', 'freq_type', None),
        'feat_list': ('features', 'feat_list', None),
        'embedding': ('features', 'embedding', lambda x: x if x else None),
        'neighbouring_size': ('features', 'neighbouring_size', None),
        'culling': ('features', 'culling', None),
        
        # Sampling
        'sampling': ('sampling', 'enabled', None),
        'units': ('sampling', 'units', None),
        'size': ('sampling', 'size', None),
        'step': ('sampling', 'step', None),
        'max_samples': ('sampling', 'max_samples', None),
        'samples_random': ('sampling', 'random', None),
        
        # Normalization
        'keep_punct': ('normalization', 'keep_punct', None),
        'keep_sym': ('normalization', 'keep_sym', None),
        'no_ascii': ('normalization', 'no_ascii', None),
        
        # SVM
        'cross_validate': ('svm', 'cross_validate', None),
        'dim_reduc': ('svm', 'dim_reduc', None),
        'norms': ('svm', 'norms', None),
        'balance': ('svm', 'balance', None),
        'class_weights': ('svm', 'class_weights', None),
        'kernel': ('svm', 'kernel', None),
        'final_pred': ('svm', 'final_pred', None),
        'get_coefs': ('svm', 'get_coefs', None),
    }

    def validate(self) -> None:
        """
        Validate configuration consistency.
        """
        if not self.features:
            raise ValueError("No features specified for extraction.")
        
        if not self.corpus.paths:
            raise ValueError("No paths specified for corpus loading.")
            
        # Validate paths type
        if not isinstance(self.corpus.paths, list):
            raise TypeError("Paths in config must be either a list or a glob pattern string.")
        
        for feat_config in self.features:
            tei_only = ["lemma", "pos", "met_line", "met_syll"]
            if feat_config.type in tei_only and self.corpus.format != "tei":
                raise ValueError(f"{feat_config.type} requires TEI format.")
            if feat_config.type in ["met_line", "met_syll"] and self.sampling.units not in ["verses"]:
                raise ValueError(f"{feat_config.type} requires verses units.")

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_kwargs(cls, **kwargs) -> 'Config':
        """
        Build Config from flat kwargs (backward compatibility).
        
        This allows the old-style function calls to work:
            load_corpus(paths, feats="chars", n=3, keep_punct=True)
        """
        config_data = {
            'corpus': {},
            'features': {},
            'sampling': {},
            'normalization': {},
            'svm': {}
        }
        
        for kwarg_name, value in kwargs.items():
            if value is None:
                continue
                
            if kwarg_name in cls.KWARGS_MAPPING:
                section, attr, transform = cls.KWARGS_MAPPING[kwarg_name]
                final_value = transform(value) if transform else value
                config_data[section][attr] = final_value
        
        # Build the config with proper nesting
        return cls.from_dict(config_data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        kwargs = {}
        
        if 'corpus' in data:
            corpus_data = data['corpus'].copy()
            # Handle 'paths' key variations
            if 'paths' not in corpus_data and 'data_paths' in data:
                corpus_data['paths'] = data['data_paths']
            kwargs['corpus'] = CorpusConfig.from_dict(corpus_data) if corpus_data else CorpusConfig()
        
        if 'features' in data:
            features_data = data['features']
            if isinstance(features_data, dict):
                kwargs['features'] = [FeatureConfig.from_dict(features_data)]
            elif isinstance(features_data, list):
                if features_data:
                    kwargs['features'] = [FeatureConfig.from_dict(f) for f in features_data]
                else:
                    kwargs['features'] = []
        
        if 'sampling' in data and data['sampling']:
            sampling_data = data['sampling'].copy()
            # Handle alternative key names
            if 'sample_size' in sampling_data:
                sampling_data['size'] = sampling_data.pop('sample_size')
            if 'samples_random' in sampling_data:
                sampling_data['random'] = sampling_data.pop('samples_random')
            if 'sample_step' in sampling_data:
                sampling_data['step'] = sampling_data.pop('sample_step')
            if 'sample_units' in sampling_data:
                sampling_data['units'] = sampling_data.pop('sample_units')
            kwargs['sampling'] = SamplingConfig.from_dict(sampling_data)
        
        if 'normalization' in data and data['normalization']:
            kwargs['normalization'] = NormalizationConfig.from_dict(data['normalization'])
        
        if 'svm' in data and data['svm']:
            kwargs['svm'] = SVMConfig.from_dict(data['svm'])
        
        if 'output_prefix' in data:
            kwargs['output_prefix'] = data['output_prefix']
        
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Handle flat JSON format (paths at root level)
        if 'paths' in data and 'corpus' not in data:
            data['corpus'] = {'paths': data.pop('paths')}
            if 'format' in data:
                data['corpus']['format'] = data.pop('format')
            if 'identify_lang' in data:
                data['corpus']['identify_lang'] = data.pop('identify_lang')
        
        return cls.from_dict(data)


# Global configuration (optional singleton pattern)
_current_config: Optional[Config] = None

def get_config() -> Optional[Config]:
    return _current_config

def set_config(config: Config) -> None:
    global _current_config
    _current_config = config

def reset_config() -> None:
    global _current_config
    _current_config = None