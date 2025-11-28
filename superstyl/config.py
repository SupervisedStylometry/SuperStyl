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
    normalization: Optional[NormalizationConfig] = None

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
        data = data.copy()
        if "normalization" in data and data["normalization"] is not None:
            data["normalization"] = NormalizationConfig.from_dict(data["normalization"])
        return cls(**data)


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
    """Configuration for SVM training."""
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

    def validate(self) -> None:
        """
        Validate configuration consistency.
        """
        for feat_config in self.features:
            tei_only = ["lemma", "pos", "met_line", "met_syll"]
            if feat_config.type in tei_only and self.corpus.format != "tei":
                raise ValueError(f"{feat_config.type} requires TEI format.")
            if feat_config.type in ["met_line", "met_syll"] and self.sampling.units != "lines":
                raise ValueError(f"{feat_config.type} requires lines units.")

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        kwargs = {}
        
        if 'corpus' in data:
            kwargs['corpus'] = CorpusConfig.from_dict(data['corpus'])
        
        if 'features' in data:
            features_data = data['features']
            if isinstance(features_data, list):
                kwargs['features'] = [FeatureConfig.from_dict(f) for f in features_data]
            else:
                kwargs['features'] = [FeatureConfig.from_dict(features_data)]
        
        if 'sampling' in data:
            sampling_data = data['sampling'].copy()
            if 'sample_size' in sampling_data:
                sampling_data['size'] = sampling_data.pop('sample_size')
            if 'samples_random' in sampling_data:
                sampling_data['random'] = sampling_data.pop('samples_random')
            kwargs['sampling'] = SamplingConfig.from_dict(sampling_data)
        
        if 'normalization' in data:
            kwargs['normalization'] = NormalizationConfig.from_dict(data['normalization'])
        
        if 'svm' in data:
            kwargs['svm'] = SVMConfig.from_dict(data['svm'])
        
        if 'output_prefix' in data:
            kwargs['output_prefix'] = data['output_prefix']
        
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_cli_args(cls, args) -> 'Config':
        """Create Config from CLI args (single feature only)."""
        args_dict = vars(args)
        
        cli_mapping = {
            's': ('corpus', 'paths', lambda x: x if isinstance(x, list) else [x]),
            'x': ('corpus', 'format', None),
            'identify_lang': ('corpus', 'identify_lang', None),
            't': ('features', 'type', None),
            'n': ('features', 'n', None),
            'k': ('features', 'k', None),
            'freqs': ('features', 'freq_type', None),
            'embedding': ('features', 'embedding', lambda x: x if x else None),
            'neighbouring_size': ('features', 'neighbouring_size', None),
            'culling': ('features', 'culling', None),
            'sampling': ('sampling', 'enabled', None),
            'sample_units': ('sampling', 'units', None),
            'sample_size': ('sampling', 'size', None),
            'sample_step': ('sampling', 'step', None),
            'max_samples': ('sampling', 'max_samples', None),
            'samples_random': ('sampling', 'random', None),
            'keep_punct': ('normalization', 'keep_punct', None),
            'keep_sym': ('normalization', 'keep_sym', None),
            'no_ascii': ('normalization', 'no_ascii', None),
            'o': (None, 'output_prefix', None),
        }
        
        config_data = {'corpus': {}, 'features': {}, 'sampling': {}, 'normalization': {}}
        
        for cli_arg, (section, attr, transform) in cli_mapping.items():
            if cli_arg in args_dict and args_dict[cli_arg] is not None:
                value = transform(args_dict[cli_arg]) if transform else args_dict[cli_arg]
                if section is None:
                    config_data[attr] = value
                else:
                    config_data[section][attr] = value
        
        return cls.from_dict(config_data)

    @classmethod
    def from_svm_cli_args(cls, args) -> 'Config':
        """Create Config from SVM CLI args."""
        args_dict = vars(args)
        
        cli_mapping = {
            'cross_validate': ('svm', 'cross_validate', None),
            'k': ('svm', 'k', None),
            'dim_reduc': ('svm', 'dim_reduc', None),
            'norms': ('svm', 'norms', None),
            'balance': ('svm', 'balance', None),
            'class_weights': ('svm', 'class_weights', None),
            'kernel': ('svm', 'kernel', None),
            'final': ('svm', 'final_pred', None),
            'get_coefs': ('svm', 'get_coefs', None),
            'plot_rolling': ('svm', 'plot_rolling', None),
            'plot_smoothing': ('svm', 'plot_smoothing', None),
            'o': (None, 'output_prefix', None),
        }
        
        config_data = {'svm': {}}
        
        for cli_arg, (section, attr, transform) in cli_mapping.items():
            if cli_arg in args_dict and args_dict[cli_arg] is not None:
                value = transform(args_dict[cli_arg]) if transform else args_dict[cli_arg]
                if section is None:
                    config_data[attr] = value
                else:
                    config_data[section][attr] = value
        
        return cls.from_dict(config_data)


# Global configuration
_current_config: Optional[Config] = None

def get_config() -> Optional[Config]:
    return _current_config

def set_config(config: Config) -> None:
    global _current_config
    _current_config = config

def reset_config() -> None:
    global _current_config
    _current_config = None