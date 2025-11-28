
from dataclasses import dataclass, field, fields, asdict
from typing import Optional, List, Any, Dict, Type, TypeVar
from pathlib import Path
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
        """
        Convert configuration to dictionary.
        """
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, BaseConfig):
                result[f.name] = value.to_dict()
            else:
                result[f.name] = value
        return result
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Create configuration from dictionary.
        """
        return cls(**data)
    
    def validate(self) -> None:
        """
        Validate configuration. Override in subclasses.
        """
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
            raise ValueError(f"Invalid sampling units: {self.units}. Must be 'words' or 'verses'.")
        if self.random and self.step is not None:
            raise ValueError("Random sampling is not compatible with continuous sampling (step).")
        if self.random and self.max_samples is None:
            raise ValueError("Random sampling needs a fixed number of samples (max_samples).")


@dataclass
class FeatureConfig(BaseConfig):
    """
    Configuration for feature extraction.
    """
    type: str = "words"
    n: int = 1
    k: int = 5000
    freq_type: str = "relative"
    feat_list: Optional[List] = None
    embedding: Optional[str] = None
    neighbouring_size: int = 10
    culling: float = 0
    normalization: Optional[NormalizationConfig] = None

    VALID_TYPES = ["words", "chars", "affixes", "lemma", "pos", "met_line", "met_syll"]
    VALID_FREQ_TYPES = ["relative", "absolute", "binary"]

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if self.type not in self.VALID_TYPES:
            raise ValueError(f"Invalid feature type: {self.type}. Must be one of {self.VALID_TYPES}.")
        if self.freq_type not in self.VALID_FREQ_TYPES:
            raise ValueError(f"Invalid frequency type: {self.freq_type}. Must be one of {self.VALID_FREQ_TYPES}.")
        if self.n < 1:
            raise ValueError("n must be a positive integer.")

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
            raise ValueError(f"Invalid format: {self.format}. Must be one of {self.VALID_FORMATS}.")


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
            raise ValueError(f"Invalid cross_validate: {self.cross_validate}. Must be one of {self.VALID_CV}.")
        if self.dim_reduc not in self.VALID_DIM_REDUC:
            raise ValueError(f"Invalid dim_reduc: {self.dim_reduc}. Must be one of {self.VALID_DIM_REDUC}.")
        if self.balance not in self.VALID_BALANCE:
            raise ValueError(f"Invalid balance: {self.balance}. Must be one of {self.VALID_BALANCE}.")
        if self.kernel not in self.VALID_KERNELS:
            raise ValueError(f"Invalid kernel: {self.kernel}. Must be one of {self.VALID_KERNELS}.")


@dataclass  
class Config(BaseConfig):
    """
    Main configuration class for SuperStyl.
    
    Aggregates all sub-configurations and provides factory methods
    to create configurations from various sources (CLI, JSON, dict).
    """
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    svm: SVMConfig = field(default_factory=SVMConfig)
    output_prefix: Optional[str] = None

    def validate(self) -> None:
        """Validate configuration consistency across sub-configs."""
        # TEI-specific features require TEI format
        tei_only_feats = ["lemma", "pos", "met_line", "met_syll"]
        if self.features.type in tei_only_feats and self.corpus.format != "tei":
            raise ValueError(
                f"{self.features.type} features are only possible with TEI format, "
                f"but format is '{self.corpus.format}'."
            )
        
        # Metrical features require lines units
        if self.features.type in ["met_line", "met_syll"] and self.sampling.units != "lines":
            raise ValueError(
                f"{self.features.type} features are only possible with lines units in TEI format."
            )

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        kwargs = {}
        
        # Map of field names to their config classes
        config_classes = {
            'corpus': CorpusConfig,
            'features': FeatureConfig,
            'sampling': SamplingConfig,
            'normalization': NormalizationConfig,
            'svm': SVMConfig,
        }
        
        for key, config_cls in config_classes.items():
            if key in data:
                sub_data = data[key].copy()
                # Handle sampling field name mapping
                if key == 'sampling':
                    if 'sample_size' in sub_data:
                        sub_data['size'] = sub_data.pop('sample_size')
                    if 'samples_random' in sub_data:
                        sub_data['random'] = sub_data.pop('samples_random')
                kwargs[key] = config_cls.from_dict(sub_data)
        
        if 'output_prefix' in data:
            kwargs['output_prefix'] = data['output_prefix']
        
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_cli_args(cls, args) -> 'Config':
        """
        Create configuration from CLI argument namespace.
        
        Uses a mapping approach instead of hasattr() chains.
        """
        args_dict = vars(args)
        
        # Mapping: CLI arg name -> (config_section, config_attr, transform_func)
        # transform_func is optional, defaults to identity
        cli_mapping = {
            # Corpus
            's': ('corpus', 'paths', lambda x: x if isinstance(x, list) else [x]),
            'x': ('corpus', 'format', None),
            'identify_lang': ('corpus', 'identify_lang', None),
            
            # Features
            't': ('features', 'type', None),
            'n': ('features', 'n', None),
            'k': ('features', 'k', None),
            'freqs': ('features', 'freq_type', None),
            'embedding': ('features', 'embedding', lambda x: x if x else None),
            'neighbouring_size': ('features', 'neighbouring_size', None),
            'culling': ('features', 'culling', None),
            
            # Sampling
            'sampling': ('sampling', 'enabled', None),
            'sample_units': ('sampling', 'units', None),
            'sample_size': ('sampling', 'size', None),
            'sample_step': ('sampling', 'step', None),
            'max_samples': ('sampling', 'max_samples', None),
            'samples_random': ('sampling', 'random', None),
            
            # Normalization
            'keep_punct': ('normalization', 'keep_punct', None),
            'keep_sym': ('normalization', 'keep_sym', None),
            'no_ascii': ('normalization', 'no_ascii', None),
            
            # Output
            'o': (None, 'output_prefix', None),
        }
        
        # Build config dict from args using mapping
        config_data = {
            'corpus': {},
            'features': {},
            'sampling': {},
            'normalization': {},
        }
        
        for cli_arg, (section, attr, transform) in cli_mapping.items():
            if cli_arg in args_dict and args_dict[cli_arg] is not None:
                value = args_dict[cli_arg]
                if transform:
                    value = transform(value)
                
                if section is None:
                    config_data[attr] = value
                else:
                    config_data[section][attr] = value
        
        return cls.from_dict(config_data)

    @classmethod
    def from_svm_cli_args(cls, args) -> 'Config':
        """Create configuration from SVM CLI argument namespace."""
        args_dict = vars(args)
        
        # Mapping for SVM CLI args
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
                value = args_dict[cli_arg]
                if transform:
                    value = transform(value)
                
                if section is None:
                    config_data[attr] = value
                else:
                    config_data[section][attr] = value
        
        return cls.from_dict(config_data)


# Global configuration management
_current_config: Optional[Config] = None


def get_config() -> Optional[Config]:
    """Get the current global configuration."""
    return _current_config


def set_config(config: Config) -> None:
    """Set the global configuration."""
    global _current_config
    _current_config = config


def reset_config() -> None:
    """Reset the global configuration."""
    global _current_config
    _current_config = None