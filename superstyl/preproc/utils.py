import unidecode
import langdetect
import unicodedata
import regex as re

from superstyl.config import NormalizationConfig



def extract_author_from_path(path: str) -> str:
    """Extract author from file path (before first underscore)."""
    return path.split('/')[-1].split("_")[0]


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r"\s+", " ", text).strip()


def detect_lang(text: str) -> str:
    """Detect language of text using langdetect."""
    return langdetect.detect(text)


def normalise(text: str, norm_config : NormalizationConfig=NormalizationConfig()) -> str:
    """
    Normalize input text according to specified options.
    
    Args:
        text: Input text to normalize
        keep_punct: Keep punctuation and case distinction
        keep_sym: Keep punctuation, case, numbers, symbols, marks
        no_ascii: Disable conversion to ASCII
    
    Returns:
        Normalized text
    """
    if norm_config.keep_sym:
        out = re.sub(r"[^\p{L}\p{P}\p{N}\p{S}\p{M}\p{Co}]+", " ", text)
    else:
        if norm_config.keep_punct:
            out = re.sub(r"[^\p{L}\p{P}\p{M}]+", " ", text)
        else:
            out = re.sub(r"[^\p{L}\p{M}]+", " ", text.lower())
        
        if not norm_config.no_ascii:
            out = unidecode.unidecode(out)
    
    out = unicodedata.normalize("NFC", out)
    return normalize_whitespace(out)