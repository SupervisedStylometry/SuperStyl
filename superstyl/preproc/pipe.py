from lxml import etree
import nltk.tokenize
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from superstyl.config import Config, NormalizationConfig, SamplingConfig
from superstyl.preproc.utils import *


# ============================================================================
# Constants and Configuration
# ============================================================================

XSLT_TEMPLATES = {
    'xml_text': '''
        <xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
            xmlns:xs="http://www.w3.org/2001/XMLSchema"
            exclude-result-prefixes="xs"
            version="1.0">
            <xsl:output method="text"/>
            <xsl:template match="/">
                <xsl:apply-templates select="song/text"/>
            </xsl:template>
        </xsl:stylesheet>''',
    
    'tei_units': '''
        <xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
            xmlns:tei="http://www.tei-c.org/ns/1.0"  
            version="1.0">
            <xsl:output method="text"/>
            <xsl:param name="units"></xsl:param>
            <xsl:param name="feats"></xsl:param>
            <xsl:param name="keep_punct"></xsl:param>
            
            <xsl:template match="/">
                <xsl:choose>
                    <xsl:when test="$units = 'verses'">
                        <xsl:apply-templates select="descendant::tei:l"/>
                    </xsl:when>
                    <xsl:when test="$units = 'words'">
                        <xsl:apply-templates select="descendant::tei:w"/>
                    </xsl:when>
                </xsl:choose>
            </xsl:template>
            
            <xsl:template match="tei:l">
                <xsl:choose>
                    <xsl:when test="$feats = 'met'">
                        <xsl:choose>
                            <xsl:when test="$keep_punct = 'true'">
                                <xsl:value-of select="@met"/>
                            </xsl:when>
                            <xsl:otherwise>
                                <xsl:value-of select="translate(@met, '.', '')"/>
                            </xsl:otherwise>
                        </xsl:choose>
                    </xsl:when>
                    <xsl:otherwise>
                        <xsl:apply-templates select="descendant::tei:w"/>
                    </xsl:otherwise>
                </xsl:choose>
                <xsl:text>&#xA;</xsl:text>
            </xsl:template>
            
            <xsl:template match="tei:w">
                <xsl:text> </xsl:text>
                <xsl:choose>
                    <xsl:when test="$feats = 'met'">
                        <xsl:value-of select="@met"/>
                    </xsl:when>
                    <xsl:when test="$feats = 'lemma'">
                        <xsl:value-of select="@lemma"/>
                    </xsl:when>
                    <xsl:when test="$feats = 'pos'">
                        <xsl:value-of select="@pos"/>
                    </xsl:when>
                    <xsl:otherwise>
                        <xsl:apply-templates/>
                    </xsl:otherwise>
                </xsl:choose>
                <xsl:if test="$units = 'words'">
                    <xsl:text>&#xA;</xsl:text>
                </xsl:if>
            </xsl:template>
        </xsl:stylesheet>''',
    
    'txm_units': '''
        <xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
            xmlns:tei="http://www.tei-c.org/ns/1.0" 
            xmlns:txm="http://textometrie.org/1.0" 
            version="1.0">
            <xsl:output method="text"/>
            <xsl:param name="units"></xsl:param>
            <xsl:param name="feats"></xsl:param>
            
            <xsl:template match="/">
                <xsl:choose>
                    <xsl:when test="$units = 'verses'">
                        <xsl:apply-templates select="descendant::tei:l"/>
                    </xsl:when>
                    <xsl:when test="$units = 'words'">
                        <xsl:apply-templates select="descendant::tei:w"/>
                    </xsl:when>
                </xsl:choose>
            </xsl:template>
            
            <xsl:template match="tei:l">
                <xsl:apply-templates select="descendant::tei:w[
                    not(txm:ana[@type='#frpos'] = 'NOMpro')]"/>
                <xsl:text>&#xA;</xsl:text>
            </xsl:template>
            
            <xsl:template match="tei:w">
                <xsl:text> </xsl:text>
                <xsl:choose>
                    <xsl:when test="$feats = 'lemma'">
                        <xsl:value-of select="txm:lemma"/>
                    </xsl:when>
                    <xsl:when test="$feats = 'pos'">
                        <xsl:value-of select="txm:ana[@type='#frpos']"/>
                    </xsl:when>
                    <xsl:otherwise>
                        <xsl:apply-templates select="txm:form"/>
                    </xsl:otherwise>
                </xsl:choose>
                <xsl:if test="$units = 'words'">
                    <xsl:text>&#xA;</xsl:text>
                </xsl:if>
            </xsl:template>
        </xsl:stylesheet>'''
}




class FileLoader(ABC):
    """Abstract base class for file loaders."""
    
    @abstractmethod
    def load(self, path: str, **kwargs) -> Tuple[str, str]:
        """Load file and return (author, text) tuple."""
        pass


class TXTLoader(FileLoader):
    """Loader for plain text files."""
    
    def load(self, path: str, **kwargs) -> Tuple[str, str]:
        with open(path, 'r') as f:
            text = ' '.join(f.readlines())
        
        author = extract_author_from_path(path)
        return author, normalize_whitespace(text)


class XMLLoader(FileLoader):
    """Loader for XML files."""
    
    def __init__(self):
        self.xslt = etree.XSLT(etree.XML(XSLT_TEMPLATES['xml_text']))
    
    def load(self, path: str, **kwargs) -> Tuple[str, str]:
        with open(path, 'r') as f:
            doc = etree.parse(f)
        
        authors = doc.findall("//author")
        author_texts = [a.text for a in authors]
        
        if len(author_texts) != 1:
            print(f"Warning: Expected 1 author in {path}, found {len(author_texts)}")
            author = author_texts[0] if author_texts else "unknown"
        else:
            author = author_texts[0]
        
        text = str(self.xslt(doc))
        return author, normalize_whitespace(text)


class XMLUnitLoader(ABC):
    """Base class for XML loaders that extract units."""
    
    def __init__(self, template_name: str):
        self.xslt = etree.XSLT(etree.XML(XSLT_TEMPLATES[template_name]))
    
    def extract_units(self, path: str, units: str = "verses", 
                     feats: str = "words") -> List[str]:
        """Extract units from XML file."""
        with open(path, 'r') as f:
            doc = etree.parse(f)
        
        params = self._get_xslt_params(units, feats)
        result = str(self.xslt(doc, **params))
        return result.splitlines()
    
    @abstractmethod
    def _get_xslt_params(self, units: str, feats: str) -> Dict:
        """Get XSLT parameters for transformation."""
        pass


class TEIUnitLoader(XMLUnitLoader):
    """Loader for TEI files with unit extraction."""
    
    def __init__(self):
        super().__init__('tei_units')
    
    def _get_xslt_params(self, units: str, feats: str) -> Dict:
        feats_param = "met" if feats in ["met_syll", "met_line"] else feats
        return {
            'units': etree.XSLT.strparam(units),
            'feats': etree.XSLT.strparam(feats_param)
        }
    
    def load(self, path: str, feats: str = "words", **kwargs) -> Tuple[str, str]:
        """Load TEI file and return (author, text) tuple."""
        author = extract_author_from_path(path)
        units = self.extract_units(path, units="words", feats=feats)
        text = normalize_whitespace(' '.join(units))
        return author, text


class TXMUnitLoader(XMLUnitLoader):
    """Loader for TXM files with unit extraction."""
    
    def __init__(self):
        super().__init__('txm_units')
    
    def _get_xslt_params(self, units: str, feats: str) -> Dict:
        return {
            'units': etree.XSLT.strparam(units),
            'feats': etree.XSLT.strparam(feats)
        }
    
    def load(self, path: str, feats: str = "words", **kwargs) -> Tuple[str, str]:
        """Load TXM file and return (author, text) tuple."""
        author = extract_author_from_path(path)
        units = self.extract_units(path, units="words", feats=feats)
        text = normalize_whitespace(' '.join(units))
        return author, text


# Loader factory
LOADERS = {
    'txt': TXTLoader(),
    'xml': XMLLoader(),
    'tei': TEIUnitLoader(),
    'txm': TXMUnitLoader()
}


def XML_to_text(path: str) -> Tuple[str, str]:
    """Legacy function for XML loading."""
    return LOADERS['xml'].load(path)


def TXT_to_text(path: str) -> Tuple[str, str]:
    """Legacy function for TXT loading."""
    return LOADERS['txt'].load(path)


def tei_to_units(path: str, feats: str = "words", units: str = "verses") -> List[str]:
    """Legacy function for TEI unit extraction."""
    return LOADERS['tei'].extract_units(path, units, feats)


def txm_to_units(path: str, units: str = "verses", feats: str = "words") -> List[str]:
    """Legacy function for TXM unit extraction."""
    return LOADERS['txm'].extract_units(path, units, feats)


def specialXML_to_text(path: str, format: str = "tei", feats: str = "words") -> Tuple[str, str]:
    """Legacy function for special XML loading."""
    return LOADERS[format].load(path, feats=feats)


# ============================================================================
# Sampling Functions
# ============================================================================

class Sampler:
    """
    Handles text sampling operations.
    """
    
    @staticmethod
    def extract_tokens(path: str, config: Config=Config()) -> List[str]:
        """
        Extract tokens from a document based on format and units.
        """
        feats=config.features[0].type

        if config.sampling.units == "words" and config.corpus.format == "txt":
            author, text = LOADERS['txt'].load(path)
            text = normalise(text, config.normalization)
            return nltk.tokenize.wordpunct_tokenize(text)
        
        elif config.corpus.format == "tei":
            return LOADERS['tei'].extract_units(path, config.corpus.units, feats)
        
        elif config.sampling.units == "verses" and config.corpus.format == "txm":
            return LOADERS['txm'].extract_units(path, config.sampling.units, feats)
        
        else:
            raise ValueError(f"Unsupported combination: units={config.sampling.units}, format={config.corpus.format}")
    
    @staticmethod
    def create_samples(tokens: List[str], sampling_config: SamplingConfig=SamplingConfig()) -> List[Dict]:
        """
        Create samples from tokens.
        """
        step = sampling_config.step if sampling_config.step is not None else sampling_config.size
        
        samples = []
        
        if sampling_config.random:
            for k in range(sampling_config.max_samples):
                samples.append({
                    "start": f"{k}s",
                    "end": f"{k}e",
                    "text": list(random.choices(tokens, k=sampling_config.size))
                })
        else:
            current = 0
            while current + sampling_config.size <= len(tokens):
                samples.append({
                    "start": current,
                    "end": current + sampling_config.size,
                    "text": list(tokens[current:current + sampling_config.size])
                })
                current += step
        
        return samples
    
    @classmethod
    def get_samples(cls, path: str, config: Config=Config()) -> List[Dict]:
        """
        Extract samples from a document.
        
        Args:
            path: Path to document
            config: Config file

        Returns:
            List of sample dictionaries
        """
        max_samples = config.sampling.max_samples or 10
        config.sampling.validate()
        
        tokens = cls.extract_tokens(path, config)
        return cls.create_samples(tokens, config.sampling)


def max_sampling(documents: List[Dict], max_samples: int = 10) -> List[Dict]:
    """
    Randomly select up to max_samples per author/class.
    
    Args:
        documents: List of text dict
        max_samples: Maximum samples per author
    
    Returns:
        Filtered list of documents
    """
    # Count documents per author
    author_counts = {}
    for doc in documents:
        author_counts[doc['aut']] = author_counts.get(doc['aut'], 0) + 1
    
    # Filter authors with too many samples
    result = []
    for author, count in author_counts.items():
        author_docs = [d for d in documents if d['aut'] == author]

        if count > max_samples:
            result.extend(random.sample(author_docs, k=max_samples))
        else:
            result.extend(author_docs)
    
    return result


# ============================================================================
# Main Loading Functions
# ============================================================================

def load_texts(paths: List[str], config: Config=Config()) -> List[Dict]:
    """
    Load a collection of documents.
    
    Args:
        paths: List of file paths
        config: Config file
    
    Returns:
        List of document dictionaries
    """
    loader = LOADERS.get(config.corpus.format)
    if not loader:
        raise ValueError(f"Unsupported format: {config.corpus.format}")
    
    documents = []
    feats=config.features[0].type

    for path in paths:
        name = path.split('/')[-1]
        author, text = loader.load(path, feats=feats)
        
        lang = detect_lang(text) if config.corpus.identify_lang else "NA"
        
        # Normalize text
        text = normalise(text, config.normalization)
        
        documents.append({
            "name": name,
            "aut": author,
            "text": text,
            "lang": lang
        })
    
    if config.sampling.max_samples is not None:
        documents = max_sampling(documents, config.sampling.max_samples)
    
    return documents


def docs_to_samples(paths: List[str], config: Config=Config()) -> List[Dict]:
    """
    Load documents with sampling.
    
    Args:
        paths: List of file paths
        config: Config file
    
    Returns:
        List of sample dictionaries
    """
    loader = LOADERS.get(config.corpus.format)
    if not loader:
        raise ValueError(f"Unsupported format: {config.corpus.format}")
    
    all_samples = []
    feats=config.features[0].type

    for path in paths:
        author = extract_author_from_path(path)
        
        # Detect language if needed
        if config.corpus.identify_lang:
            _, text = loader.load(path, feats=feats)
            lang = detect_lang(text)
        else:
            lang = 'NA'
        
        # Get samples
        samples = Sampler.get_samples(path, config)
        
        # Create sample documents
        for sample in samples:
            name = f"{path.split('/')[-1]}_{sample['start']}-{sample['end']}"
            text = normalise(' '.join(sample['text']), config.normalization)
            
            all_samples.append({
                "name": name,
                "aut": author,
                "text": text,
                "lang": lang
            })
    
    if config.sampling.max_samples is not None:
        all_samples = max_sampling(all_samples, config.sampling.max_samples)
    
    return all_samples