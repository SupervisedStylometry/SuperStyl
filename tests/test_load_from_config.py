import unittest
import os
import tempfile
import json
import pandas as pd
import sys
import glob

from superstyl.load_from_config import load_corpus_from_config


# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Test data paths
        self.test_paths = sorted(glob.glob(os.path.join(THIS_DIR, "testdata/*.txt")))
        
        # Create a test configuration
        self.config = {
            "paths": self.test_paths,
            "format": "txt",
            "sampling": {
                "enabled": False
            },
            "features": [
                {
                    "type": "words",
                    "n": 1,
                    "k": 100,
                    "freq_type": "relative"
                },
                {
                    "type": "chars",
                    "n": 2,
                    "k": 100,
                    "freq_type": "relative"
                }
            ]
        }
        
        # Create a single feature configuration for testing
        self.single_config = {
            "paths": self.test_paths,
            "format": "txt",
            "features": [
                {
                    "name": "words",
                    "type": "words",
                    "n": 1,
                    "k": 100
                }
            ]
        }
        
        # Save configurations to temp files
        self.json_config_path = os.path.join(self.temp_dir.name, "test_config.json")
        with open(self.json_config_path, 'w') as f:
            json.dump(self.config, f)
            
        self.single_config_path = os.path.join(self.temp_dir.name, "single_config.json")
        with open(self.single_config_path, 'w') as f:
            json.dump(self.single_config, f)
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_load_corpus_from_json_config_multiple_features(self):
        # FEATURE: Load corpus from JSON config with multiple features
        # GIVEN: Config file with multiple feature specifications
        
        # WHEN: Loading corpus from config
        corpus, features = load_corpus_from_config(
            config_path=self.json_config_path
        )
        
        # THEN: Corpus and features are loaded correctly
        self.assertIsInstance(corpus, pd.DataFrame)
        self.assertIn("author", corpus.columns)
        self.assertIn("lang", corpus.columns)
        
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)
        
        # Check for prefixed column names in merged dataset
        self.assertTrue(any(col.startswith("words_") for col in corpus.columns))
        self.assertTrue(any(col.startswith("chars_") for col in corpus.columns))
        
        # Check output files
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, "words_feats.json")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, "chars_feats.json")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, "merged_features.csv")))
    
    def test_load_corpus_single_feature(self):
        # FEATURE: Load corpus with a single feature type
        # GIVEN: Config file with a single feature specification
        
        # WHEN: Loading corpus from config
        corpus, features = load_corpus_from_config(
            config_path=self.single_config_path
        )
        
        # THEN: Corpus and features are loaded correctly without prefix
        self.assertIsInstance(corpus, pd.DataFrame)
        self.assertIn("author", corpus.columns)
        self.assertIn("lang", corpus.columns)
        
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)
        
        # Check that we got the direct output, not a merged dataset
        # (No prefixed column names)
        for col in corpus.columns:
            if col not in ["author", "lang"]:
                self.assertFalse("_" in col)
        
        # Check output files
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, "words_feats.json")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, "words.csv")))
    
    def test_load_corpus_with_sampling(self):
        # FEATURE: Load corpus with sampling enabled
        # GIVEN: Config with sampling enabled and sample size defined
        sampling_config = self.config.copy()
        sampling_config["sampling"]["enabled"] = True
        sampling_config["sampling"]["sample_size"] = 2
        
        sampling_config_path = os.path.join(self.temp_dir.name, "sampling_config.json")
        with open(sampling_config_path, 'w') as f:
            json.dump(sampling_config, f)
        
        # WHEN: Loading corpus from config with sampling
        corpus, features = load_corpus_from_config(
            config_path=sampling_config_path
        )
        
        # THEN: Samples are created and file names contain segment info
        first_corpus_index = corpus.index[0]
        self.assertIn("-", first_corpus_index)
    
    def test_load_corpus_with_feature_list(self):
        # FEATURE: Load corpus with a predefined feature list
        # GIVEN: A predefined feature list and config that references it
        feature_list = [["the", 0], ["is", 0], ["text", 0]]
        feature_list_path = os.path.join(self.temp_dir.name, "feature_list.json")
        
        with open(feature_list_path, 'w') as f:
            json.dump(feature_list, f)
        
        # Update config to use feature list
        feature_list_config = self.single_config.copy()
        feature_list_config["features"][0]["feat_list"] = feature_list_path
        
        config_path = os.path.join(self.temp_dir.name, "feature_list_config.json")
        with open(config_path, 'w') as f:
            json.dump(feature_list_config, f)
        
        # WHEN: Loading corpus from config with feature list
        corpus, features = load_corpus_from_config(
            config_path=config_path
        )
        
        # THEN: Only features from the predefined list are used
        feature_words = [f[0] for f in features]
        
        # The feature list should contain only words from the predefined list
        # that actually appear in the corpus
        for word in ["the", "is", "text"]:
            if word in ' '.join([text for text in self.test_paths]):
                self.assertIn(word, feature_words)
    
    def test_invalid_config_format(self):
        # FEATURE: Handle invalid config file format
        # GIVEN: An invalid config file
        invalid_path = os.path.join(self.temp_dir.name, "config.yaml")
        with open(invalid_path, 'w') as f:
            f.write("invalid: format")
        
        # WHEN/THEN: Loading corpus from invalid config raises ValueError
        with self.assertRaises(ValueError):
            load_corpus_from_config(invalid_path)

if __name__ == "__main__":
    unittest.main()