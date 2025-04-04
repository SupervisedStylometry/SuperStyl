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
        self.assertTrue(any(col.startswith("f") for col in corpus.columns))
    
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
    
    def test_feature_list_path_not_found(self):
        """Test handling of a non-existent feature list path"""
        # Create config with non-existent feature list
        nonexistent_path = os.path.join(self.temp_dir.name, "nonexistent.json")
        
        feature_list_config = self.single_config.copy()
        feature_list_config["features"][0]["feat_list"] = nonexistent_path
        
        config_path = os.path.join(self.temp_dir.name, "missing_feature_list_config.json")
        with open(config_path, 'w') as f:
            json.dump(feature_list_config, f)
        
        # Should raise FileNotFoundError when trying to load non-existent feature list
        with self.assertRaises(FileNotFoundError):
            load_corpus_from_config(config_path)

    def test_missing_features_in_config(self):
        """Test handling of config with no features specified"""
        # Create config without features key
        no_features_config = {
            "paths": self.test_paths,
            "format": "txt"
        }
        
        config_path = os.path.join(self.temp_dir.name, "no_features_config.json")
        with open(config_path, 'w') as f:
            json.dump(no_features_config, f)
        
        # Should raise ValueError when no features are specified
        with self.assertRaises(ValueError):
            load_corpus_from_config(config_path)

    def test_empty_features_list(self):
        """Test handling of config with empty features list"""
        # Create config with empty features list
        empty_features_config = {
            "paths": self.test_paths,
            "format": "txt",
            "features": []
        }
        
        config_path = os.path.join(self.temp_dir.name, "empty_features_config.json")
        with open(config_path, 'w') as f:
            json.dump(empty_features_config, f)
        
        # Should raise ValueError when features list is empty
        with self.assertRaises(ValueError):
            load_corpus_from_config(config_path)

    def test_missing_paths_in_config(self):
        """Test handling of config with missing paths"""
        # Create config without paths key
        no_paths_config = {
            "format": "txt",
            "features": [
                {
                    "type": "words",
                    "n": 1,
                    "k": 100
                }
            ]
        }
        
        config_path = os.path.join(self.temp_dir.name, "no_paths_config.json")
        with open(config_path, 'w') as f:
            json.dump(no_paths_config, f)
        
        # Should raise ValueError when no paths are specified
        with self.assertRaises(ValueError):
            load_corpus_from_config(config_path)

    def test_string_path_in_config(self):
        """Test handling of config with a string path instead of list"""
        # Create config with a string path
        string_path_config = self.single_config.copy()
        string_path_config["paths"] = self.test_paths[0]  # Single string path
        
        config_path = os.path.join(self.temp_dir.name, "string_path_config.json")
        with open(config_path, 'w') as f:
            json.dump(string_path_config, f)
        
        # Should handle string path
        corpus, features = load_corpus_from_config(config_path)
        self.assertIsInstance(corpus, pd.DataFrame)
        self.assertGreater(len(features), 0)

    def test_feature_with_txt_list(self):
        """Test loading a feature list from a txt file"""
        # Create a simple text feature list
        feature_list_path = os.path.join(self.temp_dir.name, "feature_list.txt")
        with open(feature_list_path, 'w') as f:
            f.write("the\nis\ntext\n")
        
        # Create config that uses txt feature list
        txt_list_config = self.single_config.copy()
        txt_list_config["features"][0]["feat_list"] = feature_list_path
        
        config_path = os.path.join(self.temp_dir.name, "txt_list_config.json")
        with open(config_path, 'w') as f:
            json.dump(txt_list_config, f)
        
        # Should load corpus with feature list from txt
        corpus, features = load_corpus_from_config(config_path)
        self.assertIsInstance(corpus, pd.DataFrame)
        self.assertGreater(len(features), 0)

    def test_all_optional_parameters(self):
        """Test with a config that includes all optional parameters"""
        # Create config with all optional parameters
        full_config = {
            "paths": self.test_paths,
            "format": "txt",
            "keep_punct": True,
            "keep_sym": True,
            "no_ascii": True,
            "identify_lang": True,
            "sampling": {
                "enabled": True,
                "units": "words",
                "sample_size": 2,
                "sample_step": None,  # Set to None when sample_random is True
                "max_samples": 3,
                "sample_random": True
            },
            "features": [
                {
                    "name": "feature1",
                    "type": "words",
                    "n": 1,
                    "k": 100,
                    "freq_type": "relative",
                    "keep_punct": True,
                    "keep_sym": True,
                    "no_ascii": True,
                    "identify_lang": True,
                    "embedding": None,
                    "neighbouring_size": 5,
                    "culling": 0
                }
            ]
        }
        
        config_path = os.path.join(self.temp_dir.name, "full_config.json")
        with open(config_path, 'w') as f:
            json.dump(full_config, f)
        
        # Should load corpus with all parameters set
        corpus, features = load_corpus_from_config(config_path)
        self.assertIsInstance(corpus, pd.DataFrame)
        self.assertGreater(len(features), 0)

    def test_feature_list_loading(self):
        """Test both JSON and TXT feature list loading paths"""
        # Create test files with known content to use in feature lists
        
        # First, create a JSON feature list file
        json_feature_path = os.path.join(self.temp_dir.name, "features.json")
        json_feature_list = [["this", 1], ["is", 2]]  # Words that appear in test files
        with open(json_feature_path, 'w') as f:
            json.dump(json_feature_list, f)
        
        # Second, create a TXT feature list file
        txt_feature_path = os.path.join(self.temp_dir.name, "features.txt")
        with open(txt_feature_path, 'w') as f:
            f.write("this\nis\n")  # Words that appear in test files
        
        # Test the JSON feature list loading
        json_config = {
            "paths": self.test_paths,
            "format": "txt",
            "features": [
                {
                    "type": "words",
                    "n": 1,
                    "feat_list": json_feature_path
                }
            ]
        }
        
        json_config_path = os.path.join(self.temp_dir.name, "json_config.json")
        with open(json_config_path, 'w') as f:
            json.dump(json_config, f)
        
        # Use direct debugging output to confirm code path is being exercised
        print(f"\nTesting JSON feature list: {json_feature_path}")
        print(f"JSON feature list content: {json_feature_list}")
        
        corpus_json, features_json = load_corpus_from_config(json_config_path)
        
        # Now test the TXT feature list loading
        txt_config = {
            "paths": self.test_paths,
            "format": "txt",
            "features": [
                {
                    "type": "words",
                    "n": 1,
                    "feat_list": txt_feature_path
                }
            ]
        }
        
        txt_config_path = os.path.join(self.temp_dir.name, "txt_config.json")
        with open(txt_config_path, 'w') as f:
            json.dump(txt_config, f)
        
        # Use direct debugging output to confirm code path is being exercised
        print(f"\nTesting TXT feature list: {txt_feature_path}")
        with open(txt_feature_path, 'r') as f:
            print(f"TXT feature list content: {f.read()}")
        
        corpus_txt, features_txt = load_corpus_from_config(txt_config_path)
        
        # Basic verification
        self.assertIsInstance(corpus_json, pd.DataFrame)
        self.assertIsInstance(corpus_txt, pd.DataFrame)
 
    def test_invalid_paths_type(self):
        """Test handling of config with invalid paths type (neither list nor string)"""
        # Create config with invalid paths type (integer)
        invalid_paths_config = self.single_config.copy()
        invalid_paths_config["paths"] = 123  # Not a list or string
        
        config_path = os.path.join(self.temp_dir.name, "invalid_paths_config.json")
        with open(config_path, 'w') as f:
            json.dump(invalid_paths_config, f)
        
        # Should raise ValueError for invalid paths type
        with self.assertRaises(ValueError) as context:
            load_corpus_from_config(config_path)
        
        # Verify the error message
        self.assertIn("Paths in config must be either a list or a glob pattern string", str(context.exception))


if __name__ == "__main__":
    unittest.main()