import unittest
import superstyl.load
import superstyl.preproc.features_extract
from superstyl.load_from_config import load_corpus_from_config
import os
import tempfile
import json
import glob

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class ErrorHandlingTests(unittest.TestCase):
    """Tests for error handling and ValueError raising"""
    
    def setUp(self):
        """Set up test files paths"""
        self.test_paths = sorted(glob.glob(os.path.join(THIS_DIR, "testdata/*.txt")))
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()
    
    # =========================================================================
    # Tests pour load.py - ValueError pour formats incompatibles
    # =========================================================================
    
    def test_load_corpus_lemma_requires_tei(self):
        # SCENARIO: lemma features require TEI format
        # GIVEN: Attempting to use lemma with non-TEI format
        
        # WHEN/THEN: Should raise ValueError
        with self.assertRaises(ValueError) as context:
            superstyl.load.load_corpus(
                self.test_paths,
                feats="lemma",
                format="txt"
            )
        
        self.assertIn("lemma", str(context.exception))
        self.assertIn("tei", str(context.exception).lower())
    
    def test_load_corpus_pos_requires_tei(self):
        # SCENARIO: pos features require TEI format
        # GIVEN: Attempting to use pos with non-TEI format
        
        # WHEN/THEN: Should raise ValueError
        with self.assertRaises(ValueError) as context:
            superstyl.load.load_corpus(
                self.test_paths,
                feats="pos",
                format="txt"
            )
        
        self.assertIn("pos", str(context.exception))
        self.assertIn("tei", str(context.exception).lower())
    
    def test_load_corpus_met_line_requires_tei(self):
        # SCENARIO: met_line features require TEI format
        # GIVEN: Attempting to use met_line with non-TEI format
        
        # WHEN/THEN: Should raise ValueError
        with self.assertRaises(ValueError) as context:
            superstyl.load.load_corpus(
                self.test_paths,
                feats="met_line",
                format="txt"
            )
        
        self.assertIn("met_line", str(context.exception))
        self.assertIn("tei", str(context.exception).lower())
    
    def test_load_corpus_met_syll_requires_tei(self):
        # SCENARIO: met_syll features require TEI format
        # GIVEN: Attempting to use met_syll with non-TEI format
        
        # WHEN/THEN: Should raise ValueError
        with self.assertRaises(ValueError) as context:
            superstyl.load.load_corpus(
                self.test_paths,
                feats="met_syll",
                format="txt"
            )
        
        self.assertIn("met_syll", str(context.exception))
        self.assertIn("tei", str(context.exception).lower())
    
    def test_load_corpus_met_line_requires_lines_unit(self):
        # SCENARIO: met_line requires units='lines'
        # GIVEN: Attempting to use met_line with units='words'
        
        # Create a dummy TEI file for this test
        tei_path = os.path.join(self.temp_dir.name, "test_met.xml")
        with open(tei_path, 'w') as f:
            f.write('<?xml version="1.0"?><TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><l met="01">test</l></body></text></TEI>')
        
        # WHEN/THEN: Should raise ValueError
        with self.assertRaises(ValueError) as context:
            superstyl.load.load_corpus(
                [tei_path],
                feats="met_line",
                format="tei",
                units="words"  # Wrong unit type
            )
        
        self.assertIn("met_line", str(context.exception))
        self.assertIn("lines", str(context.exception))
    
    def test_load_corpus_met_syll_requires_lines_unit(self):
        # SCENARIO: met_syll requires units='lines'
        # GIVEN: Attempting to use met_syll with units='words'
        
        # Create a dummy TEI file for this test
        tei_path = os.path.join(self.temp_dir.name, "test_met2.xml")
        with open(tei_path, 'w') as f:
            f.write('<?xml version="1.0"?><TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><l met="01">test</l></body></text></TEI>')
        
        # WHEN/THEN: Should raise ValueError
        with self.assertRaises(ValueError) as context:
            superstyl.load.load_corpus(
                [tei_path],
                feats="met_syll",
                format="tei",
                units="words"  # Wrong unit type
            )
        
        self.assertIn("met_syll", str(context.exception))
        self.assertIn("lines", str(context.exception))
    
    # =========================================================================
    # Tests pour features_extract.py - ValueError pour paramètres invalides
    # =========================================================================
    
    def test_count_features_empty_text(self):
        # SCENARIO: Empty text should raise ValueError
        # GIVEN: An empty string as text
        
        # WHEN/THEN: Should raise ValueError
        with self.assertRaises(ValueError) as context:
            superstyl.preproc.features_extract.count_features(
                "",  # Empty text
                feats="words",
                n=1
            )
        
        self.assertIn("empty", str(context.exception).lower())
    
    def test_count_features_invalid_n_zero(self):
        # SCENARIO: n must be positive
        # GIVEN: n=0
        
        # WHEN/THEN: Should raise ValueError
        with self.assertRaises(ValueError) as context:
            superstyl.preproc.features_extract.count_features(
                "test text",
                feats="words",
                n=0  # Invalid n
            )
        
        self.assertIn("positive", str(context.exception).lower())
    
    def test_count_features_invalid_n_negative(self):
        # SCENARIO: n must be positive
        # GIVEN: n=-1
        
        # WHEN/THEN: Should raise ValueError
        with self.assertRaises(ValueError) as context:
            superstyl.preproc.features_extract.count_features(
                "test text",
                feats="words",
                n=-1  # Invalid n
            )
        
        self.assertIn("positive", str(context.exception).lower())
    
    def test_count_features_invalid_n_not_integer(self):
        # SCENARIO: n must be an integer
        # GIVEN: n=1.5 (float)
        
        # WHEN/THEN: Should raise ValueError
        with self.assertRaises(ValueError) as context:
            superstyl.preproc.features_extract.count_features(
                "test text",
                feats="words",
                n=1.5  # Not an integer
            )
        
        self.assertIn("integer", str(context.exception).lower())
    
    def test_count_features_invalid_not_string(self):
        # SCENARIO: text must be a string
        # GIVEN: text is not a string (e.g., None)
        
        # WHEN/THEN: Should raise ValueError
        with self.assertRaises(ValueError) as context:
            superstyl.preproc.features_extract.count_features(
                None,  # Not a string
                feats="words",
                n=1
            )
        
        self.assertIn("string", str(context.exception).lower())
    
    def test_count_features_unsupported_feats_type(self):
        # SCENARIO: feats must be a supported type
        # GIVEN: An unsupported feats type
        
        # WHEN/THEN: Should raise ValueError
        with self.assertRaises(ValueError) as context:
            superstyl.preproc.features_extract.count_features(
                "test text",
                feats="unsupported_type",  # Invalid feats type
                n=1
            )
        
        self.assertIn("Unsupported", str(context.exception))
    
    def test_get_counts_invalid_frequency_type(self):
        # SCENARIO: freqsType must be valid
        # GIVEN: An unsupported frequency type
        
        myTexts = [{"name": "test", "text": "test text"}]
        
        # WHEN/THEN: Should raise ValueError
        with self.assertRaises(ValueError) as context:
            superstyl.preproc.features_extract.get_counts(
                myTexts,
                feats="words",
                freqsType="invalid_type"  # Invalid frequency type
            )
        
        self.assertIn("Unsupported frequency type", str(context.exception))
    
    # =========================================================================
    # Tests pour load_from_config.py - Branches non couvertes
    # =========================================================================
    
    def test_load_from_config_with_json_feature_list(self):
        # SCENARIO: Load corpus with JSON feature list (ligne 119)
        # GIVEN: A config with a JSON feature list
        
        # Create a JSON feature list
        feature_list = [["the", 0], ["is", 0]]
        feature_list_path = os.path.join(self.temp_dir.name, "features.json")
        with open(feature_list_path, 'w') as f:
            json.dump(feature_list, f)
        
        # Create config
        config = {
            "paths": self.test_paths,
            "format": "txt",
            "features": [
                {
                    "name": "test_feature",
                    "type": "words",
                    "n": 1,
                    "feat_list": feature_list_path  # JSON feature list
                }
            ]
        }
        
        config_path = os.path.join(self.temp_dir.name, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # WHEN: Loading corpus from config
        corpus, features = load_corpus_from_config(config_path)
        
        # THEN: Should load successfully with JSON feature list
        self.assertIsNotNone(corpus)
        self.assertIsNotNone(features)
    
    def test_load_from_config_test_mode_uses_feat_list(self):
        # SCENARIO: In test mode, use provided feat_list (ligne 156)
        # GIVEN: A config with feat_list in test mode
        
        # Create a JSON feature list
        feature_list = [["the", 0], ["is", 0], ["text", 0]]
        feature_list_path = os.path.join(self.temp_dir.name, "test_features.json")
        with open(feature_list_path, 'w') as f:
            json.dump(feature_list, f)
        
        # Create config with multiple features (triggers is_test logic)
        config = {
            "paths": self.test_paths,
            "format": "txt",
            "features": [
                {
                    "name": "feat1",
                    "type": "words",
                    "n": 1,
                    "feat_list": feature_list_path
                },
                {
                    "name": "feat2",
                    "type": "chars",
                    "n": 2,
                    "feat_list": feature_list_path
                }
            ]
        }
        
        config_path = os.path.join(self.temp_dir.name, "multi_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # WHEN: Loading corpus from config
        corpus, features = load_corpus_from_config(config_path, is_test=True)
        
        # THEN: Should use the provided feature list
        self.assertIsNotNone(corpus)
        self.assertIsNotNone(features)
        # features should be a list of feature lists
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 2)  # Two feature sets


if __name__ == '__main__':
    unittest.main()