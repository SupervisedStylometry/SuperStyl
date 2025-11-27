import unittest
import tempfile
import os
import csv
import json
import pandas as pd
from superstyl.preproc.select import (
    _load_metadata, 
    _should_exclude, 
    read_clean, 
    apply_selection
)


class TestSelectRefactored(unittest.TestCase):
    """Tests for the refactored select.py module"""
    
    def setUp(self):
        """Create temporary directory and test files"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name
        
        # Create a sample CSV file
        self.csv_path = os.path.join(self.temp_path, "test_data.csv")
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Unnamed: 0', 'author', 'lang', 'feature1', 'feature2'])
            writer.writerow(['doc1', 'Smith', 'en', 0.5, 0.3])
            writer.writerow(['doc2', 'Dupont', 'fr', 0.4, 0.6])
            writer.writerow(['doc3', 'Garcia', 'es', 0.7, 0.2])
            writer.writerow(['doc4', 'Smith', 'en', 0.6, 0.4])
            writer.writerow(['doc5', 'Dupont', 'fr', 0.3, 0.7])
    
    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()
    
    # =========================================================================
    # Tests for _load_metadata() helper function
    # =========================================================================
    
    def test_load_metadata_with_no_params(self):
        """Test _load_metadata when no metadata or excludes needed"""
        # WHEN
        metadata, excludes = _load_metadata(
            self.csv_path, 
            metadata_path=None, 
            excludes_path=None, 
            lang=None
        )
        
        # THEN
        self.assertIsNone(metadata)
        self.assertIsNone(excludes)
    
    def test_load_metadata_from_main_csv_for_lang(self):
        """Test _load_metadata creating metadata from main CSV when lang specified"""
        # WHEN
        metadata, excludes = _load_metadata(
            self.csv_path, 
            metadata_path=None, 
            excludes_path=None, 
            lang='en'  # Lang specified, so metadata needed
        )
        
        # THEN
        self.assertIsNotNone(metadata)
        self.assertIsNone(excludes)
        self.assertIn('doc1', metadata.index)
        self.assertEqual(metadata.loc['doc1', 'lang'], 'en')
    
    def test_load_metadata_from_separate_file(self):
        """Test _load_metadata loading from separate metadata file"""
        # GIVEN: Create a metadata file
        metadata_path = os.path.join(self.temp_path, "metadata.csv")
        with open(metadata_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'true'])
            writer.writerow(['doc1', 'en'])
            writer.writerow(['doc2', 'fr'])
        
        # WHEN
        metadata, excludes = _load_metadata(
            self.csv_path, 
            metadata_path=metadata_path, 
            excludes_path=None, 
            lang=None
        )
        
        # THEN
        self.assertIsNotNone(metadata)
        self.assertIsNone(excludes)
        self.assertEqual(metadata.loc['doc1', 'lang'], 'en')
        self.assertEqual(metadata.loc['doc2', 'lang'], 'fr')
    
    def test_load_metadata_with_excludes(self):
        """Test _load_metadata loading excludes list"""
        # GIVEN: Create an excludes file
        excludes_path = os.path.join(self.temp_path, "excludes.csv")
        with open(excludes_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id'])
            writer.writerow(['doc3'])
            writer.writerow(['doc5'])
        
        # WHEN
        metadata, excludes = _load_metadata(
            self.csv_path, 
            metadata_path=None, 
            excludes_path=excludes_path, 
            lang=None
        )
        
        # THEN
        # When excludes_path is not None, metadata is loaded from CSV (for potential filtering)
        self.assertIsNotNone(metadata)
        self.assertIsNotNone(excludes)
        self.assertIn('doc3', excludes)
        self.assertIn('doc5', excludes)
        self.assertEqual(len(excludes), 2)
    
    # =========================================================================
    # Tests for _should_exclude() helper function
    # =========================================================================
    
    def test_should_exclude_no_exclusions(self):
        """Test _should_exclude when no exclusions apply"""
        # WHEN
        is_excluded, reason = _should_exclude(
            'doc1', 
            metadata=None, 
            excludes=None, 
            lang=None
        )
        
        # THEN
        self.assertFalse(is_excluded)
        self.assertIsNone(reason)
    
    def test_should_exclude_by_language(self):
        """Test _should_exclude excluding by language"""
        # GIVEN: Create metadata
        metadata = pd.DataFrame({
            'lang': ['en', 'fr', 'es']
        }, index=['doc1', 'doc2', 'doc3'])
        
        # WHEN: Check doc with wrong language
        is_excluded, reason = _should_exclude(
            'doc2',  # fr
            metadata=metadata, 
            excludes=None, 
            lang='en'  # Only want en
        )
        
        # THEN
        self.assertTrue(is_excluded)
        self.assertIn('not in: en', reason)
        self.assertIn('doc2', reason)
    
    def test_should_exclude_by_language_passes(self):
        """Test _should_exclude NOT excluding when language matches"""
        # GIVEN: Create metadata
        metadata = pd.DataFrame({
            'lang': ['en', 'fr', 'es']
        }, index=['doc1', 'doc2', 'doc3'])
        
        # WHEN: Check doc with correct language
        is_excluded, reason = _should_exclude(
            'doc1',  # en
            metadata=metadata, 
            excludes=None, 
            lang='en'  # Want en
        )
        
        # THEN
        self.assertFalse(is_excluded)
        self.assertIsNone(reason)
    
    def test_should_exclude_by_excludes_list(self):
        """Test _should_exclude excluding by excludes list"""
        # WHEN
        is_excluded, reason = _should_exclude(
            'doc3',
            metadata=None, 
            excludes=['doc3', 'doc5'], 
            lang=None
        )
        
        # THEN
        self.assertTrue(is_excluded)
        self.assertIn('Wilhelmus', reason)
        self.assertIn('doc3', reason)
    
    def test_should_exclude_not_in_excludes_list(self):
        """Test _should_exclude NOT excluding when not in list"""
        # WHEN
        is_excluded, reason = _should_exclude(
            'doc1',
            metadata=None, 
            excludes=['doc3', 'doc5'], 
            lang=None
        )
        
        # THEN
        self.assertFalse(is_excluded)
        self.assertIsNone(reason)
    
    def test_should_exclude_keyerror_handling(self):
        """Test _should_exclude handling missing keys gracefully"""
        # GIVEN: Metadata without doc99
        metadata = pd.DataFrame({
            'lang': ['en', 'fr']
        }, index=['doc1', 'doc2'])
        
        # WHEN: Check a doc not in metadata
        is_excluded, reason = _should_exclude(
            'doc99',  # Not in metadata
            metadata=metadata, 
            excludes=None, 
            lang='en'
        )
        
        # THEN: Should not crash, should not exclude
        self.assertFalse(is_excluded)
        self.assertIsNone(reason)
    
    # =========================================================================
    # Tests for read_clean() function
    # =========================================================================
    
    def test_read_clean_no_split(self):
        """Test read_clean without splitting"""
        # GIVEN
        output_json = os.path.join(self.temp_path, "selection.json")
        
        # WHEN
        read_clean(
            self.csv_path,
            savesplit=output_json,
            split=False
        )
        
        # THEN: Should create _selected.csv
        selected_path = self.csv_path.replace('.csv', '_selected.csv')
        self.assertTrue(os.path.exists(selected_path))
        
        # Check content
        with open(selected_path, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 6)  # Header + 5 data rows
        
        # Check selection JSON
        with open(output_json, 'r') as f:
            selection = json.load(f)
            self.assertIn('train', selection)
            self.assertIn('elim', selection)
            self.assertNotIn('valid', selection)  # No split
            self.assertEqual(len(selection['train']), 5)
            self.assertEqual(len(selection['elim']), 0)
    
    def test_read_clean_with_split(self):
        """Test read_clean with splitting into train/valid"""
        # GIVEN
        output_json = os.path.join(self.temp_path, "selection_split.json")
        
        # WHEN
        read_clean(
            self.csv_path,
            savesplit=output_json,
            split=True,
            split_ratio=0.5  # 50% for easier testing
        )
        
        # THEN: Should create _train.csv and _valid.csv
        train_path = self.csv_path.replace('.csv', '_train.csv')
        valid_path = self.csv_path.replace('.csv', '_valid.csv')
        self.assertTrue(os.path.exists(train_path))
        self.assertTrue(os.path.exists(valid_path))
        
        # Check selection JSON
        with open(output_json, 'r') as f:
            selection = json.load(f)
            self.assertIn('train', selection)
            self.assertIn('valid', selection)
            self.assertIn('elim', selection)
            # With 50% split, should have some in train and some in valid
            total = len(selection['train']) + len(selection['valid'])
            self.assertEqual(total, 5)  # All 5 docs processed
    
    def test_read_clean_with_language_filter(self):
        """Test read_clean filtering by language"""
        # GIVEN
        output_json = os.path.join(self.temp_path, "selection_lang.json")
        
        # WHEN: Only keep English docs
        read_clean(
            self.csv_path,
            savesplit=output_json,
            lang='en',
            split=False
        )
        
        # THEN
        with open(output_json, 'r') as f:
            selection = json.load(f)
            # Should keep 2 English docs (doc1, doc4)
            self.assertEqual(len(selection['train']), 2)
            # Should eliminate 3 non-English docs (doc2, doc3, doc5)
            self.assertEqual(len(selection['elim']), 3)
    
    def test_read_clean_with_excludes(self):
        """Test read_clean with excludes list"""
        # GIVEN: Create an excludes file
        excludes_path = os.path.join(self.temp_path, "excludes.csv")
        with open(excludes_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id'])
            writer.writerow(['doc2'])
            writer.writerow(['doc4'])
        
        output_json = os.path.join(self.temp_path, "selection_excl.json")
        
        # WHEN
        read_clean(
            self.csv_path,
            excludes_path=excludes_path,
            savesplit=output_json,
            split=False
        )
        
        # THEN
        with open(output_json, 'r') as f:
            selection = json.load(f)
            # Should keep 3 docs (doc1, doc3, doc5)
            self.assertEqual(len(selection['train']), 3)
            # Should eliminate 2 docs (doc2, doc4)
            self.assertEqual(len(selection['elim']), 2)
            self.assertIn('doc2', selection['elim'])
            self.assertIn('doc4', selection['elim'])
    
    def test_read_clean_with_metadata_file(self):
        """Test read_clean with separate metadata file"""
        # GIVEN: Create a metadata file
        metadata_path = os.path.join(self.temp_path, "metadata.csv")
        with open(metadata_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'true'])
            writer.writerow(['doc1', 'en'])
            writer.writerow(['doc2', 'en'])
            writer.writerow(['doc3', 'fr'])
            writer.writerow(['doc4', 'en'])
            writer.writerow(['doc5', 'fr'])
        
        output_json = os.path.join(self.temp_path, "selection_meta.json")
        
        # WHEN: Filter by English using metadata file
        read_clean(
            self.csv_path,
            metadata_path=metadata_path,
            savesplit=output_json,
            lang='en',
            split=False
        )
        
        # THEN
        with open(output_json, 'r') as f:
            selection = json.load(f)
            # Should keep 3 English docs according to metadata
            self.assertEqual(len(selection['train']), 3)
            self.assertIn('doc1', selection['train'])
            self.assertIn('doc2', selection['train'])
            self.assertIn('doc4', selection['train'])
    
    def test_read_clean_combined_filters(self):
        """Test read_clean with both language filter and excludes"""
        # GIVEN: Create an excludes file
        excludes_path = os.path.join(self.temp_path, "excludes_combined.csv")
        with open(excludes_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id'])
            writer.writerow(['doc4'])  # Exclude one English doc
        
        output_json = os.path.join(self.temp_path, "selection_combined.json")
        
        # WHEN: Filter by English AND exclude doc4
        read_clean(
            self.csv_path,
            excludes_path=excludes_path,
            savesplit=output_json,
            lang='en',
            split=False
        )
        
        # THEN
        with open(output_json, 'r') as f:
            selection = json.load(f)
            # Should keep only doc1 (doc4 excluded, others wrong lang)
            self.assertEqual(len(selection['train']), 1)
            self.assertIn('doc1', selection['train'])
            # Should eliminate 4 docs
            self.assertEqual(len(selection['elim']), 4)
    
    # =========================================================================
    # Tests for apply_selection() function
    # =========================================================================
    
    def test_apply_selection(self):
        """Test apply_selection applying a pre-existing selection"""
        # GIVEN: Create a selection JSON
        selection_path = os.path.join(self.temp_path, "presplit.json")
        selection = {
            'train': ['doc1', 'doc3'],
            'valid': ['doc2', 'doc4'],
            'elim': ['doc5']
        }
        with open(selection_path, 'w') as f:
            json.dump(selection, f)
        
        # WHEN
        apply_selection(self.csv_path, selection_path)
        
        # THEN: Should create _train.csv and _valid.csv
        train_path = self.csv_path.replace('.csv', '_train.csv')
        valid_path = self.csv_path.replace('.csv', '_valid.csv')
        self.assertTrue(os.path.exists(train_path))
        self.assertTrue(os.path.exists(valid_path))
        
        # Check train file content
        with open(train_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            train_ids = [row[0] for row in reader]
            self.assertEqual(set(train_ids), {'doc1', 'doc3'})
        
        # Check valid file content
        with open(valid_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            valid_ids = [row[0] for row in reader]
            self.assertEqual(set(valid_ids), {'doc2', 'doc4'})
    
    def test_apply_selection_eliminates_correctly(self):
        """Test apply_selection correctly eliminates specified docs"""
        # GIVEN: Create a selection JSON with eliminations
        selection_path = os.path.join(self.temp_path, "presplit_elim.json")
        selection = {
            'train': ['doc1', 'doc2'],
            'valid': ['doc3'],
            'elim': ['doc4', 'doc5']
        }
        with open(selection_path, 'w') as f:
            json.dump(selection, f)
        
        # WHEN
        apply_selection(self.csv_path, selection_path)
        
        # THEN: Eliminated docs should not appear in either file
        train_path = self.csv_path.replace('.csv', '_train.csv')
        valid_path = self.csv_path.replace('.csv', '_valid.csv')
        
        with open(train_path, 'r') as f:
            content = f.read()
            self.assertNotIn('doc4', content)
            self.assertNotIn('doc5', content)
        
        with open(valid_path, 'r') as f:
            content = f.read()
            self.assertNotIn('doc4', content)
            self.assertNotIn('doc5', content)


if __name__ == '__main__':
    unittest.main()