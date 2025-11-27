import unittest
import superstyl.preproc.pipe
import os
import glob

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class XMLLoadingTests(unittest.TestCase):
    """Tests for XML, TEI, and TXM file loading functions"""
    
    def setUp(self):
        """Set up test files paths"""
        self.xml_path = os.path.join(THIS_DIR, "testdata", "Smith_Song1.xml")
        self.tei_path = os.path.join(THIS_DIR, "testdata", "Dupont_TEIPoem1.xml")
        self.txm_path = os.path.join(THIS_DIR, "testdata", "Smith_TXM1.xml")
    
    def test_XML_to_text(self):
        # SCENARIO: Load text from a simple XML file
        # GIVEN: An XML file with author and text elements
        
        # WHEN: Loading the XML file
        aut, text = superstyl.preproc.pipe.XML_to_text(self.xml_path)
        
        # THEN: Author and text are correctly extracted
        self.assertEqual(aut, "Smith")
        self.assertIn("test song", text)
        self.assertIn("lyrics", text)
        # Check that whitespace is normalized
        self.assertNotIn("  ", text)
    
    def test_tei_to_units_words(self):
        # SCENARIO: Extract words from a TEI file
        # GIVEN: A TEI file with annotated words
        
        # WHEN: Extracting words as units
        units_tokens = superstyl.preproc.pipe.tei_to_units(
            self.tei_path, 
            feats="words", 
            units="words"
        )
        
        # THEN: Words are extracted, one per line
        self.assertIsInstance(units_tokens, list)
        self.assertGreater(len(units_tokens), 0)
        # Each word should be on a separate line
        self.assertIn("This", [u.strip() for u in units_tokens])
        self.assertIn("is", [u.strip() for u in units_tokens])
    
    def test_tei_to_units_verses(self):
        # SCENARIO: Extract verses (lines) from a TEI file
        # GIVEN: A TEI file with verse lines
        
        # WHEN: Extracting verses as units
        units_tokens = superstyl.preproc.pipe.tei_to_units(
            self.tei_path, 
            feats="words", 
            units="verses"
        )
        
        # THEN: Each verse is on a separate line
        self.assertIsInstance(units_tokens, list)
        # We should have 2 lines in our test file
        self.assertEqual(len(units_tokens), 2)
    
    def test_tei_to_units_lemma(self):
        # SCENARIO: Extract lemmas from a TEI file
        # GIVEN: A TEI file with lemma annotations
        
        # WHEN: Extracting lemmas
        units_tokens = superstyl.preproc.pipe.tei_to_units(
            self.tei_path, 
            feats="lemma", 
            units="words"
        )
        
        # THEN: Lemmas are extracted
        self.assertIsInstance(units_tokens, list)
        self.assertIn("this", [u.strip() for u in units_tokens])
        self.assertIn("be", [u.strip() for u in units_tokens])
    
    def test_tei_to_units_pos(self):
        # SCENARIO: Extract POS tags from a TEI file
        # GIVEN: A TEI file with POS annotations
        
        # WHEN: Extracting POS tags
        units_tokens = superstyl.preproc.pipe.tei_to_units(
            self.tei_path, 
            feats="pos", 
            units="words"
        )
        
        # THEN: POS tags are extracted
        self.assertIsInstance(units_tokens, list)
        self.assertIn("DET", [u.strip() for u in units_tokens])
        self.assertIn("VERB", [u.strip() for u in units_tokens])
    
    def test_tei_to_units_met_syll(self):
        # SCENARIO: Extract metrical syllables from a TEI file
        # GIVEN: A TEI file with metrical annotations
        
        # WHEN: Extracting metrical syllables with met_syll feature
        units_tokens = superstyl.preproc.pipe.tei_to_units(
            self.tei_path, 
            feats="met_syll", 
            units="verses"
        )
        
        # THEN: Metrical annotations are extracted
        self.assertIsInstance(units_tokens, list)
        # The @met attributes should be present
        self.assertGreater(len(units_tokens), 0)
    
    def test_tei_to_units_met_line(self):
        # SCENARIO: Extract metrical lines from a TEI file
        # GIVEN: A TEI file with metrical annotations on lines
        
        # WHEN: Extracting metrical patterns at line level
        units_tokens = superstyl.preproc.pipe.tei_to_units(
            self.tei_path, 
            feats="met_line", 
            units="verses"
        )
        
        # THEN: Metrical patterns for each line are extracted
        self.assertIsInstance(units_tokens, list)
        self.assertEqual(len(units_tokens), 2)
        # Should contain the metrical patterns
        self.assertIn("01010101", units_tokens[0])
        self.assertIn("10101010", units_tokens[1])
    
    def test_txm_to_units_words(self):
        # SCENARIO: Extract words from a TXM file
        # GIVEN: A TXM file with annotated words
        
        # WHEN: Extracting words as units
        units_tokens = superstyl.preproc.pipe.txm_to_units(
            self.txm_path, 
            units="words"
        )
        
        # THEN: Words are extracted
        # Note: When extracting individual words (units='words'), 
        # the NOMpro filter is not applied
        self.assertIsInstance(units_tokens, list)
        self.assertGreater(len(units_tokens), 0)
        text_content = ' '.join(units_tokens)
        # All words should be present including those with NOMpro
        self.assertIn("This", text_content)
        self.assertIn("test", text_content)
    
    def test_txm_to_units_verses(self):
        # SCENARIO: Extract verses from a TXM file
        # GIVEN: A TXM file with verse lines
        
        # WHEN: Extracting verses as units
        units_tokens = superstyl.preproc.pipe.txm_to_units(
            self.txm_path, 
            units="verses"
        )
        
        # THEN: Each verse is extracted and NOMpro words are filtered out
        self.assertIsInstance(units_tokens, list)
        self.assertEqual(len(units_tokens), 2)
        # Check that NOMpro words are excluded in verse mode
        text_content = ' '.join(units_tokens)
        self.assertNotIn("here", text_content)  # "here" has NOMpro tag and should be filtered
        self.assertIn("This", text_content)  # Regular words should be present
    
    def test_txm_to_units_lemma(self):
        # SCENARIO: Extract lemmas from a TXM file
        # GIVEN: A TXM file with lemma annotations
        
        # WHEN: Extracting lemmas
        units_tokens = superstyl.preproc.pipe.txm_to_units(
            self.txm_path,
            units="words",
            feats="lemma"
        )
        
        # THEN: Lemmas are extracted
        self.assertIsInstance(units_tokens, list)
        self.assertIn("be", [u.strip() for u in units_tokens])  # lemma of "is"
        self.assertIn("this", [u.strip() for u in units_tokens])
    
    def test_txm_to_units_pos(self):
        # SCENARIO: Extract POS tags from a TXM file
        # GIVEN: A TXM file with POS annotations
        
        # WHEN: Extracting POS tags
        units_tokens = superstyl.preproc.pipe.txm_to_units(
            self.txm_path,
            units="words",
            feats="pos"
        )
        
        # THEN: POS tags are extracted
        self.assertIsInstance(units_tokens, list)
        self.assertIn("DET", [u.strip() for u in units_tokens])
        self.assertIn("VERB", [u.strip() for u in units_tokens])
    
    def test_specialXML_to_text_tei(self):
        # SCENARIO: Load text from a TEI file using specialXML_to_text
        # GIVEN: A TEI format file
        
        # WHEN: Loading with format="tei"
        aut, text = superstyl.preproc.pipe.specialXML_to_text(
            self.tei_path, 
            format="tei", 
            feats="words"
        )
        
        # THEN: Author is extracted from filename and text is normalized
        self.assertEqual(aut, "Dupont")
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        # Check that whitespace is normalized (single spaces)
        self.assertNotIn("  ", text)
    
    def test_specialXML_to_text_txm(self):
        # SCENARIO: Load text from a TXM file using specialXML_to_text
        # GIVEN: A TXM format file
        
        # WHEN: Loading with format="txm"
        aut, text = superstyl.preproc.pipe.specialXML_to_text(
            self.txm_path, 
            format="txm", 
            feats="words"
        )
        
        # THEN: Author is extracted from filename and text is normalized
        self.assertEqual(aut, "Smith")
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        # Text should contain words from the TXM file
        self.assertIn("test", text.lower())
    
    def test_specialXML_to_text_with_lemma(self):
        # SCENARIO: Load lemmas from a TEI file
        # GIVEN: A TEI file with lemma annotations
        
        # WHEN: Loading with feats="lemma"
        aut, text = superstyl.preproc.pipe.specialXML_to_text(
            self.tei_path, 
            format="tei", 
            feats="lemma"
        )
        
        # THEN: Lemmas are in the text
        self.assertEqual(aut, "Dupont")
        self.assertIn("be", text)  # lemma of "is"
        self.assertIn("this", text)
    
    def test_specialXML_to_text_with_pos(self):
        # SCENARIO: Load POS tags from a TEI file
        # GIVEN: A TEI file with POS annotations
        
        # WHEN: Loading with feats="pos"
        aut, text = superstyl.preproc.pipe.specialXML_to_text(
            self.tei_path, 
            format="tei", 
            feats="pos"
        )
        
        # THEN: POS tags are in the text
        self.assertEqual(aut, "Dupont")
        self.assertIn("DET", text)
        self.assertIn("VERB", text)
    
    def test_specialXML_to_text_txm_with_lemma(self):
        # SCENARIO: Load lemmas from a TXM file
        # GIVEN: A TXM file with lemma annotations
        
        # WHEN: Loading with feats="lemma"
        aut, text = superstyl.preproc.pipe.specialXML_to_text(
            self.txm_path,
            format="txm",
            feats="lemma"
        )
        
        # THEN: Lemmas are in the text
        self.assertEqual(aut, "Smith")
        self.assertIn("be", text)  # lemma of "is"
        self.assertIn("this", text)
    
    def test_specialXML_to_text_txm_with_pos(self):
        # SCENARIO: Load POS tags from a TXM file
        # GIVEN: A TXM file with POS annotations
        
        # WHEN: Loading with feats="pos"
        aut, text = superstyl.preproc.pipe.specialXML_to_text(
            self.txm_path,
            format="txm",
            feats="pos"
        )
        
        # THEN: POS tags are in the text
        self.assertEqual(aut, "Smith")
        self.assertIn("DET", text)
        self.assertIn("VERB", text)


if __name__ == '__main__':
    unittest.main()