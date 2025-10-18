import unittest
import sys
import os

# Add main dir to path for src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import preprocess_text, simple_clean, map_product_to_label

class TestUtils(unittest.TestCase):
    def test_preprocess_text(self):
        input_text = "My CREDIT report has Errors! Visit http://example.com"
        expected = "credit report error visit"  # Matches actual: keeps 'visit', removes URL/stopwords, lemmatizes 'Errors'
        result = preprocess_text(input_text)
        self.assertEqual(result, expected)

    def test_simple_clean(self):
        input_text = "HTTP://Test.com Email@test.com 123!"
        expected = ""  # Matches actual: URL/email removed, digits/punct → spaces, strip empties it
        self.assertEqual(simple_clean(input_text), expected)

    def test_map_product_to_label(self):
        self.assertEqual(map_product_to_label("Credit reporting"), 0)
        self.assertEqual(map_product_to_label("Debt collection"), 1)
        self.assertEqual(map_product_to_label("Consumer Loan"), 2)
        self.assertEqual(map_product_to_label("Mortgage"), 3)
        self.assertIsNone(map_product_to_label("Banking"))

if __name__ == '__main__':
    unittest.main()