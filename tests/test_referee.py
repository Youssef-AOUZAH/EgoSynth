import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from referee import parse_score  # noqa: E402


class ParseScoreTests(unittest.TestCase):
    def test_bracketed_format_requested_by_the_prompt(self):
        self.assertEqual(parse_score("Fingers morph slightly.\nScore: [72]"), 72)

    def test_plain_and_markdown_variants(self):
        self.assertEqual(parse_score("Score: 85"), 85)
        self.assertEqual(parse_score("**Score: [90]**"), 90)
        self.assertEqual(parse_score("score=64/100"), 64)

    def test_last_match_wins(self):
        self.assertEqual(parse_score("Realism score: 40 in the first half.\nScore: [38]"), 38)

    def test_unreadable_returns_none_not_a_default(self):
        self.assertIsNone(parse_score("Score: [XX]"))
        self.assertIsNone(parse_score("The hands look fine."))
        self.assertIsNone(parse_score("a score of 35"))
        self.assertIsNone(parse_score(""))
        self.assertIsNone(parse_score(None))

    def test_out_of_range_is_rejected(self):
        self.assertIsNone(parse_score("Score: 101"))
        self.assertIsNone(parse_score("Score: 1000"))
        self.assertEqual(parse_score("Score: 0"), 0)
        self.assertEqual(parse_score("Score: 100"), 100)


if __name__ == "__main__":
    unittest.main()
