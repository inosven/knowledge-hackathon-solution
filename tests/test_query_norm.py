"""Query normalisation: members phrase things differently from the website."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "submission"))

import kb  # noqa: E402
import solution  # noqa: E402


class TokenizeNormalisation(unittest.TestCase):
    def test_number_word_before_unit_becomes_digits(self):
        toks = kb.tokenize("sixty-month certificate")
        self.assertIn("60", toks)
        self.assertIn("month", toks)
        self.assertNotIn("sixty", toks)

    def test_five_year_becomes_5_year(self):
        toks = kb.tokenize("a five year mortgage")
        self.assertIn("5", toks)
        self.assertIn("year", toks)

    def test_cd_is_alias_of_certificate(self):
        cert = kb.tokenize("certificates")[0]
        self.assertEqual(kb.tokenize("CDs")[0], cert)
        self.assertEqual(kb.tokenize("a CD")[0], cert)

    def test_ratio_like_7_1_stays_one_token(self):
        self.assertIn("7x1", kb.tokenize("7/1 ARM"))
        self.assertIn("7x1", kb.tokenize("7/1 adjustable"))

    def test_dates_are_not_rewritten_as_ratios(self):
        self.assertEqual(kb.normalize_text("effective 08/31/2026"), "effective 08/31/2026")
        self.assertEqual(kb.normalize_text("as of 8/31/26"), "as of 8/31/26")
        self.assertEqual(kb.normalize_text("7/1 arm, effective 08/31/2026"), "7x1 arm, effective 08/31/2026")
        self.assertIn("10x1", kb.tokenize("a 10/1 ARM"))
        self.assertIn("5x5", kb.tokenize("5/5 ARM"))

    def test_number_words_have_one_table(self):
        self.assertIs(solution.WORD_NUM, kb.NUM_WORDS)
        self.assertEqual(solution.parse_question("a sixty-month certificate")["term"], 60)
        self.assertEqual(solution.parse_question("five year mortgage")["term"], 60)
        self.assertEqual(solution.parse_question("thirty-six month auto loan")["term"], 36)

    def test_plain_words_unchanged(self):
        self.assertEqual(kb.tokenize("Share Savings"), ["share", "saving"])


class QueryExpansion(unittest.TestCase):
    def test_car_expands_to_auto_at_lower_weight(self):
        w = solution.expand_query(["car", "loan"])
        self.assertEqual(w["car"], 1.0)
        self.assertEqual(w["loan"], 1.0)
        self.assertIn("auto", w)
        self.assertLess(w["auto"], 1.0)

    def test_expansion_never_lowers_an_original_term(self):
        # "auto" is both an original term and a synonym of "car"
        w = solution.expand_query(["car", "auto"])
        self.assertEqual(w["auto"], 1.0)


    def test_synonym_table_keys_and_values_are_stemmed(self):
        for key, extras in solution.SYNONYM_TABLE.items():
            self.assertEqual(key, kb.stem(key), key)
            for e in extras:
                self.assertEqual(e, kb.stem(e), (key, e))

    def test_every_readable_synonym_entry_is_reachable(self):
        for key in solution.SYNONYMS:
            self.assertIn(kb.stem(key), solution.SYNONYM_TABLE, key)

    def test_stemmed_keys_expand(self):
        # "hours", "vehicle", "mortgage", "charge" all stem; they used to miss the table
        w = solution.expand_query(kb.tokenize("lobby hours"))
        self.assertIn("open", w)
        self.assertEqual(w["lobby"], 1.0)
        self.assertLess(w["open"], 1.0)
        self.assertIn(kb.stem("auto"), solution.expand_query(kb.tokenize("vehicle loan")))
        self.assertIn("home", solution.expand_query(kb.tokenize("mortgage")))
        self.assertIn("fee", solution.expand_query(kb.tokenize("charge")))

    def test_open_does_not_expand_to_hours(self):
        w = solution.expand_query(kb.tokenize("How do I open a checking account"))
        self.assertNotIn(kb.stem("hours"), w)


class WeightedSearch(unittest.TestCase):
    def test_weights_scale_term_scores(self):
        idx = kb.BM25([["car", "loan"], ["auto", "loan"], ["boat"]])
        hits = dict(idx.search(["car", "auto"], weights={"car": 1.0, "auto": 0.5}))
        self.assertIn(0, hits)
        self.assertIn(1, hits)
        self.assertGreater(hits[0], hits[1])

    def test_search_without_weights_still_works(self):
        idx = kb.BM25([["car", "loan"], ["boat"]])
        self.assertEqual(idx.search(["car"])[0][0], 0)


if __name__ == "__main__":
    unittest.main()
