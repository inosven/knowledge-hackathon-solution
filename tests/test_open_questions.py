"""Open questions: About-Dupaco facts, product pages, unknown products, fee phrasing, retrieval passages.

Pure unit tests run on small synthetic inputs; the answer tests build the full knowledge base from snapshots/
once for the module (about 6 seconds).
"""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "submission"))

import kb  # noqa: E402
import scoring  # noqa: E402
import solution  # noqa: E402
from run_eval import load_snapshots  # noqa: E402

S = None


def setUpModule():
    global S
    S = solution.build()
    for snap in load_snapshots(ROOT / "snapshots"):
        S.ingest(snap)


def _is_refusal(ans):
    return (not ans.confident) or bool(scoring.REFUSAL_RE.search(ans.text))


def _page(url, text, date=None, kind="newsroom"):
    return kb.PageVer(url=url, snapshot_id="s", taken_on="2026-09-01", fetched_at="", title="", text=text, kind=kind,
                      authority=kb.AUTHORITY.get(kind, 50), content_date=date, links=[], body_hash="", path=url)


BOILER = ("Dupaco Community Credit Union is a not-for-profit, member-owned financial cooperative headquartered in {hq}. "
          "Dupaco serves residents in {c} counties throughout Iowa. Founded in 1948 by ten Dubuque Packing Company employees, "
          "membership has grown to more than {m} with assets exceeding $3.6 billion. It has over {e} employees and 22 branch office locations.")


# ---------------------------------------------------------------------------------------------- unit tests

class AboutExtraction(unittest.TestCase):
    def test_newest_release_wins_for_counts(self):
        pages = {"/a": _page("/a", BOILER.format(hq="Dubuque, Iowa", c=112, m="150,000", e=500), "2025-01-01"),
                 "/b": _page("/b", BOILER.format(hq="Dubuque, Iowa", c=118, m="160,000", e=600), "2026-03-01")}
        ab = kb.extract_about(pages)
        self.assertEqual(ab["counties"]["value"], "118")
        self.assertEqual(ab["employees"]["value"], "600")
        self.assertEqual(ab["founded"]["value"], "1948")
        self.assertEqual(ab["hq"]["value"], "Dubuque, Iowa")
        self.assertEqual(ab["members"]["url"], "/b")

    def test_growing_totals_ignore_a_recent_release_with_a_reused_older_figure(self):
        pages = {"/a": _page("/a", BOILER.format(hq="Dubuque, Iowa", c=118, m="181,000", e=600), "2026-04-23"),
                 "/b": _page("/b", BOILER.format(hq="Dubuque, Iowa", c=118, m="179,000", e=600), "2026-05-05"),
                 "/old": _page("/old", BOILER.format(hq="Dubuque, Iowa", c=118, m="190,000", e=600), "2025-06-01")}
        ab = kb.extract_about(pages)
        self.assertEqual(ab["members"]["value"], "181,000")   # the 2025 figure is outside the recent window

    def test_ceo_skips_former_title_holders(self):
        pages = {"/x": _page("/x", "Long-time President and CEO Robert Smith retired. Today Dupaco President and CEO Jane Doe said hi.", "2026-01-01")}
        self.assertEqual(kb.extract_about(pages)["ceo"]["value"], "Jane Doe")


class AboutRouting(unittest.TestCase):
    def test_fields(self):
        f = solution.about_field
        self.assertEqual(f("who is in charge of dupaco?"), "ceo")
        self.assertEqual(f("how many people does dupaco employ?"), "employees")
        self.assertEqual(f("what town is dupaco based in?"), "hq")
        self.assertEqual(f("when was the credit union established?"), "founded")
        self.assertEqual(f("how many members does the credit union have?"), "members")

    def test_branch_and_product_questions_are_not_about_questions(self):
        f = solution.about_field
        self.assertIsNone(f("where is the galena branch located?"))
        self.assertIsNone(f("how many members can be on a joint account?"))
        self.assertIsNone(f("when did dupaco start offering auto loans?"))


class TextHelpers(unittest.TestCase):
    def test_related_articles_block_does_not_swallow_the_faq(self):
        md = "# Savings\n\n" + "intro text " * 30 + "\n\n## Handpicked for You\n\n[Blog post](https://x)\n\n## Frequently Asked Questions\n\n- [How do I put money in?](#)\n\nUse payroll deductions."
        text, _ = kb.clean_markdown(md)
        self.assertNotIn("Blog post", text)
        self.assertIn("payroll deductions", text)

    def test_short_plurals_and_doubled_consonants_stem(self):
        self.assertEqual(kb.stem("fees"), kb.stem("fee"))
        self.assertEqual(kb.stem("referring"), kb.stem("refer"))
        self.assertEqual(kb.stem("referred"), kb.stem("refer"))
        self.assertEqual(kb.stem("billing"), "bill")

    def test_log_in_is_one_word(self):
        self.assertIn("login", kb.tokenize("I can't log in"))
        self.assertIn("login", kb.tokenize("Shine login help"))

    def test_sentences_carry_their_heading(self):
        ss = kb.split_sentences_h("### Free Checking\n\nThis account keeps things simple for everyone.\n\n### Other\n\nA different section sentence here.")
        self.assertEqual(ss[0][1], "Free Checking")
        self.assertEqual(ss[1][1], "Other")

    def test_fee_phrases_map_to_schedule_words(self):
        self.assertIn("domestic", solution.fee_content_words("outgoing wire within the U.S."))
        self.assertIn("domestic", solution.fee_content_words("wire to another bank in the US"))
        self.assertIn("international", solution.fee_content_words("send a wire overseas"))
        self.assertNotIn("use", solution.fee_content_words("charge me for using an ATM"))

    def test_product_name_phrase(self):
        self.assertEqual(solution.product_name_phrase("what's the rate on dupaco's bitcoin savings account?"), ["bitcoin", "savings"])
        self.assertEqual(solution.product_name_phrase("monthly fee for the platinum elite checking account"), ["platinum", "elite", "checking"])
        self.assertIsNone(solution.product_name_phrase("what is the 60-month certificate rate?"))

    def test_quote_sentences_are_detected(self):
        self.assertTrue(solution.Dupaco._QUOTE_RE.search("“We love members,” said Dupaco President Jane Doe."))
        self.assertFalse(solution.Dupaco._QUOTE_RE.search("Open a savings account with $25 or more."))


# ---------------------------------------------------------------------------------------------- answers

class AboutAnswers(unittest.TestCase):
    def test_founding_year(self):
        a = S.answer("In what year was Dupaco established?")
        self.assertIn("1948", a.text)
        self.assertTrue(any(s.url.rstrip("/").endswith("/about") for s in a.sources))

    def test_headquarters(self):
        a = S.answer("Which city is Dupaco's headquarters in?")
        self.assertIn("Dubuque", a.text)
        self.assertTrue(any(s.url.rstrip("/").endswith("/about") for s in a.sources))

    def test_ceo_is_current_not_former(self):
        a = S.answer("Who leads Dupaco Community Credit Union?")
        self.assertIn("Joe Hearn", a.text)
        self.assertNotIn("Hoefer", a.text)
        self.assertTrue(any("joe-hearn" in s.url for s in a.sources))

    def test_member_count_uses_current_boilerplate(self):
        a = S.answer("How many members does the credit union have?")
        for old in ("173,000", "171,000", "143,000"):
            self.assertNotIn(old, a.text)
        self.assertRegex(a.text, r"1[78]\d,000")

    def test_employees_and_counties(self):
        self.assertIn("600", S.answer("How many people work for Dupaco?").text)
        self.assertIn("118", S.answer("How many counties are in Dupaco's service area? How many counties does Dupaco serve?").text)


class ProductAnswers(unittest.TestCase):
    def test_named_products(self):
        v = S.kb.views[-1]
        names = lambda q: [d["name"] for d in solution.named_products(v, solution.parse_question(q))]
        self.assertEqual(names("Tell me about 1st Rate Checking"), ["1st Rate Checking"])
        self.assertEqual(names("Is there a fee on an HSA?"), ["Health Savings Accounts"])
        self.assertIn("Business 1st Rate Checking", names("What is Business 1st Rate Checking?"))
        self.assertNotIn("1st Rate Checking", names("What is Business 1st Rate Checking?"))

    def test_product_name_is_not_a_rate_question(self):
        a = S.answer("Tell me about 1st Rate Checking")
        self.assertIn("$10", a.text)
        self.assertNotIn("APY", a.text)
        self.assertTrue(any("checking" in s.url for s in a.sources))

    def test_explicit_rate_question_still_gets_rates(self):
        a = S.answer("What APY does 1st Rate Checking pay?")
        self.assertIn("APY", a.text)

    def test_free_checking_fee_comes_from_product_page(self):
        a = S.answer("Is there a monthly charge on Free Checking?")
        self.assertRegex(a.text.lower(), r"no monthly")
        self.assertNotIn("ATM", a.text)

    def test_product_eligibility_is_not_membership_eligibility(self):
        a = S.answer("Who qualifies for an HSA?")
        self.assertIn("high-deductible", a.text)
        self.assertNotIn("charter area", a.text)

    def test_description_does_not_answer_a_specific_question(self):
        a = S.answer("What is the yearly HSA contribution limit?")
        self.assertTrue(_is_refusal(a))
        self.assertFalse(re.search(r"\d", a.text), a.text)


class RefusalsAndFees(unittest.TestCase):
    def test_unknown_named_product_refuses_without_numbers(self):
        for q in ("What APY does the Dupaco Moonshot Savings account pay?", "What's the fee on Diamond Club Checking?"):
            a = S.answer(q)
            self.assertFalse(a.confident, q)
            self.assertFalse(re.search(r"\d", a.text), a.text)

    def test_known_lower_case_products_still_answer(self):
        self.assertTrue(S.answer("What is the rate for the Bright Savings account?").confident)
        self.assertTrue(S.answer("How much does a debit card rush cost?").confident)

    def test_wire_within_the_country_is_domestic(self):
        a = S.answer("How much is an outgoing wire to another bank in the United States?")
        self.assertIn("$25.00", a.text)
        self.assertNotIn("$55", a.text)

    def test_charge_me_for_another_banks_atm(self):
        a = S.answer("Will you charge me for using another bank's ATM?")
        self.assertIn("$0.50", a.text)

    def test_fee_cell_detail_keeps_its_own_price(self):
        a = S.answer("Is coin counting free?")
        self.assertIn("5%", a.text)

    def test_join_gives_opening_deposit_and_link(self):
        a = S.answer("How can I become a Dupaco member?")
        self.assertIn("$25", a.text)
        self.assertIn("join.dupaco.com", a.text)
        self.assertTrue(any(s.url.rstrip("/").endswith("/about/join") for s in a.sources))


class RetrievalPassages(unittest.TestCase):
    def test_help_page_beats_unrelated_pages_and_keeps_list_items(self):
        a = S.answer("Why can't I log in to Shine Online Banking?")
        self.assertTrue(any("shine-login-help" in s.url for s in a.sources))
        self.assertIn("ad block", a.text.lower())

    def test_faq_after_related_articles_is_found(self):
        a = S.answer("How can I add money to my HSA?")
        self.assertIn("payroll", a.text)

    def test_passage_stops_at_rate_footnotes(self):
        a = S.answer("Who can open a Kids Account?")
        self.assertNotRegex(a.text, r"effective|\d+/\d+/20\d\d")

    def test_same_sentence_on_two_pages_cites_both(self):
        a = S.answer("How do I avoid the monthly fee on 1st Rate Checking?")
        urls = {s.url.rstrip("/") for s in a.sources}
        self.assertTrue({"https://www.dupaco.com/checking", "https://www.dupaco.com/personal/checking-accounts"} <= urls, urls)


if __name__ == "__main__":
    unittest.main()
