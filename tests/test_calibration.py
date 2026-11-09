"""2026-11-09: confidence calibration of the free-text fallback, looser fee gate, word slots, step lists.

Pure unit tests run on small inputs; the answer tests build the full knowledge base from snapshots/ once for the
module. Wordings deliberately differ from dev/open_eval.jsonl and dev/blind_eval.jsonl.
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


def _refuses_cleanly(tc, ans, q):
    tc.assertTrue(_is_refusal(ans), f"{q}: {ans.text}")
    tc.assertFalse(re.search(r"\d", ans.text), f"{q}: {ans.text}")


# ----------------------------------------------------------------------------- pure units

class CorePhrase(unittest.TestCase):
    def test_runs_end_in_head_noun(self):
        self.assertIn((kb.stem("reverse"), kb.stem("mortgage")), solution.core_pairs("What rate do you offer on a reverse mortgage?"))
        self.assertIn((kb.stem("withdrawal"), kb.stem("limit")), solution.core_pairs("What's my daily ATM cash withdrawal limit?"))

    def test_verb_object_and_product_line_compounds_are_not_enforced(self):
        self.assertEqual(solution.core_pairs("How do I put money into my HSA?"), [])
        self.assertEqual(solution.core_pairs("Can I add my debit and credit cards to a wallet?"), [])

    def test_pair_in_window(self):
        toks = kb.tokenize("construction and lot loans and investment property")
        self.assertTrue(solution.pair_in(toks, kb.stem("lot"), kb.stem("loan")))
        self.assertFalse(solution.pair_in(kb.tokenize("a lot can feel out of your control with loans"), kb.stem("lot"), kb.stem("loan")))

    def test_qualifiers(self):
        q = solution.product_qualifiers("Do you do reverse mortgages for seniors?")
        self.assertEqual([x[0] for x in q], ["reverse"])
        self.assertEqual(solution.product_qualifiers("What is the APR on a used car loan?"), [])

    def test_states_amount_needs_figure_next_to_focus_word(self):
        focus = {kb.stem("penalty")}
        self.assertFalse(solution.Dupaco._states_amount("You pay a penalty to take it out early. Deposits are insured up to $250,000.", focus))
        self.assertTrue(solution.Dupaco._states_amount("The penalty is 90 days of dividends.", focus))
        self.assertFalse(solution.Dupaco._states_amount("Maximum deposit limits may apply.", {kb.stem("limit")}))


class Slots(unittest.TestCase):
    def test_amounts_in_words(self):
        self.assertEqual(solution._money_values("about three hundred thousand dollars"), [300000.0])
        self.assertEqual(solution._money_values("ten grand"), [10000.0])
        self.assertEqual(solution._money_values("roughly 250k"), [250000.0])
        self.assertEqual(solution._money_values("$2.5 million"), [2500000.0])
        self.assertEqual(solution._money_values("thousands of members"), [])

    def test_duration_words_and_suv(self):
        info = solution.parse_question("Rate for a 2022 SUV over six years?")
        self.assertEqual(info["term"], 72)
        self.assertEqual(info["model_year"], 2022)
        self.assertEqual(info["amounts"], [])
        self.assertEqual(solution.detect_family(info["ql"]), "auto")


class FeeUnits(unittest.TestCase):
    def test_box_dimensions_compare_as_pairs(self):
        q = solution.fee_content_words("yearly cost for a 3x10 safe-deposit box")
        self.assertIn("3x10", q)
        self.assertIn("3x10", solution.fee_content_words("Safe deposit box 3 X 10 X 21"))
        self.assertNotIn("3x10", solution.fee_content_words("Safe deposit box 5 X 10 X 21"))

    def test_adverbs_of_destination(self):
        self.assertIn("domestic", solution.fee_content_words("wire money domestically"))
        self.assertIn("international", solution.fee_content_words("send money internationally"))


class ListAfter(unittest.TestCase):
    def test_whole_short_items(self):
        page = "Please allow these domains:\n\n- alpha.com\n\n- beta.com\n\n- gamma.com\n\nThanks for reading."
        self.assertEqual(solution.Dupaco._list_after(page, "Please allow these domains:"), ["alpha.com", "beta.com", "gamma.com"])

    def test_cut_at_item_boundary(self):
        page = "Steps:\n" + "\n".join(f"- item number {i} " + "x" * 40 for i in range(20))
        items = solution.Dupaco._list_after(page, "Steps:", max_chars=200)
        self.assertTrue(items and all(it.startswith("item number") and it.endswith("x") for it in items))
        self.assertLessEqual(sum(len(i) for i in items), 200)


class NumberedList(unittest.TestCase):
    def test_list_under_heading(self):
        page = "### Reset steps\n\n1. Open the app.\n\n2. Tap Forgot password.\n\n3. Follow the prompts.\n\n### Other\n\n1. Not this"
        self.assertEqual(solution.Dupaco._list_under(page, "Reset steps"), ["Open the app", "Tap Forgot password", "Follow the prompts"])


# ----------------------------------------------------------------------------- answers

class Calibration(unittest.TestCase):
    def test_unknown_product_qualifier_refuses(self):
        for q in ("Does Dupaco offer reverse mortgage loans, and what's the APR?",
                  "What's the interest rate on a bridge loan?"):
            _refuses_cleanly(self, S.answer(q), q)

    def test_product_without_rate_row_refuses(self):
        for q in ("What APR do you charge for lot loans?", "What interest rate is charged on a kids savings account?"):
            _refuses_cleanly(self, S.answer(q), q)

    def test_unpublished_amounts_refuse(self):
        for q in ("What's the early withdrawal penalty on a certificate?",
                  "Is there a maximum amount I can deposit through mobile deposit?",
                  "How much cash can I withdraw at an ATM per day with my debit card?"):
            _refuses_cleanly(self, S.answer(q), q)

    def test_ordinary_rates_still_answer(self):
        a = S.answer("What is the APR on a used car loan for a 2022 model year over 60 months?")
        self.assertTrue(a.confident)
        self.assertIn("%", a.text)
        a = S.answer("What's the rate for a home equity loan?")
        self.assertTrue(a.confident)

    def test_shared_branching(self):
        a = S.answer("Can I do my banking at another credit union?")
        self.assertTrue(a.confident)
        self.assertIn("5,600", a.text)


class LooserFeeGate(unittest.TestCase):
    def test_wire_paraphrases_agree(self):
        a = S.answer("What's the fee to wire funds domestically?")
        b = S.answer("How much does a domestic wire cost?")
        self.assertIn("$25.00", a.text)
        self.assertEqual(scoring._numbers(a.text), scoring._numbers(b.text))
        c = S.answer("My son is overseas. How much is it to send him an international wire?")
        self.assertIn("$55.00", c.text)

    def test_coin_counter_for_non_members(self):
        a = S.answer("Do non-members pay to use the coin counter?")
        self.assertIn("5%", a.text)

    def test_cash_advance_fee_not_apr(self):
        a = S.answer("How much does it cost to take a cash advance on a Dupaco credit card?")
        self.assertIn("$10.00", a.text)
        self.assertIn("3.00%", a.text)
        self.assertNotIn("APR", a.text)

    def test_hsa_fee_and_waiver(self):
        a = S.answer("What's the HSA monthly service fee and how do I get it waived?")
        self.assertIn("$2.00", a.text)
        self.assertIn("$2,500", a.text)

    def test_box_size(self):
        a = S.answer("How much per year is a 5x10 safe deposit box?")
        self.assertIn("$60.00", a.text)
        self.assertNotIn("$40.00", a.text)

    def test_expedited_is_rush_and_not_an_unknown_product(self):
        a = S.answer("What's the fee for an expedited debit card?")
        self.assertTrue(a.confident)
        self.assertIn("$50.00", a.text)

    def test_personal_card_fee_before_business(self):
        a = S.answer("What is the late payment fee on a credit card?")
        self.assertNotIn("Processing fee", a.text)
        self.assertNotIn("$35", a.text)


class SlotAnswers(unittest.TestCase):
    def test_amount_in_words_picks_tier(self):
        a = S.answer("What APY does High-Yield Savings pay on two hundred fifty thousand dollars?")
        self.assertIn("2.05%", a.text)
        self.assertNotIn("1.85%", a.text)

    def test_unknown_balance_gives_one_tier(self):
        a = S.answer("What is the High-Yield Savings APY?")
        self.assertEqual(len(re.findall(r"% APY", a.text)), 1, a.text)
        a = S.answer("What are all the High-Yield Savings rate tiers?")
        self.assertGreater(len(re.findall(r"% APY", a.text)), 1, a.text)

    def test_word_duration_and_suv(self):
        a = S.answer("What APR for a 2022 pickup financed over six years?")
        self.assertIn("5.79%", a.text)


class Passages(unittest.TestCase):
    def test_step_answer_for_mobile_deposit(self):
        a = S.answer("How can I deposit a check using my smartphone?")
        self.assertIn("endorse", a.text.lower())

    def test_short_list_items_are_kept(self):
        a = S.answer("Which domains do I need to allow for Shine login if I block ads?")
        for d in ("iesnare.com", "iovation.com"):
            self.assertIn(d, a.text)


class BranchesAndAbout(unittest.TestCase):
    def test_compass_side_picks_branch(self):
        a = S.answer("Which branch is on the east side of Madison, and what's its address?")
        self.assertIn("3762 E. Washington", a.text)
        self.assertNotIn("7701", a.text)

    def test_where_exactly(self):
        a = S.answer("Where exactly is your Grimes branch?")
        self.assertTrue(a.confident)
        self.assertIn("Grimes", a.text)

    def test_branch_cites_newsroom_story_with_its_address(self):
        a = S.answer("Where is the Waukee learning lab?")
        self.assertTrue(any("/newsroom/" in s.url for s in a.sources))

    def test_counties_and_founding_routes(self):
        a = S.answer("How many counties are in Dupaco's service area?")
        self.assertIn("118", a.text)
        self.assertTrue(any(s.url.rstrip("/").endswith("/about") for s in a.sources))
        a = S.answer("Who started Dupaco, and when was it started?")
        self.assertIn("1948", a.text)


if __name__ == "__main__":
    unittest.main()
