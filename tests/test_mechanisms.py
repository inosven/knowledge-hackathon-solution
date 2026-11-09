"""2026-11-09 (round 2): five cross-cutting mechanisms.

1. recency of dated articles that state the same kind of figure, 2. route triggers that need real context
("hours"), 3. the asked-for value type (phone, date, amount, person) in the passage, 4. calibrated refusals
(how-to/service questions that are answerable; amounts above a published limit; holiday hours), 5. tier/variant
slips (negated qualifiers, box sizes, product of a quoted term, one named branch).

Wordings deliberately differ from every dev/blind eval set. Pure unit tests run on small inputs; the answer tests
build the full knowledge base from snapshots/ once for the module.
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
D = solution.Dupaco


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


def _has(text, value):
    return scoring.contains(text, value)


# ----------------------------------------------------------------------------- 1. recency

class Recency(unittest.TestCase):
    def test_figure_kinds(self):
        self.assertIn("big$", D._fig_kinds("helped prevent nearly $7 million in potential fraud losses"))
        self.assertIn("pct", D._fig_kinds("a low 8.95% APR"))
        self.assertIn("count", D._fig_kinds("contributed 6,136 volunteer hours"))
        self.assertIn("date", D._fig_kinds("The deadline is Wednesday, February 25, 2026."))
        self.assertEqual(D._fig_kinds("Dupaco is a member-owned cooperative."), set())

    def test_newer_running_total_wins(self):
        q = "Since 2016, how much Thank Use has Dupaco returned to members in total?"
        a = S.answer(q)
        self.assertTrue(_has(a.text, "$29 million"), a.text)
        self.assertFalse(_has(a.text, "$27.7 million"), a.text)

    def test_newer_year_total_beats_so_far_figure(self):
        q = "How much potential fraud loss did Dupaco's fraud team stop in 2025?"
        a = S.answer(q)
        self.assertTrue(_has(a.text, "$10.5 million"), a.text)
        self.assertFalse(_has(a.text, "$7 million"), a.text)


# ----------------------------------------------------------------------------- 2. route triggers

class RouteTriggers(unittest.TestCase):
    def test_hours_needs_opening_context(self):
        for q in ("What time does the Waukee lobby close on Saturday?", "what are your hours on saturday",
                  "When is the drive-up open until?", "Lobby hours at the Platteville branch?"):
            self.assertTrue(D._hours_q(q.lower()), q)
        for q in ("How many volunteer hours did staff log last year?", "My work hours were reduced, can I defer a loan payment?",
                  "Is the ATM available 24 hours a day?", "My hours got cut, what are my options for my car loan?"):
            self.assertFalse(D._hours_q(q.lower()), q)

    def test_holiday_questions(self):
        for q in ("Are branches closed on Christmas Eve?", "Will Dupaco be open on Memorial Day?",
                  "do branches close early the day after thanksgiving"):
            self.assertTrue(D._holiday_hours_q(q.lower()), q)
        self.assertFalse(D._holiday_hours_q("how does the holiday club account work?"))

    def test_other_hours_do_not_get_branch_hours(self):
        a = S.answer("How many hours did Dupaco employees volunteer in 2025?")
        self.assertNotIn("Lobby", a.text)
        a = S.answer("My shifts and hours got reduced; does Dupaco offer loan payment deferral?")
        self.assertNotIn("Lobby", a.text)

    def test_holiday_hours_not_published_refuse(self):
        for q in ("Are Dupaco branches open on New Year's Day?", "Is Dupaco closed on Thanksgiving Day?"):
            _refuses_cleanly(self, S.answer(q), q)

    def test_branch_phone_by_city_without_branch_word(self):
        a = S.answer("what's the phone number for dupaco in cedar falls")
        self.assertTrue(_has(a.text, "277-3940"), a.text)


# ----------------------------------------------------------------------------- 3. answer types

class AnswerTypes(unittest.TestCase):
    def test_detection(self):
        self.assertEqual(solution.answer_types("who should i call about a homeowners insurance quote?"), ["phone"])
        self.assertEqual(solution.answer_types("when does the grant application close, and how much is it worth?"), ["date", "amount"])
        self.assertEqual(solution.answer_types("who is serving as treasurer of the board?"), ["person"])
        self.assertEqual(solution.answer_types("when i travel can i use other credit unions?"), [])
        self.assertEqual(solution.person_role("who became treasurer after the meeting?"), {kb.stem("treasurer")})

    def test_clauses_keep_abbreviations(self):
        self.assertIn("Application period: Oct. 1-Nov. 12, 2026", D._clauses("Application period: Oct. 1-Nov. 12, 2026"))
        self.assertEqual(len(D._clauses("Call 800-373-7600, ext. 210. Or email us.")), 2)

    def test_explicit_dates(self):
        self.assertEqual(D._explicit_dates("begins Jan. 5, 2027 and the deadline was February 25, 2026"), ["2027-01-05", "2026-02-25"])

    def test_phone_completion_on_same_page(self):
        a = S.answer("What number do I call to get an auto insurance quote from Dupaco?")
        self.assertTrue(_has(a.text, "210"), a.text)

    def test_amount_sentence_preferred(self):
        a = S.answer("What's the biggest hardship grant the Dupaco Foundation offers?")
        self.assertTrue(_has(a.text, "$3,500"), a.text)

    def test_payment_example_not_rate_table(self):
        a = S.answer("What's the monthly payment on a 10,000 dollar home equity loan over 60 months?")
        self.assertTrue(_has(a.text, "$196.81"), a.text)

    def test_reads_on_to_the_figure(self):
        a = S.answer("Is there a credit-building loan and what APR does it carry?")
        self.assertTrue(_has(a.text, "8.95%"), a.text)

    def test_date_from_heading(self):
        a = S.answer("By what date must a nonprofit apply for the Gift of Innovation?")
        self.assertTrue(_has(a.text, "Nov. 12"), a.text)


# ----------------------------------------------------------------------------- 4. calibrated refusals

class Calibration(unittest.TestCase):
    def test_unwrap_pdf_lines(self):
        t = kb.unwrap_lines("Go to My Settings,\nthen Card Security.\nSelect Submit\na Travel Notice.\n\n# 2\nRush fee $25.00\nOther fee $5.00"
                            "\n\nJane Doe\nChief Financial Officer")
        self.assertIn("Go to My Settings, then Card Security.\nSelect Submit a Travel Notice.", t)
        self.assertIn("\nRush fee $25.00\nOther fee $5.00", t)
        self.assertIn("Jane Doe\nChief Financial Officer", t)      # title-case list lines are not prose

    def test_fee_on_named_products_page(self):
        a = S.answer("What's the late payment fee on a Dupaco Rewards Visa?")
        self.assertTrue(_has(a.text, "$15"), a.text)

    def test_travel_notice(self):
        for q in ("How do I set up a travel notice in Shine?",
                  "I'm traveling overseas next week, how do I keep my debit card from being declined?"):
            a = S.answer(q)
            self.assertTrue(_has(a.text, "Card Security"), f"{q}: {a.text}")

    def test_amount_above_published_limit_refuses(self):
        for q in ("What rate would I get on a $2 million mortgage?", "Mortgage rate for a 900k home loan, 30 year fixed?"):
            _refuses_cleanly(self, S.answer(q), q)

    def test_amount_within_limit_still_answers(self):
        a = S.answer("What's the 30-year fixed mortgage rate for a $250,000 loan?")
        self.assertTrue(a.confident, a.text)
        self.assertIn("%", a.text)

    def test_unpublished_amount_still_refuses(self):
        q = "Is there a cap on how much I can deposit with mobile deposit?"
        _refuses_cleanly(self, S.answer(q), q)


# ----------------------------------------------------------------------------- 5. tier / variant slips

class Variants(unittest.TestCase):
    def test_negated_qualifiers(self):
        i = solution.parse_question("What's the 17-month certificate rate, not the jumbo one?")
        self.assertFalse(i["jumbo"])
        self.assertIn("jumbo", i["negated"])
        i = solution.parse_question("Non-business savings rate please")
        self.assertFalse(i["business"])

    def test_not_jumbo_answers_regular(self):
        a = S.answer("What's the 17-month certificate special APY, regular not jumbo?")
        self.assertNotIn("Jumbo", a.text)
        self.assertIn("17 Month", a.text)

    def test_box_size_must_match(self):
        a = S.answer("How much does a 5x10 safe deposit box cost?")
        self.assertTrue(_has(a.text, "$60"), a.text)
        for wrong in ("$35", "$40", "$25"):
            self.assertFalse(_has(a.text, wrong), a.text)

    def test_recreational_vocabulary_and_model_year(self):
        i = solution.parse_question("rate for a 2020 four-wheeler over 48 months")
        self.assertEqual(solution.detect_family(i["ql"]), "atv")
        self.assertEqual(i["model_year"], 2020)
        a = S.answer("What's the APR on a 48-month loan for a 2020 snowmobile?")
        self.assertTrue(_has(a.text, "8.89%"), a.text)

    def test_term_is_not_a_product_modifier(self):
        self.assertEqual(solution.core_pairs("a 48-month loan for a boat"), [])

    def test_named_street_narrows_city(self):
        a = S.answer("What's the address of the Dubuque branch on Hillcrest?")
        self.assertTrue(_has(a.text, "3299 Hillcrest"), a.text)
        for other in ("1465 Sycamore", "2245 Flint Hill", "3999 Pennsylvania"):
            self.assertFalse(_has(a.text, other), a.text)


if __name__ == "__main__":
    unittest.main()
