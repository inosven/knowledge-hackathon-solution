"""Fee questions: answer only with a fee whose label names what was asked; otherwise refuse.

Builds the full knowledge base from snapshots/ once for the module (about 6 seconds).
"""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "submission"))

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


class FeeGateUnit(unittest.TestCase):
    def test_generic_words_do_not_count(self):
        self.assertEqual(solution.fee_content_words("How much does Dupaco charge for a cashier's check?"),
                         {"cashier", "check"})

    def test_alias_lets_cashier_match_official(self):
        q = solution.fee_content_words("cashier's check fee")
        self.assertEqual(solution.fee_label_cover(q, "Official check"), frozenset({"cashier", "check"}))

    def test_unrelated_label_has_no_cover(self):
        q = solution.fee_content_words("What does the overdraft protection cost?")
        self.assertFalse(solution.fee_label_cover(q, "Money orders"))


class FeeAnswers(unittest.TestCase):
    def ask(self, q):
        return S.answer(q)

    # ---- previously answered confidently with the wrong fee
    def test_overdraft_protection_is_not_identity_theft_fee(self):
        a = self.ask("What does the overdraft protection cost?")
        self.assertNotIn("1.95", a.text)
        self.assertNotIn("Identity Theft", a.text)
        if _is_refusal(a):
            self.assertFalse(re.search(r"\d", a.text), a.text)
        else:
            # the checking page says Overdraft Protection transfers "at no extra cost"
            self.assertRegex(a.text, r"(?i)overdraft protection.*no extra cost")
            self.assertNotRegex(a.text, r"\$\s?\d")

    def test_cashiers_check_is_official_check_fee(self):
        a = self.ask("How much does Dupaco charge for a cashier's check?")
        if _is_refusal(a):
            self.assertFalse(re.search(r"\d", a.text), a.text)
        else:
            self.assertIn("Official check", a.text)
            self.assertIn("$5.00", a.text)
            self.assertNotIn("wiring", a.text)

    def test_credit_card_fee_never_uses_deposit_schedule(self):
        a = self.ask("What is the late payment fee on a credit card?")
        self.assertNotIn("Processing fee", a.text)

    def test_unlisted_fee_refuses_without_numbers(self):
        a = self.ask("How much is the fee for a bounced check?")
        self.assertTrue(_is_refusal(a))
        self.assertFalse(re.search(r"\d", a.text), a.text)

    # ---- must still answer correctly (values from the Feb 2025 fee schedule PDF)
    def test_domestic_wire(self):
        a = self.ask("What is the fee for an outgoing domestic wire transfer?")
        self.assertTrue(a.confident)
        self.assertIn("$25.00", a.text)
        self.assertIn("Domestic wire", a.text)

    def test_international_wire(self):
        a = self.ask("What does an international wire cost?")
        self.assertTrue(a.confident)
        self.assertIn("$55.00", a.text)

    def test_non_dupaco_atm(self):
        a = self.ask("How much does it cost to use a non-Dupaco ATM?")
        self.assertTrue(a.confident)
        self.assertIn("$0.50", a.text)
        self.assertIn("Non-Dupaco ATM", a.text)

    def test_overdraft_fee(self):
        a = self.ask("What is the overdraft fee?")
        self.assertTrue(a.confident)
        self.assertIn("$30.00", a.text)
        self.assertIn("overdraft", a.text.lower())

    def test_money_order(self):
        a = self.ask("How much is a money order?")
        self.assertTrue(a.confident)
        self.assertIn("$5.00", a.text)

    def test_foreign_transaction_fee_on_debit_card(self):
        a = self.ask("What is the foreign transaction fee on a Dupaco debit card?")
        self.assertTrue(a.confident)
        self.assertIn("Foreign transaction fee", a.text)
        self.assertIn("1%", a.text)


if __name__ == "__main__":
    unittest.main()
