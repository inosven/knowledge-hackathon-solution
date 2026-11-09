"""Branch table (kb.extract_branches) and the address / per-branch hours answers built on it.

Builds the full knowledge base from snapshots/ once for the module.
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


def _refuses_cleanly(test, ans, question):
    test.assertFalse(ans.confident, ans.text)
    extra = scoring._numbers(ans.text) - scoring._numbers(question)
    test.assertEqual(extra, set(), ans.text)


class BranchTable(unittest.TestCase):
    def test_table_covers_non_dubuque_branches(self):
        rows = S.kb.views[-1].branches
        cities = {r["city"] for r in rows}
        for c in ("Dyersville", "Manchester", "Cedar Rapids", "Waterloo", "Platteville", "Madison", "Galena",
                  "Carroll", "Peosta", "Waukee", "Grimes", "Cedar Falls", "Coralville", "Marion"):
            self.assertIn(c, cities)
        self.assertGreaterEqual(len(rows), 20)

    def test_every_snapshot_has_a_table(self):
        for v in S.kb.views:
            self.assertGreaterEqual(len(v.branches), 20, v.snapshot_id)

    def test_contact_zip_and_newsroom_hours_bound(self):
        rows = {r["name"]: r for r in S.kb.views[-1].branches}
        self.assertEqual(rows["Galena"]["zip"], "61036")
        self.assertIn("12:30", rows["Waukee"]["lobby"])
        self.assertIn("ATM only", rows["Platteville"]["drive_up"])
        self.assertEqual(rows["Dyersville"]["lobby"], "")


class BranchAddress(unittest.TestCase):
    def test_dyersville(self):
        a = S.answer("Where is the Dyersville branch?")
        self.assertTrue(a.confident)
        self.assertIn("807 Ninth St. SE", a.text)
        self.assertIn("Dyersville, IA", a.text)

    def test_other_non_dubuque_branches(self):
        cases = [
            ("What is the address of the Galena Branch?", ["11375 Oldenburg", "IL 61036"]),
            ("Where is Dupaco's branch in Waukee?", ["1090 E. Hickman Road", "Waukee, IA"]),
            ("Where is the Platteville branch located?", ["1100 E. Business Highway 151", "Platteville, WI"]),
            ("Where is the Mineral Point Road branch?", ["7701 Mineral Point Road", "Madison, WI"]),
            ("What's the address of the Manchester branch?", ["W. Main St.", "Manchester, IA"]),
        ]
        for q, wants in cases:
            a = S.answer(q)
            self.assertTrue(a.confident, q)
            for w in wants:
                self.assertIn(w, a.text, q)

    def test_city_with_several_branches_lists_each_when_asked_for_all(self):
        a = S.answer("Where are the Madison branches?")
        self.assertIn("7701 Mineral Point Road", a.text)
        self.assertIn("3762 E. Washington Ave.", a.text)

    def test_one_branch_asked_gives_one_address(self):
        # 2026-11-09: another branch's street address is a wrong answer for this member; name it without the number
        a = S.answer("Where is the Madison branch?")
        self.assertTrue(a.confident)
        nums = [n for n in ("7701 Mineral Point", "3762 E. Washington") if n in a.text]
        self.assertEqual(len(nums), 1, a.text)

    def test_unknown_city_refuses_without_digits(self):
        for q in ("Where is the Omaha branch?", "Is there a Dupaco branch in Davenport?",
                  "What is the address of the Davenport office?"):
            _refuses_cleanly(self, S.answer(q), q)


class BranchHours(unittest.TestCase):
    def test_named_branch_gets_its_own_hours_only(self):
        q = "What are the lobby hours at the Waukee branch?"
        a = S.answer(q)
        self.assertTrue(a.confident)
        self.assertIn("8:30 a.m.-5:30 p.m.", a.text)
        self.assertIn("12:30", a.text)
        # other branches' schedules must not leak in
        self.assertNotIn("9 a.m.", a.text)
        self.assertNotIn("6 p.m.", a.text)
        self.assertNotIn("vary by branch", a.text)

    def test_drive_up_only(self):
        a = S.answer("What are the drive-up hours at the Platteville branch?")
        self.assertIn("9 a.m.-noon", a.text)
        self.assertNotIn("8 a.m.-1 p.m.", a.text)
        self.assertNotIn("Lobby", a.text)

    def test_generic_hours_unchanged(self):
        a = S.answer("What are Dupaco's general lobby hours?")
        self.assertTrue(a.confident)
        self.assertIn("8:30", a.text)
        self.assertIn("vary by branch", a.text)
        d = S.answer("What are the drive-up hours?")
        self.assertIn("8 a.m.", d.text)
        self.assertIn("6 p.m.", d.text)

    def test_mullan_avenue_refuses(self):
        q = "Can you give me the lobby hours for the Mullan Avenue branch?"
        _refuses_cleanly(self, S.answer(q), q)

    def test_existing_branch_without_hours_refuses(self):
        q = "What are the Dyersville branch hours?"
        _refuses_cleanly(self, S.answer(q), q)

    def test_unknown_branch_hours_refuse(self):
        q = "What are the hours at the Omaha branch?"
        _refuses_cleanly(self, S.answer(q), q)


if __name__ == "__main__":
    unittest.main()
