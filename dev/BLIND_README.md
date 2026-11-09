# Blind eval set (dev/blind_eval.jsonl)

**Status 2026-11-09: burned.** It was used to debug the submission and is now a development set (see dev/README.md).

56 questions written blind (without reading submission/, tests/, dev/open_eval.jsonl or results_dev/),
grounded only in the snapshot pages. Categories: fees (10), rates (7), temporal_as_of (4),
company_facts (9), how_to (9), branch_info (9), should_refuse (8); 11 paraphrase groups (pairs).
Every must_include was checked with harness/scoring.contains against the expected source pages in
the latest snapshot (or the as_of snapshot); every must_not_include value occurs somewhere in the data.

Run:

    python3 harness/run_eval.py --submission submission --eval dev/blind_eval.jsonl --out results_blind

**Keep this set unseen by anyone tuning the system.** Do not read the questions, use them for
debugging, or iterate the submission against them; it exists only to estimate hidden-set
generalisation. If it is used for tuning, it stops being a blind estimate and a new set must be written.
