# Blind set 2 (dev/blind2_eval.jsonl)

63 member-style questions written blind (without reading the submission code or the other dev eval sets),
grounded in the snapshot pages under `snapshots/`. Every `must_include` value was checked with
`harness/scoring.contains` against the expected source page in the latest snapshot (or the `as_of`
snapshot); every `must_not_include` value is a real wrong/stale/other-tier value present in the data.

Categories: fee, rate, temporal_as_of, entity_attribution, program_info, how_to, branch_info, should_refuse.
14 paraphrase groups (`group` ids `b2-g-*`).

Run:

    python3 harness/run_eval.py --submission submission --eval dev/blind2_eval.jsonl --out results_blind2

**Keep this set unseen by anyone tuning the system.** Do not read the questions or per-question results
while changing the submission, and do not add these questions (or close variants) to any tuning set.
Once it has been used to guide changes, it is no longer a blind test.
