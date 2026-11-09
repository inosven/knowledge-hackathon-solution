# Blind evaluation set 3 (blind3)

Date: 2026-11-09

## How it was built
- I wrote it blind. I did not read `submission/`, `dev/` (earlier question sets), `tests/`, the internal working notes or earlier `results*/`. I read only `README.md`, a short project entry note, `eval/public_eval.jsonl` (for the schema), `harness/run_eval.py`, `harness/scoring.py` and `snapshots/`.
- Every question comes from reading the snapshot pages directly. Rate tables were diffed row by row across all 5 snapshots to find values that changed. Newsroom, foundation, grant, branch, fee-schedule, impact-report, promo landing and how-to pages were read in full.
- 68 questions, ids b3-001..b3-068, in the same schema as the public eval. There are 12 paraphrase pairs (groups `g12m, ghys, gboat, gauto48, gwire, gcashadv, ghardship, gfraud, gvpml, glostcard, gmadison, ggrimes`), 7 `should_refuse` items and 8 `as_of` items. The as_of dates fall between snapshots, so the newest snapshot taken on or before the date applies.
- Labels follow the newest source valid at `as_of`. Superseded values from older snapshots, stale landing pages or older news releases are `must_not_include` traps. Total member count and total assets were deliberately left out.
- `must_include` tokens were machine-checked with the harness's own `contains()` against the listed `expected_sources` in the relevant snapshot. The one exception is b3-033: the source spells it "Thirty-two", and the label is the digit `32`.
- Topics deliberately not in the public eval: High-Yield Savings, IRA and HSA tiers, boat/RV and home-improvement tiers, the fee schedule (wires, safe deposit, overdraft, cash advance), the hardship grant, Gift of Innovation, the KCCC program change, Foundation nonprofit grant rounds, the 2025 Impact Report (fraud, volunteer hours, net income), Holiday Club results, the Achieve Money Market promo, Mortgage Rate Relief 2026, board elections, new hires and awards, scholarships, the Visa PIN, lost cards, student loans, the Business Visa, Launchpad, Switch & Earn, and branches in Madison, Platteville, Cedar Rapids, Grimes and Dubuque.

## Run (once, no reruns, no label changes)
    python3 harness/run_eval.py --submission submission --eval dev/blind3_eval.jsonl --out results_blind3

## Result
final_score 0.6415 | accuracy (qa) 0.6446 | consistency 0.9732 (paraphrase 0.9464, repeat 1.0) | stale trap hits 1 | p95 0.04 s | errors 0.
No conflicts gold was used, so the conflicts pillar is not included.
Per-question output: `results_blind3/per_question.jsonl`; summary: `results_blind3/report.json`.
