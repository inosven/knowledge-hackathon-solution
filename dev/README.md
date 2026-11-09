# Dev eval set (open-ended)

`open_eval.jsonl`: 58 hand-labeled questions that exercise the free-text fallback paths
(`h_retrieve`, `h_prose`): fee schedule lookups, general facts, how-to, product descriptions,
plus 7 `should_refuse` items and 8 `branch_info` items (dev-051 to dev-058, branch addresses, per-branch hours, branch phone). Same schema as `eval/public_eval.jsonl`. 13 paraphrase groups
(`group` starting with `g-`). Values were checked against `snapshots/05_2026-09-28_latest`.
This is a development set, not the official one. Do not tune the system to these exact strings.

**2026-11-09:** `blind_eval.jsonl` (56 questions, see `BLIND_README.md`) has been used for debugging the fallback paths
and is now a second development set, no longer a blind estimate. Run it with
`python3 harness/run_eval.py --submission submission --eval dev/blind_eval.jsonl --out results_blind`.
A fresh blind set has to be written to estimate hidden-set generalisation again.

**2026-11-09 (round 2):** `blind2_eval.jsonl` (63 questions, see `BLIND2_README.md`) has also been used for debugging
(five cross-cutting mechanisms, see the "Answering" section of `submission/NOTES.md`) and is now a third development set. Run it with
`python3 harness/run_eval.py --submission submission --eval dev/blind2_eval.jsonl --out results_blind2`.

Run:

    python3 harness/run_eval.py --submission submission --eval dev/open_eval.jsonl --out results_dev

List failing questions:

    python3 - <<'PY'
    import json
    qs={json.loads(l)['id']:json.loads(l) for l in open('dev/open_eval.jsonl')}
    for l in open('results_dev/per_question.jsonl'):
        r=json.loads(l)
        if r['score']<0.999:
            q=qs[r['id']]
            print(r['id'],r['category'],r['score'],'missed',r['missed'],'trap',r['trap_hits'])
            print('  Q:',q['question'],'| exp:',q['expected_answer'][:150])
            print('  A:',r['answer']['text'][:300])
    PY
