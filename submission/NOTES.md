# Dupaco knowledge system: how it works

No LLM, no network, no third-party packages. Everything is deterministic Python standard library,
so answers are identical run to run.
`ingest()` of all five snapshots takes about twenty seconds; a typical `answer()` call takes
10 to 50 ms (p95 under 0.3 s on the public set).

## Ingest

For every page in every snapshot we keep a `PageVer` with:

* cleaned text (images dropped, links flattened, "related articles" blocks cut off),
* a page kind from the URL (`rates`, `product`, `landing`, `newsroom`, `blog`, `calculator`, `other`)
  and an authority score derived from it (rates page 100, product page 80, ... calculator 15),
* a content date: the newest "Rates are effective as of MM/DD/YYYY" line for rate pages, or the
  publication date for newsroom and blog posts. This is separate from the crawl date.

Six kinds of knowledge are extracted per snapshot (`SnapshotView`):

1. **Structured facts from markdown tables.** Each cell in a rate column (APY, dividend rate, APR,
   rate, minimum to open/earn) becomes a `Fact` with the heading above the table, the row label,
   parsed attributes (term in months, balance or loan-amount tier, model-year range, jumbo/special,
   Roth/Traditional, purchase/refinance), the effective date printed under the table, and the page's
   URL and authority. Footnote markers glued to numbers (`5.39% APR1`, `1.80%1`) are stripped.
   The cell's own unit beats a sloppy column header (a calculator page labels an APR column "APY").
   Bare headings ("Regular", "Special") inherit their parent heading ("Certificates").
   Two-column fee schedules and "label: value" prose lines (fees, promo rates) are extracted too.
2. **A sentence index (BM25)** over every page, with boilerplate removed: a sentence that appears on
   many pages is navigation or footer and is dropped.
3. **Links** (anchor text, URL, the pages they appear on) for "what link should I use" questions.
4. **A branch table**: one row per branch (name, street, city, phone, lobby and drive-up hours,
   closed flag), merged across the locator, contact and news pages by city and street words.
5. **Company facts** from the "About Dupaco" boilerplate that ends every press release (member and
   asset totals, branch count, employees). The newest dated release wins, except for totals that
   only grow: there the largest figure among recent releases wins, because some releases reuse an
   older boilerplate paragraph.
6. **Product sections**: the heading, body and fine print of each named product or program, so a
   question that names a product is answered from that product's own section.

## Deciding what is current

* `as_of=None` means the newest snapshot. With `as_of`, we use the newest snapshot taken on or
  before that date and answer only from it, so a later rate never leaks into an "as of May" answer.
* Facts that describe the same thing (same product, same tier, same term, same metric) are grouped.
  Auto-loan tiers are grouped by position (newest-model tier first) because the tier labels drift
  between crawls ("2023-26 Model Year" became "2024-27 Model Year").
* Within a group the winner is chosen by: newest effective date, then page authority (official
  rates page > product page > other > landing page > newsroom/blog > calculator), then crawl time.
  So a calculator page whose table says "effective 1/1/2026" loses to the rates page saying
  "effective 08/31/2026", and `5.39%` beats `5.34%`.
* For prose facts (branch counts, awards, closures) the most recently dated page wins, because
  press releases are authoritative for their own facts; "22 branch office locations" from an
  April 2026 release beats "24" from a 2022 release.
* Qualifiers are respected, not treated as conflicts: 48 vs 60 months, rate vs APY, standard vs
  jumbo, personal vs business, balance tiers, purchase vs refinance. A question with a balance or
  loan amount gets exactly the tier that contains it; a model year gets the tier that contains it.
* If a snapshot has nothing at all on a topic (not "the product exists but that term is not
  listed"), we look in earlier snapshots, since a page can simply drop out of a crawl, and say so
  in `conflict_note`.

## Answering

Questions are first normalised (number words, "5/1" style ratios, common abbreviations) and
expanded with a small synonym table. They are then routed by rules: routing number, links, branch
count, branch hours, addresses and phones, membership eligibility, contact information, company
facts, fees, named products, then product families (auto, certificate, mortgage, credit
card, savings...). Product questions are answered from the structured facts with templates, so the
answer carries only the chosen value(s) and no stale numbers; dates and the reasons for picking a
value go into `conflict_note`, not the member-facing text. Everything else falls back to BM25
sentence retrieval re-ranked by page recency, authority and query coverage; FAQ-style questions on
a page are answered with the sentence that follows them. The fallback detects what the question
asks for (an amount, a date, a phone number or a person) and prefers the sentence that carries
that kind of value next to the question's key words; among dated news articles stating the same
kind of figure, the newest one valid at `as_of` wins.

When the site does not publish something (7/1 ARM rate, construction-loan rate, a branch that is
not in the locator) we return `confident=False` with a short plain-text refusal that contains no
numbers. The fallback also refuses when the question's main noun phrase does not appear in the
best passage, its heading or its page title, when the question names a product variant the site
does not list, when an amount lies outside every published tier, and when a holiday's opening
hours are not published.

## Known limits

The rate, fee, tier and `as_of` paths are structured and stable. Open-ended questions about news
releases and programs depend on word overlap: on question sets written independently of the code
they score about 0.64, against 0.95 on the public set. Typical misses are an older release being
chosen over a newer one, the right page but the wrong sentence, and refusals on unusual wording.

## Conflicts

`conflicts()` reports (a) rows where pages in the newest snapshot disagree, (b) rows whose winning
value changed between snapshots, (c) expired promotions that still quote a different value, and
(d) prose contradictions such as the branch count. Identical value sets (personal/business/IRA copies
of the same certificate) are merged into one entry. Each entry names the value we believe and why.

## Running

    python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl

No API keys are needed.
