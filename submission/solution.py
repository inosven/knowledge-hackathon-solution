"""
Dupaco knowledge system: rules + structured facts + BM25 sentence retrieval. No LLM, no network.

See NOTES.md for how "current" is decided.
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(next(d / "harness" for d in _HERE.parents if (d / "harness" / "contract.py").exists())))
from contract import Answer, Conflict, Snapshot, Source  # noqa: E402

import kb  # noqa: E402
from kb import Fact, KnowledgeBase, SnapshotView, norm_label, tokenize  # noqa: E402

REFUSAL = ("I couldn't find that on Dupaco's website. It isn't listed there, so please contact Dupaco "
           "directly and a team member can help.")

WORD_NUM = kb.NUM_WORDS   # one canonical number-word table, shared with the tokenizer


# Words members use that the website does not. Added to a retrieval query at a lower weight than the
# member's own words, so a page that uses the member's exact word still ranks first.
SYNONYMS = {
    "car": ["auto", "vehicle"], "vehicle": ["auto", "car"], "auto": ["vehicle", "car"], "truck": ["auto", "vehicle"],
    "mortgage": ["home", "loan"], "house": ["home"],
    "fee": ["charge", "cost"], "charge": ["fee"], "cost": ["fee"], "price": ["fee", "cost"],
    "visa": ["credit", "card"], "card": ["visa"],
    "heloc": ["home", "equity", "line", "credit"],
    "atm": ["cash"], "wire": ["transfer"],
    "email": ["contact"], "cellphone": ["mobile"], "smartphone": ["mobile"],
    "hours": ["open", "lobby"],
    "near": ["location", "branch"], "nearest": ["location", "branch"],
    "apy": ["yield", "dividend"], "interest": ["rate", "dividend"],
    "overdraw": ["overdraft"], "overdrawn": ["overdraft"], "overdrew": ["overdraft"],
    "add": ["put", "deposit", "contribute"], "qualify": ["eligible"], "qualifies": ["eligible"],
    "travel": ["nationwide", "wherever", "anywhere"], "traveling": ["nationwide", "wherever", "anywhere"],
    "travelling": ["nationwide", "wherever", "anywhere"], "phone": ["call", "contact", "mobile", "app"],
}


# Member phrasings of a service the site names differently (credit-union terms of art).
PHRASE_SYNONYMS = [
    (re.compile(r"\b(?:(?:bank|banking|business|transactions?) (?:at|with|through) )?(?:other|another|different|participating|any) credit unions?\b|\bcredit unions? (?:nationwide|across the country|out of state)"), "shared branching"),
    (re.compile(r"\bdeposit (?:a |my )?checks? (?:with|from|using|on|by) (?:my |your |a )?(?:phone|cell|smartphone|mobile)"), "mobile deposit"),
    # telling the credit union about a trip so cards keep working = a travel notice
    (re.compile(r"\b(?:going|traveling|travelling|heading|flying|trip|vacation|vacationing) (?:to|abroad|overseas|out of (?:the )?(?:country|state))\b"
                r"[^?]{0,90}\b(?:cards?|debit|credit|blocked|declined)\b|\bcards? (?:don'?t |won'?t |doesn'?t |do not |will not )?(?:get |be )?"
                r"(?:blocked|declined|frozen|shut off|flagged)\b|\btravel (?:plans?|notices?|notifications?|alerts?)\b"), "travel notice"),
    # a young child's account = the kids' account
    (re.compile(r"\b(?:[1-9]|1[0-2]|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)[- ](?:year|yr)s?[- ]olds?\b"
                r"|\b(?:my|our|for) (?:kids?|child|children|toddler|baby|grandkids?|grandchild(?:ren)?|little ones?)\b"), "kids"),
]


def _build_synonym_table(table: dict[str, list[str]]) -> dict[str, tuple[str, ...]]:
    """SYNONYMS stays human-readable; queries arrive stemmed (kb.tokenize), so look-ups use stemmed keys and values."""
    out: dict[str, list[str]] = {}
    for key, extras in table.items():
        k = kb.stem(key)
        bucket = out.setdefault(k, [])
        for e in extras:
            e = kb.stem(e)
            if e != k and e not in bucket:
                bucket.append(e)
    return {k: tuple(v) for k, v in out.items()}


SYNONYM_TABLE = _build_synonym_table(SYNONYMS)


def expand_query(toks: list[str]) -> dict[str, float]:
    """Term -> weight: 1.0 for the member's own (stemmed) words, less for synonyms added on their behalf."""
    weights = {t: 1.0 for t in toks}
    for t in toks:
        for extra in SYNONYM_TABLE.get(t, ()):
            weights.setdefault(extra, 0.6)
    return weights


# Fee label gate. A fee answer is only confident when the fee schedule's label names the thing asked about.
# Words that say "this is a fee question" but not which fee:
FEE_GENERIC = {kb.stem(w) for w in """fee fees charge charges charged cost costs price pay paid much dupaco account accounts
    service services there bank use using used me what i'm im i've ive i'd need want my your like would""".split()} | kb.STOP
# Member phrasings of where money goes -> the schedule's words ("Domestic wire", "International wire transfer").
FEE_PHRASES = [
    (re.compile(r"\b(?:within|inside|in|across) the (?:u\.?\s?s\.?(?:a\.?)?|united states|us|usa|country)(?![a-z])|\bto (?:another|a) bank in the (?:u\.?s\.?|us|united states)|\bstateside\b", re.I), " domestic "),
    (re.compile(r"\bdomestically\b|\bnationally\b", re.I), " domestic "),
    (re.compile(r"\binternationally\b|\babroad\b|\boverseas\b|\boutside (?:of )?the (?:u\.?\s?s\.?(?:a\.?)?|united states|us|usa|country)(?![a-z])|\b(?:to|in) (?:a |another )?foreign (?:country|bank)\b", re.I), " international "),
]


_DIM_RE = re.compile(r"\b(\d{1,2})\s*[x×]\s*(\d{1,2})(?:\s*[x×]\s*(\d{1,2}))?\b", re.I)


def _dims(m: re.Match) -> str:
    """'3 X 10 X 21' -> '3x10 10x21', '3x10' -> '3x10': box sizes compare as width-by-depth pairs."""
    d = [x for x in m.groups() if x]
    return " " + " ".join(f"{a}x{b}" for a, b in zip(d, d[1:])) + " "


def fee_normalize(text: str) -> str:
    for rx, rep in FEE_PHRASES:
        text = rx.sub(rep, text)
    return _DIM_RE.sub(_dims, text)
# Member wording -> fee-schedule wording (the schedule says "Official check", "Consumer overdrafts").
FEE_LABEL_ALIAS = {kb.stem(k): {kb.stem(x) for x in v} for k, v in {
    "cashier": ["official"], "cashiers": ["official"],
    "nsf": ["overdraft"], "bounced": ["overdraft"], "bounce": ["overdraft"], "overdraw": ["overdraft"],
    "overdrawn": ["overdraft"], "another": ["non-dupaco"], "other": ["non-dupaco"],
    "expedited": ["rush"], "expedite": ["rush"], "express": ["rush"], "overnight": ["rush"], "faster": ["rush"],
    "counter": ["count"], "counters": ["count"], "counting": ["count"], "nonmember": ["non", "member"],
    "certified": ["certified"], "notarize": ["notary"], "notarized": ["notary"],
}.items()}
# Product-scope words: if the question names one, the fee label must name it too, so a
# credit-card late fee is never answered with a deposit-account fee. (Sections are unreliable: some are the PDF URL.)
FEE_SCOPE = {kb.stem(w) for w in "credit loan loans mortgage visa heloc certificate certificates ira business auto hsa health savings".split()}
_FOOTNOTE_SENT_RE = re.compile(r"\brates? (?:are|is) effective|\baccurate as of\b|\beffective (?:as of )?\d|\bsubject to change without notice", re.I)
_PRICE_RE = re.compile(r"\$\s?\d|\d\s?%|\bfree\b|no (?:extra |additional |monthly |annual |service )*(?:cost|charge|fee)|at no cost|waive", re.I)


def fee_content_words(text: str) -> set[str]:
    text = fee_normalize(text)
    words = set(tokenize(text)) - FEE_GENERIC
    if re.search(r"\bchecking\b", text, re.I) and not re.search(r"\bchecks?\b", text, re.I):
        words.discard(kb.stem("checking"))   # "checking account" is the account, not a paper check
    return words


def fee_label_cover(qwords: set[str], label: str) -> frozenset:
    """Question content words named by a fee label (directly or through FEE_LABEL_ALIAS)."""
    lw = fee_content_words(label)
    return frozenset(w for w in qwords if w in lw or FEE_LABEL_ALIAS.get(w, set()) & lw)


# Product names: the member's short forms -> the words of the product-page heading (kb.product_sections).
PRODUCT_ALIAS = {kb.stem(k): {kb.stem(x) for x in v} for k, v in {
    "hsa": ["health", "savings"], "hsas": ["health", "savings"], "first": ["1st"], "cd": ["certificate"],
}.items()}
# Words of a product name that only say what kind of account it is; the rest ("1st Rate", "Free", "Platinum") is the name.
PRODUCT_KIND_WORDS = {kb.stem(w) for w in "checking savings account accounts certificate certificates loan loans card cards visa club".split()}
_RATE_CUE_RE = re.compile(r"\brates?\b|\baprs?\b|\bapys?\b|interest|percent|%|yield|dividend|\bearn")


def _with_aliases(toks) -> set:
    q = set(toks)
    for t in list(q):
        q |= PRODUCT_ALIAS.get(t, set())
    return q


def named_products(view: SnapshotView, info: dict) -> list[dict]:
    """Product-page sections whose heading the question names in full ("1st Rate Checking", "HSA" -> Health Savings Accounts).
    Only the most specific names are kept ("Business 1st Rate Checking" over "1st Rate Checking")."""
    q = _with_aliases(info["toks"])
    hits = [d for d in view.products.values() if d["toks"] <= q]
    return [d for d in hits if not any(d["toks"] < e["toks"] for e in hits)]


def strip_product_names(ql: str, prods: list[dict]) -> str:
    """The question without the product names, so 'rate' in '1st Rate Checking' is not read as a rate question."""
    for d in prods:
        words = [w for w in re.findall(r"[a-z0-9$]+", d["name"].lower())]
        if words:
            ql = re.sub(r"\b" + r"\W+".join(re.escape(w.rstrip("s")) + r"s?" for w in words) + r"\b", " ", ql)
            if words[0] == "1st":
                ql = re.sub(r"\bfirst\W+" + r"\W+".join(re.escape(w.rstrip("s")) + r"s?" for w in words[1:]) + r"\b", " ", ql)
    return ql


_DEFN_FILLER = {kb.stem(w) for w in "dupaco dupaco's offer offered product account accounts mean work works exactly".split()}


def product_residual(info: dict, prods: list[dict]) -> set:
    """Question words beyond the product name: empty for "What is 1st Rate Checking?", {'contribut', 'limit', ...}
    for "What is the annual HSA contribution limit?" (that is not a request for the product description)."""
    names = set().union(*(d["toks"] for d in prods)) if prods else set()
    return {t for t in info["toks"] if t not in names and not (PRODUCT_ALIAS.get(t, set()) & names) and t not in _DEFN_FILLER}


def product_scope(prods: list[dict]) -> set:
    """Name words a fee label must carry when the question names a product ('free' for Free Checking)."""
    out: set = set()
    for d in prods:
        out |= set(d["toks"]) - PRODUCT_KIND_WORDS
    return out


# Unknown products: "Bitcoin savings account", "Platinum Elite Checking". The words in front of a product noun are its name;
# if the website never uses that name, the product does not exist and the answer is a refusal, not the nearest product.
_PRODUCT_NOUN_RE = re.compile(r"\b(checking|savings|account|accounts|card|cards|loan|loans|certificate|certificates|cds?|iras?|club|mortgage|heloc)\b")
_NAME_STOP = kb.STOP | {"dupaco", "dupacos", "dupaco's", "what", "whats", "what's", "the", "a", "an", "my", "your", "our", "new", "current",
                        "much", "monthly", "annual", "yearly", "interest", "rate", "rates", "fee", "fees", "apy", "apr", "minimum", "open",
                        "opening", "get", "need", "want", "like", "do", "does", "is", "for", "on", "of", "with", "to", "in", "at", "about",
                        "there", "any", "kind", "type", "types", "which", "best", "good", "cheap", "cheapest", "high", "higher", "highest",
                        "low", "lowest", "regular", "standard", "basic", "personal", "own", "one", "same"}


def product_name_phrase(ql: str) -> Optional[list[str]]:
    """Name words directly in front of the first product noun ('bitcoin' in 'Dupaco's Bitcoin savings account'), or None."""
    m = _PRODUCT_NOUN_RE.search(ql)
    if not m:
        return None
    before = re.findall(r"[a-z0-9$'’-]+", ql[:m.start()])
    words: list[str] = []
    for w in reversed(before[-2:]):
        w = w.strip("'’-")
        if not w or w in _NAME_STOP or re.search(r"\d", w) or w.endswith(("'s", "’s")):
            break
        words.insert(0, w)
    if not words:
        return None
    return words + [m.group(1)]


def name_on_site(view: SnapshotView, words: list[str], raw: str = "") -> bool:
    """False when the question gives the product a capitalised name ("Bitcoin savings", "Platinum Elite Checking") and no
    sentence on the site has all the name words together with the product noun. Lower-case descriptions ("lost debit
    card") are left alone: they describe a situation, not a product name."""
    toks = [t for t in tokenize(" ".join(words)) if t]
    if not toks or not view.bm25:
        return True
    if not any(re.search(r"(?<!^)\b" + re.escape(w[:1].upper() + w[1:]) + r"\b", raw.strip()) for w in words[:-1]):
        return True
    sets = []
    for t in set(toks):
        post = view.bm25.postings.get(t)
        if not post:
            return False
        sets.append({i for i, _ in post})
    sets.sort(key=len)
    common = sets[0]
    for x in sets[1:]:
        common = common & x
        if not common:
            return False
    return True


# ----------------------------------------------------------------------------- core phrase gate
# A retrieved passage is only a confident answer when it is about the thing asked. The thing asked is the question's
# core noun phrase(s): runs of adjacent content words ("reverse mortgage", "lot loan", "withdrawal limit"). Words
# that frame a question rather than name a thing break a run; weak words ("lot", "cash", "daily") can be part of a
# phrase but are never on their own evidence that a passage matches.
PHRASE_GENERIC = {kb.stem(w) for w in """dupaco dupaco's dupacos other others any some new current currently much many kind kinds type types
    best good way ways please exactly really also today now right available offer offers offered get use using need want like
    still account accounts rate rates interest typical usual usually roughly about around approximately ever possible
    i'm im i've ive i'd you're there here can't cannot don't doesn't whats what's info information details detail
    help tell know find see go much more most less least hundred thousand million billion grand dollar dollars bucks""".split()} | kb.STOP
WEAK_WORDS = {kb.stem(w) for w in """lot lots cash daily day days time times thing things way make made take keep back go going
    year years month months week weeks money place people person put want need set number amount dollar dollars""".split()}
PRODUCT_HEAD_NOUNS = {kb.stem(w) for w in """mortgage mortgages loan loans account card cards certificate certificates savings checking
    cd cds ira iras heloc insurance club""".split()} | {"certificate"}
_QUANTITY_Q_RE = re.compile(r"\bhow much\b|\bmaximum\b|\bmax\b|\blimits?\b|\bpenalt(?:y|ies)\b|\bminimum\b|\bcap\b|\bhow many\b"
                            r"|\b(?:what|which)(?:'s| is| are)? (?:the |my |your )?(?:\w+ ){0,3}(?:fee|cost|price|amount|rate|apy|apr)s?\b"
                            r"|\brates?\b|\baprs?\b|\bapys?\b|\binterest rate|\byield\b|\bdividend\b")
_HEDGE_RE = re.compile(r"\bmay apply\b|\bmay vary\b|\bvar(?:y|ies) by\b|\bcontact us\b|\bcall us\b|\bconsult\b|\bspeak (?:with|to) an?\b"
                       r"|\bask (?:a|your) \b|\bwill be imposed\b|\bmay be imposed\b|\bsubject to approval\b", re.I)
_FINE_PRINT_RE = re.compile(r"fine print|disclosure|terms (?:and|&) conditions|footnote", re.I)
_STEP_RE = re.compile(r"\b(?:tap|click|select|choose|enter|endorse|login|follow the prompts|go to|navigate to|sign up|scan|"
                      r"step \d|first,|then|next,|finally)\b|^\s*\d+[.)]\s", re.I)
_STEP_START_RE = re.compile(r"\W*(?:\d{1,2}[.)]\s*)?(?:go to|tap|click|select|choose|enter|open|log ?in(?:to)?|sign in|navigate to|then)\b", re.I)
QTY_NOUNS = {kb.stem(w) for w in "penalty penalties limit limits maximum minimum fee fees cost price rate rates apr apy yield dividend charge amount cap".split()}
_QTY_ANSWER_RE = re.compile(r"\d|\bfree\b|\bno (?:extra |additional |monthly |annual |service )*(?:cost|charge|fee|limit|minimum|penalty)s?\b|\bat no cost\b|\bunlimited\b|\bwaive", re.I)

# Answer types: what kind of value the question asks for, and how that value looks in a sentence.
_MONTH_ABBR = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
ATYPE_TEXT_RE = {
    "phone": re.compile(r"\(?\b\d{3}\)?[\s.-]\d{3}-\d{4}\b|\bext(?:ension)?\.?\s?\d{2,4}\b", re.I),
    "date": re.compile(r"\b" + _MONTH_ABBR + r"\s+\d{1,2}(?:st|nd|rd|th)?\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b" + _MONTH_ABBR + r"\s+20\d\d\b", re.I),
    "amount": re.compile(r"\$\s?\d"),
    "person": re.compile(r"\b(?!(?:Dupaco|The|Credit|Union|Community|Board|Directors?|Foundation|Annual|Membership|Iowa|Dubuque|"
                         r"Innovation|Lab|Shine|Visa|Learning|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|President|Chief|"
                         r"Vice|Senior|Executive|Officer|Following|During|These|Fifteen|Twenty|Ten)\b)[A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+){1,2}\b"),
}
_ATYPE_Q = [
    # "phone" alone is not a phone-number question ("deposit a check with my phone"): it needs number/call wording
    ("phone", re.compile(r"\b(?:phone|telephone|contact|fax) (?:number|line|#)|\bwho (?:do|should|can) i (?:call|phone|contact)\b|\b(?:number|line) to call\b|"
                         r"\bcall\b[^?.]{0,40}\bnumber\b|\bnumber\b[^?.]{0,25}\bcall\b|\bextension\b|\bwhat(?:'s| is) (?:the|their|your) number\b")),
    ("date", re.compile(r"\bwhen (?:does|do|is|will|was|did|are|can|should|must|would)\b(?! (?:i|you|we)\b)|\bwhat (?:date|day)\b|\bdeadline\b|\bdue date\b|\bby when\b")),
    ("amount", re.compile(r"\bhow much\b|\b(?:monthly )?payment (?:be|would be|come to)\b|\bwhat would (?:my|the) (?:monthly )?payment\b|"
                          r"\bthe most\b[^?.]{0,40}\b(?:pay|give|award|grant|get)\b|\bworth\b|\bup to how much\b")),
    ("person", re.compile(r"^who\b(?! (?:do|should|can|may|is eligible|qualif|can join))|\bwho (?:is|was|became|serves|served|leads|heads|chairs|runs|took over)\b(?! eligible)")),
]


def answer_types(ql: str) -> list:
    """Value kinds the question asks for, in question order ('When does it open, and how much is it worth?')."""
    out = []
    for name, rx in _ATYPE_Q:
        m = rx.search(ql)
        if m and not (name == "person" and "phone" in [x for x, _ in out]):
            out.append((name, m.start()))
    if any(n == "phone" for n, _ in out):
        out = [(n, k) for n, k in out if n != "person"]
    return [n for n, _ in sorted(out, key=lambda x: x[1])]


def person_role(ql: str) -> set:
    """'who became chair of the board' -> {'chair'}: the role a person-answer must be named next to."""
    m = re.search(r"\bwho (?:is|was|became|serves as|served as|is serving as|took over as|heads|leads|chairs)\s+(?:the |dupaco'?s? |its |their )?([a-z][a-z-]*)", ql)
    if not m or m.group(1) in ("the", "a", "an", "in", "on", "at", "dupaco", "dupacos", "behind"):
        return set()
    return {kb.stem(m.group(1))}



def _phrase_words(q: str) -> list[list[str]]:
    """Runs of adjacent content words of the question, stemmed: 'What rate do you offer on a reverse mortgage?' ->
    [['revers', 'mortgag']]. Stop words, digits and question-frame words break a run."""
    return [[t for _, t in r] for r in phrase_runs(q)]


def phrase_runs(q: str) -> list[list[tuple[str, str]]]:
    """Like _phrase_words, with each word as (as written, stemmed)."""
    q = kb.normalize_text(q)
    runs, cur = [], []
    for raw in re.findall(r"[a-z0-9$%][a-z0-9$%.'-]*|[,.;:?!()]", q):
        parts = [x for x in raw.strip(".'-").split("-") if x] if raw not in ",.;:?!()" else []
        if not parts:
            if cur:
                runs.append(cur)
            cur = []
            continue
        for w in parts:
            w2 = w[:-2] if w.endswith("'s") else w
            t = kb.stem(kb.ALIAS.get(w2, w2))
            if not t or w in kb.STOP or w2 in kb.STOP or t in PHRASE_GENERIC or w in PHRASE_GENERIC or re.search(r"\d", t):
                if cur:
                    runs.append(cur)
                cur = []
                continue
            cur.append((w2, t))
    if cur:
        runs.append(cur)
    return [r for r in runs if len(r) >= 2]


# Head nouns of the things members ask about. A run ending in one of these is a core phrase ("lot loan",
# "withdrawal limit"); a run ending in anything else is usually a verb and its object ("put money") and is not enforced.
HEAD_NOUNS = {kb.stem(w) for w in """mortgage mortgages loan loans account card cards certificate certificates savings checking cd cds
    ira iras heloc insurance club limit limits penalty penalties fee fees deposit deposits withdrawal withdrawals transfer
    transfers wire wires branch branches union unions box boxes blocker blockers password advance statement statements
    check checks payment payments app lab plan program""".split()} | {"certificate"}


# Modifiers that only say which ordinary product line is meant ("credit card", "savings account"): a passage about
# "your Dupaco cards" answers a "debit and credit cards" question. Other modifiers ("reverse", "lot", "withdrawal") name
# a different thing and must be in the passage.
KIND_MODIFIERS = {kb.stem(w) for w in """credit debit savings checking health business personal auto car vehicle home student money
    market share visa dupaco member members online""".split()}


def core_pairs(q: str) -> list[tuple[str, str]]:
    """The head noun and the word in front of it, for every multi-word run that ends in a head noun
    ('withdrawal limit', 'reverse mortgage'); product-line compounds ('credit card') are not enforced."""
    # "60-month loan": the term unit after a number is a slot (the term), not a kind of loan
    units = {kb.stem(m.group(1)) for m in re.finditer(r"\d+\s*-?\s*(months?|years?|yrs?|days?)\b", q.lower())}
    return [(r[-2], r[-1]) for r in _phrase_words(q) if r[-1] in HEAD_NOUNS and r[-2] not in KIND_MODIFIERS
            and r[-2] not in units]


PHRASE_EQUIV = {kb.stem(k): {kb.stem(x) for x in v} for k, v in {
    "phone": ["mobile"], "cellphone": ["mobile"], "smartphone": ["mobile"], "visa": ["credit"], "cd": ["certificate"],
    "banking": ["bank"], "bank": ["banking"],
}.items()}


def _variants(t: str) -> set:
    return {t} | PRODUCT_ALIAS.get(t, set()) | PHRASE_EQUIV.get(t, set())


def pair_in(toks: list[str], a: str, b: str, window: int = 2) -> bool:
    """a within `window` tokens before b (either may match through a synonym)."""
    va, vb = _variants(a), _variants(b)
    pos_a = [i for i, t in enumerate(toks) if t in va]
    if not pos_a:
        return False
    return any(t in vb and any(0 < i - j <= window for j in pos_a) for i, t in enumerate(toks))


def pair_on_site(view: SnapshotView, a: str, b: str) -> bool:
    """Does any sentence, heading or page title of the snapshot have a right in front of b ('reverse' 'mortgage')?"""
    cache = view.__dict__.setdefault("_pair_cache", {})
    if (a, b) in cache:
        return cache[(a, b)]
    found = False
    if view.bm25:
        sa = set()
        for x in _variants(a):
            sa |= {i for i, _ in view.bm25.postings.get(x, ())}
        sb = set()
        for x in _variants(b):
            sb |= {i for i, _ in view.bm25.postings.get(x, ())}
        found = any(pair_in(view.sentences[i].toks, a, b) for i in sa & sb)
    if not found:
        heads = view.__dict__.get("_head_toks")
        if heads is None:
            hs = {st.section for st in view.sentences} | {p.title for p in view.pages.values()}
            heads = view._head_toks = [tokenize(h) for h in hs if h]
        found = any(pair_in(h, a, b) for h in heads)
    cache[(a, b)] = found
    return found


# Words in front of a product noun that the rate routes already understand (slots and sub-product filters) or that
# only describe the member's situation; anything else ("reverse", "lot", "bridge") is a product the tables must name.
QUALIFIER_OK = {kb.stem(w) for w in """new used older old fixed variable adjustable arm construction purchase refinance refi jumbo special
    regular standard intro introductory promotional balance transfer lowest highest best cheapest low current first time
    first-time year month term conventional traditional roth regular share term secured unsecured my our your
    rewards platinum signature""".split()}


def product_qualifiers(q: str) -> list[tuple[str, str, str]]:
    """(modifier as written, modifier stem, head stem) for runs ending in a product noun: 'reverse mortgages' ->
    ('reverse', 'revers', 'mortgag')."""
    out = []
    for run in phrase_runs(q):
        head = run[-1][1]
        if head not in PRODUCT_HEAD_NOUNS:
            continue
        for raw, t in run[:-1]:
            if t not in KIND_MODIFIERS and t not in QUALIFIER_OK and t not in HEAD_NOUNS:
                out.append((raw, t, head))
    return out


# ----------------------------------------------------------------------------- question parsing

_SMALL_NUM = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
              "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
              "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
              "seventy": 70, "eighty": 80, "ninety": 90}
_SCALE = {"hundred": 100, "thousand": 1000, "grand": 1000, "k": 1000, "million": 1_000_000}
_WORD_AMOUNT_RE = re.compile(r"\b((?:(?:" + "|".join(sorted(_SMALL_NUM, key=len, reverse=True)) + r"|hundred|thousand|million|and)[\s-]+)*"
                             r"(?:hundred|thousand|grand|million))\b(?:\s+dollars|\s+bucks)?")


def words_to_amount(phrase: str) -> Optional[float]:
    """'three hundred thousand' -> 300000, 'ten grand' -> 10000, 'two million' -> 2000000 (None if it isn't a number)."""
    total, cur, seen = 0, 0, False
    for w in re.findall(r"[a-z]+", phrase.lower()):
        if w == "and":
            continue
        if w in _SMALL_NUM:
            cur += _SMALL_NUM[w]
            seen = True
        elif w == "hundred":
            cur = (cur or 1) * 100
            seen = True
        elif w in _SCALE:
            total += (cur or 1) * _SCALE[w]
            cur, seen = 0, True
        else:
            return None
    return float(total + cur) if seen and (total + cur) >= 100 else None


def _money_values(s: str) -> list[float]:
    vals = []
    # amounts in words ("three hundred thousand dollars", "ten grand") and bare "250k"
    for m in _WORD_AMOUNT_RE.finditer(s):
        words = re.findall(r"[a-z]+", m.group(1))
        if re.search(r"\d[\d.,]*\s*$", s[:m.start()]) or not any(w in _SMALL_NUM or w == "hundred" for w in words[:-1] or words):
            continue          # "$2.5 million" is read by the $ pattern; a bare "thousand" is not an amount
        v = words_to_amount(m.group(1))
        if v:
            vals.append(v)
    for m in re.finditer(r"(?<![$\d.,])\b(\d+(?:\.\d+)?)\s*(k|grand|thousand|million)\b", s, re.I):
        v = float(m.group(1)) * (1_000_000 if m.group(2).lower() == "million" else 1000)
        vals.append(v)
    for m in re.finditer(r"\$\s?([\d,]+(?:\.\d+)?)\s*(k|thousand|m|million)?\b", s, re.I):
        v = float(m.group(1).replace(",", ""))
        suf = (m.group(2) or "").lower()
        if suf in ("k", "thousand"):
            v *= 1000
        elif suf in ("m", "million"):
            v *= 1_000_000
        vals.append(v)
    if not vals:
        for m in re.finditer(r"\b(\d{1,3}(?:,\d{3})+|\d{4,7})(?:\.\d+)?\b(?!\s*(?:model|vehicle|%|percent|k\b|grand|thousand|million))", s):
            v = float(m.group(1).replace(",", ""))
            if v >= 100:
                vals.append(v)
    return vals


def parse_question(q: str) -> dict:
    ql = q.lower().replace("–", "-").replace("—", "-").replace("’", "'")
    info: dict = {"ql": ql, "toks": tokenize(q), "raw": q}
    m = re.search(r"\b(\d{1,3})\s*-?\s*(?:month|months|mo)\b", ql)
    if m:
        info["term"] = int(m.group(1))
    else:
        m = re.search(r"\b(\d{1,2})\s*-?\s*(?:year|yr|years)\b(?!\s*old)", ql)
        if m:
            info["term"] = int(m.group(1)) * 12
        else:
            m = re.search(r"\b(" + kb.NUM_WORD_ALT + r")\s*-?\s*(month|year)s?\b", ql)
            if m:
                info["term"] = WORD_NUM[m.group(1)] * (12 if m.group(2) == "year" else 1)
    m = re.search(r"\b(20\d\d)\b[^.]{0,25}\b(model|vehicle|car|truck|motorcycle|suv|van|minivan|pickup|crossover|atv|utv|snowmobile|jet ?ski|four[- ]?wheeler|quad|side[- ]by[- ]side|boat|rv|camper)|\b(model[- ]year|vehicle|car|suv|truck)\b[^.]{0,20}\b(20\d\d)\b", ql)
    if m:
        info["model_year"] = int(m.group(1) or m.group(4))
    info["amounts"] = [a for a in _money_values(ql) if a != info.get("model_year")]
    if "apy" in ql or "yield" in ql:
        info["metric"] = "apy"
    elif "apr" in ql:
        info["metric"] = "apr"
    elif "dividend" in ql:
        info["metric"] = "dividend_rate"
    # a negated qualifier ("the regular one, not the jumbo", "non-business", "without the promo") excludes that
    # variant instead of asking for it
    neg = {m.group(1) for m in re.finditer(r"\b(?:not|non|no|without|excluding|except|other than|isn'?t|aren'?t|rather than|instead of)"
                                          r"[- ](?:a |an |the |for |for a |for the )?(jumbo|special|promo|promotional|business|commercial|ira|roth|traditional)\b", ql)}
    info["negated"] = neg
    info["jumbo"] = "jumbo" in ql and "jumbo" not in neg
    info["special"] = ("special" in ql and "special" not in neg) or ("promo" in ql and not neg & {"promo", "promotional"})
    info["business"] = ("business" in ql or "commercial" in ql) and not neg & {"business", "commercial"}
    info["ira"] = bool(re.search(r"\bira\b|roth|traditional|retirement", ql)) and not neg & {"ira", "roth", "traditional"}
    info["refi"] = "refinanc" in ql
    info["superlative"] = bool(re.search(r"\b(highest|best|top|maximum|max|most)\b", ql))
    info["lowest"] = bool(re.search(r"\b(lowest|minimum|smallest|cheapest)\b", ql))
    # as-of phrase inside the question text
    m = re.search(rf"\b(?:as of|in|during|back in)\s+({kb.MONTHS})\s+(20\d\d)\b", ql)
    if m and not re.search(r"\b(today|now|current)", ql):
        mon = kb._MONTH_NUM[m.group(1)]
        y = int(m.group(2))
        last = (date(y + (mon == 12), (mon % 12) + 1, 1) - date.resolution).isoformat() if False else None
        import calendar
        info["as_of_text"] = date(y, mon, calendar.monthrange(y, mon)[1]).isoformat()
    return info


# Questions about Dupaco itself, answered from the press-release "About Dupaco" boilerplate (kb.extract_about).
_ORG = r"(?:dupaco|the credit union|your credit union|you guys|you)"
ABOUT_ROUTES = [
    ("ceo", re.compile(r"\b(?:ceo|chief executive)\b|\bpresident of " + _ORG + r"|\b" + _ORG + r"(?:'s)? president\b|^who is (?:the )?president\b"
                       r"|who (?:runs|leads|heads|manages|is in charge of|is the head of|is the leader of) " + _ORG + r"|\b(?:head|leader) of " + _ORG + r"\b")),
    ("members", re.compile(r"how many (?:members|member-owners|people (?:are|belong)|people bank)(?! (?:can|may|are allowed|per|on|in an?|to)\b)|number of (?:dupaco )?members|membership (?:size|count|total|numbers?)|how big is (?:dupaco'?s? )?membership")),
    ("employees", re.compile(r"how many (?:employees|staff|people (?:work|does " + _ORG + r" employ)|workers|team members)|number of employees|how many people (?:does " + _ORG + r" )?employ")),
    ("counties", re.compile(r"how many counties|\bcounties (?:does|do) " + _ORG + r"|which counties|\bcount(?:y|ies)\b.{0,40}\b(?:serve[sd]?|service area|footprint|cover)"
                            r"|\b(?:service area|footprint|market area)\b.{0,40}\bcount(?:y|ies)\b|number of counties")),
    ("assets", re.compile(r"\b(?:total )?assets\b|how (?:big|large) is " + _ORG + r"\b(?! membership)")),
    ("hq", re.compile(r"headquarter|\bhq\b|(?:where|what city|which city|what town)\b.{0,40}\b(?:based|located|home base)\b(?!.{0,30}\b(?:branch|atm|office|learning lab))|main office|corporate office")),
    ("founded", re.compile(r"\b(?:founded|established|chartered|formed|incorporated)\b|(?:get|got) (?:its|their|your) start|when did " + _ORG + r".{0,25}\b(?:start|begin|open)\b(?!.{0,20}\b(?:offer|sell|charg))|how old is " + _ORG + r"|what year did " + _ORG + r".{0,20}\b(?:start|begin)"
                           r"|(?:when|what year) (?:was|were) " + _ORG + r"(?: credit union)? (?:started|begun|created|founded|formed|organized|launched)\b|\b(?:started|founded) by whom\b|who (?:started|founded) " + _ORG)),
    ("recognition", re.compile(r"\bforbes\b|\b(?:awards?|recognitions?|accolades?|honors?)\b.{0,40}\b(?:dupaco|received|won|earned|gotten)|\b(?:dupaco|you) (?:has |have )?(?:won|received|earned)\b")),
]


def about_field(ql: str) -> Optional[str]:
    if detect_family(ql) and not re.search(r"ceo|president|forbes|award", ql):
        return None        # "how many members can be on a certificate" is a product question
    branchy = re.search(r"\b(?:branch|branches|atm|atms|learning lab|location|locations)\b|\boffice\b(?<!main office)(?<!corporate office)", ql)
    for field, rx in ABOUT_ROUTES:
        if field == "hq" and branchy:
            continue
        if rx.search(ql):
            return field
    return None


FAMILY_RULES = [
    ("certificate", r"certificate|\bcds?\b|share cert"),
    ("auto", r"auto loan|car loan|vehicle|\bauto\b|motorcycle|\bcar\b|truck loan|\btruck\b|\bsuvs?\b|minivan|\bvan\b|pickup|crossover"),
    ("atv", r"\batvs?\b|snowmobile|jet ?skis?|\butvs?\b|four[- ]?wheelers?|\bquads?\b|side[- ]by[- ]sides?|personal watercraft|off[- ]road vehicle"),
    ("boat_rv", r"\bboat\b|\brv\b|camper|motorhome"),
    ("heloc", r"heloc|home equity line|line of credit"),
    ("home_equity_loan", r"home equity loan|home equity"),
    ("home_improvement", r"home improvement"),
    ("mortgage", r"mortgage|home loan|house loan|\barm\b|adjustable|construction loan|fixed[- ]rate|30[- ]year|15[- ]year|20[- ]year|conventional|fha|\bva loan|usda|jumbo loan|home buy|homebuy"),
    ("personal_loan", r"personal loan|signature loan|unsecured loan"),
    ("credit_card", r"credit card|\bvisa\b|platinum|rewards card|balance transfer|purchase apr|cash advance"),
    ("student_loan", r"student loan|student|university|college loan|tuition"),
    ("checking", r"checking"),
    ("money_market", r"money market|investor'?s choice"),
    ("hsa", r"health savings|\bhsa\b"),
    ("ira", r"\bira\b|roth|traditional ira"),
    ("savings", r"savings|share account|holiday club|launchpad|bright|you name it|high[- ]yield"),
]

SAVINGS_PRODUCTS = [
    ("bright", "Bright Savings"), ("you name it", "You Name It Savings"), ("holiday", "Holiday Club"),
    ("launchpad", "Launchpad Savings"), ("high-yield", "High-Yield Savings"), ("high yield", "High-Yield Savings"),
    ("money market", "Money Market"), ("investor", "Money Market"), ("health savings", "Health Savings"), ("hsa", "Health Savings"),
    ("share savings", "Share Savings"), ("regular savings", "Share Savings"), ("basic savings", "Share Savings"),
]


def detect_family(ql: str) -> Optional[str]:
    for fam, pat in FAMILY_RULES:
        if re.search(pat, ql):
            return fam
    return None


# ----------------------------------------------------------------------------- fact ranking / grouping

def fact_rank(f: Fact) -> tuple:
    return (f.effective or "", f.authority, f.taken_on)


def group_id(f: Fact) -> tuple:
    biz = "business" in f.section.lower() or "/business" in f.url.lower()
    if f.family in ("auto", "atv"):
        return (f.family, biz, f"tier{f.attrs.get('tier_rank', 0)}", f.attrs.get("term"), f.metric)
    lab = norm_label(f.label)
    a = f.attrs
    if f.family in ("savings", "money_market", "hsa", "checking"):
        prod = row_product(f)
        tier = f"{a['bal_lo']}-{a['bal_hi']}" if "bal_lo" in a else norm_label(f.label.split("|")[-1])
        return (f.family, biz, prod, tier, f.metric)
    if f.family in ("certificate", "ira"):
        s_all = (f.section + " " + f.label).lower()
        ira = a.get("ira") or ""
        if ira == "any":
            ira = "ira"
        bits = [f"{a.get('term', '')}mo", "jumbo" if ("jumbo" in s_all) else "", "special" if ("special" in s_all) else "", ira]
        if "bal_lo" in a:
            bits.append(f"{a['bal_lo']}-{a['bal_hi']}")
        return (f.family, biz, " ".join(b for b in bits if b), f.metric)
    if f.family == "credit_card":
        lab = norm_label(f.section) + " " + lab
    return (f.family, biz, lab, f.metric)


def row_product(f: Fact) -> str:
    first = f.label.split("|")[0].strip()
    name = first if re.search(r"[a-zA-Z]{3,}", first) and "$" not in first else f.section
    name = name.lower()
    name = re.sub(r"\b(rates?|accounts?|account rates?)\b", " ", name)
    name = re.sub(r"[^a-z0-9]+", " ", name).strip()
    return name


def pick_rows(facts: list[Fact]) -> dict[tuple, dict]:
    """Group facts into rows -> {metric: best Fact}. Row id excludes metric."""
    groups: dict[tuple, list[Fact]] = defaultdict(list)
    for f in facts:
        groups[group_id(f)].append(f)
    rows: dict[tuple, dict] = defaultdict(dict)
    for gid, fs in groups.items():
        best = max(fs, key=fact_rank)
        rows[gid[:-1]][gid[-1]] = (best, [x for x in fs if x.value == best.value])
    return rows


def fmt_pct(v: str) -> str:
    v = v.strip()
    if "%" not in v and re.fullmatch(r"\d+(\.\d+)?", v):
        v += "%"
    return v


# ----------------------------------------------------------------------------- the system

class Dupaco:
    def __init__(self):
        self.kb = KnowledgeBase()
        self._cache: dict[tuple, Answer] = {}
        self._conflicts: Optional[list[Conflict]] = None
        self._definitive = False
        self._last_hit = None      # best sentence of the last h_retrieve call

    # ---- contract
    def ingest(self, snapshot: Snapshot) -> None:
        self.kb.ingest(snapshot)
        self._cache.clear()
        self._conflicts = None

    def answer(self, question: str, as_of: Optional[str] = None) -> Answer:
        key = (question.strip().lower(), as_of)
        if key in self._cache:
            return self._cache[key]
        try:
            self._definitive = False
            ans = self._answer(question, as_of)
            # Nothing found in the chosen snapshot: a page may simply have dropped out of a later crawl.
            # Fall back to earlier snapshots only when the refusal was not a definitive "product not listed".
            if not ans.confident and not self._definitive and as_of is None and len(self.kb.views) > 1:
                for v in reversed(self.kb.views[:-1]):
                    self._definitive = False
                    older = self._answer(question, v.taken_on)
                    if older.confident:
                        older.conflict_note = ((older.conflict_note + "; ") if older.conflict_note else "") + \
                            f"taken from the {v.snapshot_id} snapshot; the page was not present in the newest crawl"
                        ans = older
                        break
                    if self._definitive:
                        break
        except Exception:
            ans = Answer(text=REFUSAL, confident=False)
        self._cache[key] = ans
        return ans

    def conflicts(self) -> list[Conflict]:
        if self._conflicts is None:
            raw = self._build_conflicts()
            # identical value sets (personal/business/IRA copies of the same product) collapse into one entry
            merged: dict[tuple, Conflict] = {}
            for c in raw:
                k = (tuple(sorted(c.values)), c.resolution)
                if k in merged:
                    m = merged[k]
                    if any(w in m.topic.lower() for w in ("roth", "traditional", "business")) and not any(w in c.topic.lower() for w in ("roth", "traditional", "business")):
                        m.topic = c.topic
                    for src in c.sources:
                        if all(src.url != x.url for x in m.sources) and len(m.sources) < 8:
                            m.sources.append(src)
                else:
                    merged[k] = c
            self._conflicts = list(merged.values()) + self._text_conflicts()
        return self._conflicts

    def _text_conflicts(self) -> list[Conflict]:
        """Contradictions that live in prose rather than rate tables (e.g. branch counts in press releases)."""
        out = []
        if not self.kb.views:
            return out
        latest = self.kb.views[-1]
        # expired promotions still on the site that quote a different value than the current product
        current = [f for f in latest.facts + latest.prose if f.metric in ("apy", "apr", "rate") and not f.attrs.get("expired")]
        for pf in latest.prose:
            if not pf.attrs.get("expired") or pf.metric not in ("apy", "apr", "rate"):
                continue
            for cf in current:
                if cf.family == pf.family and cf.metric == pf.metric and cf.attrs.get("term") == pf.attrs.get("term") \
                        and bool(cf.attrs.get("jumbo")) == bool(pf.attrs.get("jumbo")) and cf.value != pf.value:
                    out.append(Conflict(topic=f"{kb.strip_footnotes(cf.label)} {pf.metric.upper()}", values=[pf.value, cf.value],
                                        sources=[Source(url=pf.url, snapshot_id=latest.snapshot_id, quote=pf.label), Source(url=cf.url, snapshot_id=latest.snapshot_id, quote=cf.label)],
                                        resolution=cf.value, reason="the promotional page is expired; the current rate page wins"))
                    break
        hits = []
        for p in latest.pages.values():
            for m in re.finditer(r"\b(\d\d) branch(?: office)? locations", p.text):
                hits.append((p.content_date or "", int(m.group(1)), p.url))
        if hits:
            vals = sorted({h[1] for h in hits})
            if len(vals) > 1:
                hits.sort(reverse=True)
                cur = hits[0][1]
                srcs = []
                for v in vals:
                    u = next(h[2] for h in hits if h[1] == v)
                    srcs.append(Source(url=u, snapshot_id=latest.snapshot_id, quote=f"{v} branch office locations"))
                out.append(Conflict(topic="number of Dupaco branch office locations", values=[f"{v} branches" for v in vals], sources=srcs,
                                    resolution=f"{cur} branches", reason="press releases from different years quote different counts; the most recently dated release is current"))
        return out

    # ---- dispatch
    def _answer(self, question: str, as_of: Optional[str]) -> Answer:
        info = parse_question(question)
        ql = info["ql"]
        view = self.kb.view_for(as_of or info.get("as_of_text"))
        if view is None:
            return Answer(text=REFUSAL, confident=False)

        if "routing" in ql:
            return self.h_routing(view)
        about = about_field(ql)
        if about:
            a = self.h_about(view, info, about)
            if a:
                return a
        if re.search(r"\bjoin\b|become a (?:dupaco )?member|membership", ql) and \
                re.search(r"^how\b|how (?:do|can|would) i|minimum|deposit|what do i need|requirements?|need to|cost|how much|steps?\b", ql) and \
                not re.search(r"\b(link|url|web ?address|webpage|web page)\b|eligib|who can|qualif", ql):
            a = self.h_join(view, info)
            if a:
                return a
        if re.search(r"\b(link|url|web ?address|webpage|web page)\b", ql) or \
                (re.search(r"\bjoin\b|become a member|sign up|open an account|membership application", ql) and re.search(r"\bonline\b|website|how (do|can) i|where", ql)):
            a = self.h_link(view, info)
            if a:
                return a
        prods = named_products(view, info)
        info["products"] = prods
        name = None if prods else product_name_phrase(ql)
        if name and not name_on_site(view, name, info["raw"]):
            self._definitive = True
            label = " ".join(w.capitalize() if i < len(name) - 1 else w for i, w in enumerate(name))
            text = (f"Dupaco's website doesn't list a {label} product. It isn't listed there, so please contact Dupaco "
                    "directly and a team member can help.") if not re.search(r"\d", label) else REFUSAL
            return Answer(text=text, confident=False)
        if prods and re.search(r"^(what is|what's|what are|tell me about|describe|explain)\b", ql) and not product_residual(info, prods):
            a = self.h_product(view, info, prods)
            if a:
                return a
        if re.search(r"eligib|who can (join|become)|qualify (to|for) (join|member)|requirements? to (join|become)|field of membership", ql) \
                and not (prods and not re.search(r"\bjoin|\bmember", ql)):
            a = self.h_eligibility(view, info)
            if a:
                return a
        if re.search(r"phone number|\bcall\b|telephone|\bnumber\b", ql) and (re.search(r"branch|office|location|learning lab", ql)
                                                                       or self._branch_city_named(view, ql)):
            a = self.h_branch_phone(view, info)
            if a:
                return a
        if re.search(r"contact (info|information|details)|phone number|customer service|member service|how (do|can) i (contact|reach|call|update my)|update my (address|phone|email|contact)|change my (address|phone|email)", ql):
            return self.h_contact(view, info)
        if re.search(r"how many (branch|location|office)", ql) or re.search(r"number of (branch|location)", ql):
            return self.h_branch_count(view)
        if self._holiday_hours_q(ql):
            a = self.h_holiday_hours(view, info)
            if a:
                return a
        if self._hours_q(ql):
            return self.h_hours(view, info)
        if re.search(r"\baddress(?:es)?\b|where is|where's|where are|where can i find|where\b[^?.]{0,20}\b(?:is|are)\b|located|location of", ql) and re.search(r"branch|office|headquarter|location|learning lab", ql):
            return self.h_address(view, info)
        if re.search(r"\b(is there|are there|do you have|does dupaco have|have (a|any))\b.*\b(branch|location|office|learning lab)", ql):
            return self.h_address(view, info)
        fam = detect_family(ql)
        if re.search(r"\bfees?\b|\bprices?\b|\bpriced\b|penalty|surcharge|how much (does|is|are|do you charge|will|would|to|per|for|a|an|each)\b|\bpay (?:to|for|anything|extra)\b|\bis it free\b|what does it cost|\bcosts?\b|\bcharged?\b(?:\s+[\w']+){0,3}\s+(?:for|to)\b|\bcharges?\b", ql) and not re.search(r"\baprs?\b|\bapys?\b|interest rate|\brates?\b", ql):
            a = self.h_prose(view, info, want_metric="fee")
            if a:
                return a
            info["fee_gate_failed"] = True
        if not prods:
            # a lower-case product the site never names ("reverse mortgage"): nothing to retrieve, say so
            # (after the fee schedule had its chance: "expedited debit card" is a fee row, not a product)
            for raw, t, head in product_qualifiers(info["raw"]):
                if not pair_on_site(view, t, head):
                    self._definitive = True
                    return Answer(text=self._not_listed_text(info["raw"], raw), confident=False)
        ql_np = strip_product_names(ql, prods) if prods else ql     # "1st Rate Checking" is a name, not a rate question
        rateish = re.search(r"\brates?\b|\baprs?\b|\bapys?\b|interest|percent|%|yield|dividend|how much|cost|charge|\bearn\b|pay on|paying|pays|points", ql_np)
        if prods and not _RATE_CUE_RE.search(ql_np):
            rateish = None
        # "what would my monthly payment be" asks for the payment example, not the rate table
        payment_q = re.search(r"\b(?:monthly )?payments? (?:be|would be|come to|look like)\b|\bmonthly payments?\b|\bpay (?:each|per|a) month\b", ql) \
            and not re.search(r"\brates?\b|\baprs?\b|\bapys?\b|interest rate", ql_np)
        nonrate = re.search(r"wallet|apple pay|google pay|samsung|mobile app|lost|stolen|activate|dispute|fraud|pin\b|limit|how (do|can) i (apply|open|pay|use)|statement|login|log in", ql)
        slotted = info.get("term") or info.get("model_year") or re.search(r"minimum|min\.? (to )?open|how much to open", ql)
        if payment_q:
            rateish = slotted = None
        if info.get("fee_gate_failed") and not re.search(r"\brates?\b|\baprs?\b|\bapys?\b|interest|yield|dividend", ql_np):
            rateish = slotted = None      # a fee or penalty question ("cash advance cost") is not a rate-table question
        if fam and fam != "student_loan" and (rateish or slotted) and not nonrate:
            gap = self._qualifier_gap(view, info, fam)
            if gap:
                return gap
            a = self.h_rates(view, info, fam)
            if a is not None:
                return a
        if info.get("fee_gate_failed"):
            # No fee-schedule label names what was asked: accept a retrieved passage only if it states a price and
            # names (most of) what was asked, counting the heading it sits under ("### Free Checking").
            qwords = fee_content_words(info["raw"])

            def priced(st, text):
                if not _PRICE_RE.search(text):
                    return False
                have = _with_aliases(set(tokenize(text)) | set(tokenize(st.section)))
                have |= {w for w in qwords if FEE_LABEL_ALIAS.get(w, set()) & have}
                return not qwords or 3 * len(qwords & have) >= 2 * len(qwords)
            return self.h_retrieve(view, info, accept=priced)
        a = self.h_retrieve(view, info)
        if not a.confident and not self._definitive:
            b = self.h_prose(view, info)
            if b and self._label_matches(info, b.text):
                return b
        return a

    @staticmethod
    def _not_listed_text(raw_q: str, modifier: str, rate_only: bool = False) -> str:
        """Refusal naming the product as the member wrote it ('reverse mortgages'); never any digits."""
        m = re.search(r"\b" + re.escape(modifier) + r"[\w-]*(?:\s+[a-z][\w-]*){0,2}?\s+(mortgages?|loans?|accounts?|cards?|certificates?|cds?|"
                      r"savings|checking|iras?|helocs?|insurance|clubs?)\b", raw_q, re.I)
        thing = m.group(0) if m else modifier
        if re.search(r"\d", thing):
            return REFUSAL
        if rate_only:
            return (f"Dupaco's website mentions {thing} but doesn't publish a rate for them. It isn't listed there, so please "
                    "contact Dupaco directly and a team member can help.")
        return (f"Dupaco's website doesn't list {thing}. It isn't listed there, so please contact Dupaco directly and a "
                "team member can help.")

    def _qualifier_gap(self, view: SnapshotView, info: dict, fam: str) -> Optional[Answer]:
        """A rate question for a product qualifier ('reverse mortgage', 'lot loan') that no rate row of the family
        carries: refuse instead of answering with the family's ordinary rows."""
        quals = product_qualifiers(info["raw"])
        if not quals:
            return None
        cache = view.__dict__.setdefault("_fam_vocab", {})
        if fam not in cache:
            fs = view.by_family.get(fam, []) + (view.by_family.get("ira", []) if fam == "certificate" else [])
            voc = set()
            for f in fs:
                voc |= set(tokenize(f.section + " " + f.label))
            voc |= set(tokenize(" ".join(re.findall(r"[a-z]+", dict(FAMILY_RULES).get(fam, "")))))
            cache[fam] = voc
        voc = cache[fam]
        for raw, t, head in quals:
            if t in voc or _variants(t) & voc:
                continue
            self._definitive = True
            return Answer(text=self._not_listed_text(info["raw"], raw, rate_only=pair_on_site(view, t, head)), confident=False)
        return None

    # ---- route triggers ---------------------------------------------------------------------------
    # "hours" alone is not a branch-hours question ("volunteer hours", "my hours got cut", "24 hours"): the word must
    # sit in opening-time context (lobby/drive-up/branch/business hours, "your hours", "hours on Saturday" ...).
    _HOURS_CTX_RE = re.compile(
        r"\b(?:lobby|drive[- ]?(?:up|thru|through)|branch(?:es)?|business|opening|office|operating|holiday|weekend|weekday|"
        r"saturday|sunday|location|bank|your|dupaco'?s?|their|its|the)\s+(?:\w+\s+)?hours\b"
        r"|\bhours\s+(?:of operation|on (?:the )?(?:weekends?|saturdays?|sundays?|weekdays?)|at (?:the|your|dupaco|a)\b|for (?:the|your|dupaco|a)\b[^?.]{0,40}\b(?:branch|office|location|learning lab|lobby|drive))"
        r"|^\s*hours\b|\bwhat are (?:the|your) hours\b|\bwhen are you open\b")
    _HOURS_NOT_RE = re.compile(r"\b(?:volunteer(?:ed|ing)?|my|our|his|her|work(?:ing)?|service|credit|class|contact)\s+hours\b|"
                               r"\bhow many hours\b|\b\d+\s*(?:-\s*)?hours?\b|\b24-hour|\bhours? (?:got|were|was|have been|are being) (?:cut|reduced)")
    _OPEN_TIME_RE = re.compile(r"what time\b[^?.]{0,40}\b(?:open|close|closes|closing)\b|\bopen (?:on|until|till|from)\b|\bclose at\b|closing time|opening time"
                               r"|\bwhat time does\b[^?.]{0,40}\b(?:branch|lobby|drive|office|dupaco)")

    @classmethod
    def _hours_q(cls, ql: str) -> bool:
        """A question about branch opening times (not about some other kind of hours)."""
        if cls._OPEN_TIME_RE.search(ql):
            return True
        if not re.search(r"\bhours?\b", ql) or cls._HOURS_NOT_RE.search(ql):
            return False
        return bool(cls._HOURS_CTX_RE.search(ql) or re.search(r"\b(?:branch|lobby|drive[- ]?up|learning lab|office|open|close[sd]?)\b", ql))

    _HOLIDAY_RE = re.compile(r"\b(?:thanksgiving|christmas|xmas|new year'?s|easter|memorial day|labor day|independence day|fourth of july|"
                             r"4th of july|july 4(?:th)?|presidents'? day|mlk day|martin luther king|veterans day|columbus day|juneteenth|"
                             r"black friday|good friday|holidays?)\b")
    _OPENING_RE = re.compile(r"\bopen\b|\bclosed?\b|\bclosing\b|\bhours\b|close early|\bwork(?:ing)? day")

    @classmethod
    def _holiday_hours_q(cls, ql: str) -> bool:
        """Opening on a holiday or around one ('the day after Thanksgiving', 'Christmas Eve')."""
        return bool(cls._HOLIDAY_RE.search(ql) and cls._OPENING_RE.search(ql)
                    and not re.search(r"holiday club|holiday (?:budget|spending|loan|savings)", ql))

    def h_holiday_hours(self, view: SnapshotView, info: dict) -> Optional[Answer]:
        """Holiday hours/closures: answer only from a sentence that names the holiday together with opening/closing;
        the regular weekly schedule or a branch's history is not an answer. Nothing published -> definitive refusal."""
        hol = {m.group(0) for m in self._HOLIDAY_RE.finditer(info["ql"])} - {"holiday", "holidays"}
        for st in view.sentences:
            tl = st.text.lower()
            if not re.search(r"\bclosed\b|\bwill close\b|\bclose early\b|\bopen\b|\bhours\b", tl):
                continue
            if (hol and any(h in tl for h in hol)) or (not hol and re.search(r"\bholiday (?:hours|closings?|schedule)|closed (?:on|for) (?:the |all )?(?:federal )?holidays?", tl)):
                if re.search(r"\bbranch|\blobby|\bdrive|\bdupaco|\boffices?\b|\bwe\b", tl):
                    return Answer(text=Dupaco._passage(view, st),
                                  sources=[Source(url=st.page.url, snapshot_id=view.snapshot_id, quote=st.text[:200])])
        self._definitive = True
        return Answer(text="Dupaco's website doesn't publish holiday hours or holiday closings. It isn't listed there, so please "
                           "contact Dupaco directly or check with your branch.", confident=False)

    def _branch_city_named(self, view: SnapshotView, ql: str) -> bool:
        """The question names a city that has a Dupaco branch ("call the Cedar Falls Dupaco")."""
        return any(re.search(r"\b" + re.escape(b["city"].lower()) + r"\b", ql) for b in view.branches if b.get("city"))

    # ---- handlers -------------------------------------------------------------------------------
    def h_routing(self, view: SnapshotView) -> Answer:
        cnt: Counter = Counter()
        srcs: dict[str, list[str]] = defaultdict(list)
        for p in view.pages.values():
            for m in re.finditer(r"routing (?:number|#)[^0-9]{0,40}(\d{9})", p.text, re.I):
                cnt[m.group(1)] += 1
                srcs[m.group(1)].append(p.url)
        if not cnt:
            return Answer(text=REFUSAL, confident=False)
        num = cnt.most_common(1)[0][0]
        urls = sorted(set(srcs[num]), key=lambda u: (-view.pages[u].authority, len(u)))[:6]
        return Answer(text=f"Dupaco's routing number is {num}.", sources=[Source(url=u, snapshot_id=view.snapshot_id) for u in urls])

    def h_link(self, view: SnapshotView, info: dict) -> Optional[Answer]:
        q = set(info["toks"]) - {"link", "url", "page", "web", "address", "find", "see", "use", "need", "want", "dupaco", "website", "online", "go", "i", "should", "how", "do", "where", "can"}
        ql = info["ql"]
        want_rates = bool(re.search(r"\brates?\b|apy|apr", ql))
        want_join = bool(re.search(r"\bjoin\b|become a member|sign up|membership|open an account", ql))
        best, best_s = None, 0.0
        for (anchor, url), cnt in view.links.items():
            if not url.startswith("http") or "dupaco" not in url:
                continue
            path = re.sub(r"^https?://", "", url).lower()
            if path.rstrip("/") in ("www.dupaco.com", "dupaco.com") or "tel:" in url or "mailto:" in url:
                continue
            ptoks = set(tokenize(re.sub(r"[/#?=_.-]", " ", path.split("?")[0]))) | set(tokenize(anchor))
            ctx = set()
            for pg in view.link_pages.get((anchor, url), ()):
                ctx |= set(tokenize(re.sub(r"[/#?=_.-]", " ", pg)))
            ov = len(q & ptoks)
            s = ov * 2.0 + 0.6 * len(q & ctx) + min(cnt, 30) / 10.0
            if "info.dupaco.com" in url or "wp-content" in url or "hubfs" in url or "/newsroom/" in url or "/learn/blog/" in url or "/tag/" in url:
                s -= 3
            if url.startswith("https://www.dupaco.com") or url.startswith("https://join.dupaco.com"):
                s += 0.5
            if want_join and "join.dupaco.com" in url and "login" not in url and "/product" not in url:
                s += 4
            if want_rates and ("#rates" in url or path.rstrip("/").endswith("/rates")):
                s += 3
            if want_rates and "rate" not in path and "rate" not in anchor:
                s -= 1.5
            if re.search(r"locator|find a (branch|location|atm)|branch|atm", ql) and "locator" in url:
                s += 4
            if ov == 0 and not any(k in url for k in ("join", "locator")):
                continue
            if s > best_s:
                best, best_s = (anchor, url), s
        if not best:
            return None
        anchor, url = best
        clean = url.split("?")[0] if "campaign" in url or "utm_" in url else url
        if "locator" in clean:
            clean = clean.split("#")[0]
        return Answer(text=f"You can use this link: {clean}", sources=[Source(url=clean.split('#')[0], snapshot_id=view.snapshot_id)])

    def h_prose(self, view: SnapshotView, info: dict, want_metric: Optional[str] = None, family: Optional[str] = None) -> Optional[Answer]:
        """Answer from 'label: value' prose facts (fees, rates quoted outside tables)."""
        q = set(info["toks"]) - {"dupaco", "much", "cost", "charge", "fee", "fees", "what", "how"}
        best, best_s = None, 0.0
        fee_mode = want_metric == "fee"
        qwords = fee_content_words(info["raw"]) if fee_mode else set()
        pscope = product_scope(info.get("products") or []) if fee_mode else set()
        purls = {u for d in (info.get("products") or []) for u in d["urls"]}
        if fee_mode:
            # Only words that occur in some fee label can tell fee rows apart; the rest ("daughter", "machine",
            # "studying") is the member's story. Coverage is measured over the distinctive words.
            vocab = self._fee_vocab(view)
            dwords = {w for w in qwords if w in vocab or FEE_LABEL_ALIAS.get(w, set()) & vocab}
            if not dwords:
                return None
        cands = []   # fee mode: (cover, gate score, old score, fact) for facts whose label passes the gate
        term = info.get("term")
        pool = list(view.prose) + [f for f in view.facts if f.family == "fee"]
        # a size or model in the question ("3x10 box") must be the label's size; "5 X 5 X 21" is another item
        qdims = set(re.findall(r"\b\d+x\d+\b", fee_normalize(info["ql"])))
        for f in pool:
            if want_metric and f.metric != want_metric:
                continue
            if family and f.family != family:
                continue
            if f.attrs.get("expired"):
                continue
            if qdims:
                ldims = set(re.findall(r"\b\d+x\d+\b", fee_normalize(f.label.lower())))
                if ldims and not qdims & ldims:
                    continue
            lt = set(tokenize(f.label))
            if not lt or not q:
                continue
            ov = len(q & lt) / len(q) - 0.04 * len(lt - q)
            if term and f.attrs.get("term") not in (None, term):
                continue
            if term and f.attrs.get("term") == term:
                ov += 0.3
            if info.get("metric") and f.metric != info["metric"] and f.metric in ("apy", "apr", "rate"):
                ov -= 0.15
            s = ov + f.authority / 400.0 + (0.1 if f.effective and f.effective >= "2026-01-01" else 0)
            if fee_mode:
                cov = fee_label_cover(qwords, f.label + " " + (f.attrs.get("detail") or ""))
                lw = _with_aliases(fee_content_words(re.sub(r"\([^)]*\)", " ", f.label)))
                # the product scope may come from the page the fee is printed on ("Cash advance fee" on the Visa terms)
                sw = lw | self._scope_words(f.section)
                # a named product's scope is also met by a fee printed on that product's own page (the Platinum
                # Visa's late fee sits in the credit-card page's fine print, not under a "Platinum" label)
                scope_ok = (qwords & FEE_SCOPE) <= (fee_content_words(f.label) | sw) and (pscope <= sw or f.url in purls)
                if cov and scope_ok:
                    core = re.sub(r"\([^)]*\)", " ", f.label)   # "(after 8 free per month)" is detail, not the item
                    extra = {w for w in fee_content_words(core) if w.isalpha() and len(w) > 1} \
                        - qwords - set().union(*(FEE_LABEL_ALIAS.get(w, set()) for w in cov))
                    sf = len(cov) / len(dwords) - 0.04 * len(extra) + f.authority / 400.0 \
                        + (0.1 if f.effective and f.effective >= "2026-01-01" else 0)
                    cands.append((cov, sf, s, len(extra), f))
                continue
            if s > best_s:
                best, best_s = f, s
        if fee_mode:
            if not info.get("business"):
                # a member's question is about personal accounts unless it says business
                cands = [c for c in cands if not self._is_business(c[4])] or cands
            best = self._pick_fee(cands)
            if best is None:
                return None
        elif not best or best_s < 0.5:
            return None
        unit = {"apy": " APY", "apr": " APR"}.get(best.metric, "")
        label = kb.strip_footnotes(best.label).rstrip(":")
        label = label[:1].upper() + label[1:]
        srcs = []
        for g in pool:
            if g.value == best.value and norm_label(g.label) == norm_label(best.label) and g.url not in srcs:
                srcs.append(g.url)
        if fee_mode:
            biz = bool(info.get("business"))
            srcs.sort(key=lambda u: ("/business" in u.lower()) != biz)
        others = {g.value for g in pool if norm_label(g.label) == norm_label(best.label) and g.metric == best.metric} - {best.value}
        note = f"other pages show {', '.join(sorted(others))} for this item; the newer/authoritative page wins" if others else None
        detail = best.attrs.get("detail") or ""
        # the cell's own wording ("for members, 5% of total for non-members"); numbers only when they price this same item
        detail = detail.replace("*", "").strip()
        detail = (" " + detail) if detail and (len(detail) < 60 and not re.search(r"\d", detail)
                                               or len(detail) < 100 and re.search(r"\d\s?%|\$\s?\d", detail) and re.match(r"(for|per|each|of|or)\b", detail)) else ""
        return Answer(text=f"{label}: {best.value}{unit}{detail}.", sources=[Source(url=u, snapshot_id=view.snapshot_id) for u in srcs[:4]], conflict_note=note)

    @staticmethod
    def _is_business(f: Fact) -> bool:
        return "business" in (f.label + " " + f.section).lower() or "/business" in f.url.lower()

    @staticmethod
    def _fee_vocab(view: SnapshotView) -> set:
        """Content words of every fee label in the snapshot (built once per view)."""
        v = getattr(view, "_fee_vocab_cache", None)
        if v is None:
            v = set()
            for f in list(view.prose) + view.by_family.get("fee", []):
                if f.metric == "fee":
                    v |= fee_content_words(f.label + " " + (f.attrs.get("detail") or ""))
            view._fee_vocab_cache = v
        return v

    @staticmethod
    def _scope_words(section: str) -> set:
        """Product-scope words of the page/section a fee is printed on; Visa and credit card pages count as both."""
        sec = re.sub(r"[-_/.]+", " ", re.sub(r"\|\s*Dupaco.*$", "", section or ""))
        sec = re.sub(r"\bcredit unions?\b", " ", sec, flags=re.I)
        w = fee_content_words(sec) & (FEE_SCOPE | {kb.stem("card")})
        if w & {kb.stem("visa"), kb.stem("credit")}:
            w |= {kb.stem("visa"), kb.stem("credit")}
        return w

    @staticmethod
    def _pick_fee(cands: list) -> Optional[Fact]:
        """Prefer the label covering the most question content words. Refuse (None) when it is a weak match or when
        two labels with different prices each cover a different, equally large part of the question
        ("overdraft protection": "Consumer overdrafts" vs "Identity Theft Protection") and neither is named in full."""
        if not cands:
            return None
        # Most question words covered first; among equals the long-standing prose score decides (it also
        # prefers the clean PDF over the OCR'd copy). The gate score only sets the bar.
        # A label the question names in full ("foreign transaction fee") beats one with words of its own.
        cands.sort(key=lambda c: (-len(c[0]), c[3] > 0, -c[2], -c[1]))
        cov, score, _, n_extra, best = cands[0]
        if score < 0.5:
            return None
        for cov2, _, _, n_extra2, g in cands[1:]:
            if len(cov2) < len(cov):
                break
            if not (cov2 & cov) and g.value != best.value and not (n_extra == 0 and n_extra2 > 0):
                return None
        return best

    def h_about(self, view: SnapshotView, info: dict, field: str) -> Optional[Answer]:
        """Facts about Dupaco itself (size, history, leadership) from the newest press-release boilerplate."""
        ab = view.about
        d = ab.get(field)
        if not d:
            return None
        about_pages = [u for u in view.pages if view.pages[u].path == "/about"]
        if field == "ceo":
            text = f"{d['value']} is Dupaco's President and CEO."
        elif field == "members":
            text = f"Dupaco has {d['detail'] or 'about'} {d['value']} members".replace("  ", " ")
            assets = ab.get("assets")
            if assets and assets["url"] in d["urls"][:1] + [d["url"]]:
                text += f", with assets exceeding {assets['value']}"
            text += "."
        elif field == "employees":
            text = f"Dupaco has {d['detail'] or 'about'} {d['value']} employees.".replace("  ", " ")
        elif field == "counties":
            text = f"Dupaco serves residents in {d['value']} counties throughout {d['detail']}."
        elif field == "assets":
            text = f"Dupaco has assets exceeding {d['value']}."
        elif field == "hq":
            text = f"Dupaco Community Credit Union is headquartered in {d['value']}."
        elif field == "founded":
            text = f"Dupaco was founded in {d['value']}" + (f" by {d['detail']}" if d["detail"] else "") + "."
        else:   # recognition: what the most relevant story says, plus the standing list from the boilerplate
            text = f"Dupaco is a {d['value']}."
            urls = [d["url"]]
            # a named awarding body ("Forbes"): also give what the newest story about it says
            names = [w for w in re.findall(r"(?<!^)\b[A-Z][A-Za-z]+\b", info["raw"].strip()) if w.lower() not in ("dupaco", "credit", "union", "community", "i")]
            r = self.h_retrieve(view, info) if names else None
            if r and r.confident and all(n.lower() in r.text.lower() for n in names):
                text = r.text.rstrip() + " " + text
                urls = [s.url for s in r.sources] + urls
            return Answer(text=text, sources=[Source(url=u, snapshot_id=view.snapshot_id) for u in dict.fromkeys(urls + about_pages)][:6])
        urls = list(dict.fromkeys([d["url"]] + d["urls"][:3] + about_pages))
        return Answer(text=text, sources=[Source(url=u, snapshot_id=view.snapshot_id) for u in urls][:6])

    def h_product(self, view: SnapshotView, info: dict, prods: list[dict]) -> Optional[Answer]:
        """'What is <product>?': the product page's own description (the sentences under its heading)."""
        d = prods[0]
        urls = sorted(d["urls"], key=lambda u: (-view.pages[u].authority, len(u)))
        plist = view.page_sents.get(urls[0], [])
        sec = [st for st in plist if st.section == d["name"]]
        parts: list[str] = []
        for st in sec:
            t = st.text.strip()
            if t.endswith("?") or len(t) < 30 or _FOOTNOTE_SENT_RE.search(t):
                continue
            if sum(len(x) for x in parts) + len(t) > 480 and parts:
                break
            parts.append(t.lstrip("-* ").strip())
        if not parts:
            return None
        text = f"{d['name']}: " + " ".join(p if p.endswith((".", "!", ")")) else p + "." for p in parts)
        return Answer(text=re.sub(r"\s+", " ", text), sources=[Source(url=u, snapshot_id=view.snapshot_id) for u in urls[:4]])

    _JOIN_SENT_RE = re.compile(r"\b(?:to (?:join|become a member)|membership starts)\b[^.]*?\$\s?\d", re.I)

    def h_join(self, view: SnapshotView, info: dict) -> Optional[Answer]:
        """How to become a member: the join page's 'open a savings account with $X' sentence plus the online application link."""
        best = None
        for st in view.sentences:
            if st.page.kind in ("newsroom", "blog", "calculator", "landing") or "/resources/" in st.page.path:
                continue
            if self._JOIN_SENT_RE.search(st.text):
                key = (st.page.path.endswith("/join"), st.page.authority, -st.pos)
                if best is None or key > best[0]:
                    best = (key, st)
        if best is None:
            return None
        st = best[1]
        text = st.text.strip().lstrip("-* ")
        srcs = [st.page.url]
        link = self.h_link(view, parse_question("join dupaco online"))
        if link and "join" in link.sources[0].url:
            url = link.text.split("link:")[-1].strip()
            text += f" You can join online at {url}"
            srcs.append(link.sources[0].url)
        return Answer(text=text, sources=[Source(url=u, snapshot_id=view.snapshot_id) for u in srcs])

    def h_eligibility(self, view: SnapshotView, info: dict) -> Optional[Answer]:
        for p in sorted(view.pages.values(), key=lambda x: (not x.path.endswith("/join"), -x.authority)):
            if "eligible to join" not in p.text.lower() and "eligible" not in p.text.lower():
                continue
            lines = [l.strip("#*- ").strip() for l in p.text.split("\n") if l.strip()]
            bullets = []
            states = ""
            for i, l in enumerate(lines):
                if re.search(r"eligible to join|who can join|eligib", l, re.I) and len(l) < 90:
                    for l2 in lines[i + 1:i + 6]:
                        if len(l2) < 160 and not l2.endswith("?") and not re.search(r"eligible to join", l2, re.I):
                            bullets.append(l2.rstrip("."))
                        else:
                            break
                if re.search(r"Iowa, Illinois and Wisconsin|Iowa, Illinois, and Wisconsin", l):
                    states = l
            if bullets:
                text = "You're eligible to join Dupaco if any of these apply: " + "; ".join(bullets[:4]) + "."
                if states:
                    text += " " + states.rstrip(".") + "."
                return Answer(text=text, sources=[Source(url=p.url, snapshot_id=view.snapshot_id)])
        return None

    def h_contact(self, view: SnapshotView, info: dict) -> Answer:
        cnt: Counter = Counter()
        urls: dict[str, list[str]] = defaultdict(list)
        for p in view.pages.values():
            for m in re.finditer(r"\b(800-\d{3}-\d{4}|\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4})\b", p.text):
                num = m.group(1)
                if num.startswith("800"):
                    cnt[num] += 1
                    urls[num].append(p.url)
        if not cnt:
            return self.h_retrieve(view, info)
        phone = cnt.most_common(1)[0][0]
        ql = info["ql"]
        if re.search(r"update|change|contact (info|information|details)", ql):
            text = (f"You can update your contact information yourself by logging in to Shine Online or Mobile Banking, "
                    f"or call Dupaco at {phone} and a team member will update it for you.")
        else:
            text = f"You can reach Dupaco at {phone}, or use the Contact Us option inside Shine Online or Mobile Banking."
        src = sorted(set(urls[phone]), key=lambda u: (not u.rstrip("/").endswith("/contact"), -view.pages[u].authority))[:4]
        return Answer(text=text, sources=[Source(url=u, snapshot_id=view.snapshot_id) for u in src])

    def h_branch_count(self, view: SnapshotView) -> Answer:
        hits = []
        for p in view.pages.values():
            for m in re.finditer(r"\b(\d\d) branch(?: office)? locations", p.text):
                hits.append((p.content_date or "", p.authority, int(m.group(1)), p.url))
        if not hits:
            return self.h_retrieve(view, parse_question("how many branches"))
        hits.sort(reverse=True)
        n = hits[0][2]
        urls = [h[3] for h in hits if h[2] == n][:4]
        for u in view.pages:
            if u.rstrip("/").endswith("/contact"):
                urls.append(u)
        return Answer(text=f"Dupaco has {n} branch office locations across Iowa, Illinois and Wisconsin.",
                      sources=[Source(url=u, snapshot_id=view.snapshot_id) for u in urls])

    _HOURS_RE = re.compile(r"\*\*Lobby\*\*\s*(.+?)\s*\*\*Drive[- ]?up\*\*\s*(.+?)(?=\*\*[A-Z]|$|\n)", re.S)

    def _hours_blocks(self, view: SnapshotView) -> list[tuple[str, str, str, str]]:
        out = []
        for p in view.pages.values():
            for m in self._HOURS_RE.finditer(p.text):
                lobby = re.sub(r"\s+", " ", m.group(1)).strip().replace("–", "-").replace("—", "-")
                drive = re.sub(r"\s+", " ", m.group(2)).strip().replace("–", "-").replace("—", "-")
                if "a.m." in lobby or "p.m." in lobby:
                    out.append((p.content_date or "", lobby, drive, p.url))
        out.sort(reverse=True)
        return out

    def h_hours(self, view: SnapshotView, info: dict) -> Answer:
        ql = info["ql"]
        blocks = self._hours_blocks(view)
        # a specific branch named?
        branch = self._branch_in_question(view, info)
        if branch == "__unknown__":
            self._definitive = True
            return Answer(text="I couldn't find a Dupaco branch by that name on the website, so I can't confirm its hours. "
                               "Please contact Dupaco or check the branch locator on the Contact page.", confident=False)
        want_drive = "drive" in ql
        want_lobby = "lobby" in ql or not want_drive
        if branch:
            return self._branch_hours(view, branch, want_lobby, want_drive or "lobby" not in ql)
        if not blocks:
            return self.h_retrieve(view, info)
        lobby_sched = []
        drive_sched = []
        for _, l, d, u in blocks:
            if l not in lobby_sched:
                lobby_sched.append(l)
            if d not in drive_sched:
                drive_sched.append(d)
        srcs = [Source(url=b[3], snapshot_id=view.snapshot_id) for b in blocks[:4]]
        for u in view.pages:
            if u.rstrip("/").endswith("/contact") or u.rstrip("/") == "https://www.dupaco.com":
                srcs.append(Source(url=u, snapshot_id=view.snapshot_id))
        parts = []
        if want_lobby:
            extra = lobby_sched[1:4]
            parts.append("Lobby hours vary by branch. The most common schedule is " + lobby_sched[0] +
                         (" Other branches keep " + " / or ".join(extra) if extra else ""))
        if want_drive or not want_lobby:
            parts.append("Drive-up hours at most branches are " + drive_sched[0] +
                         (" Some branches keep " + drive_sched[1] if len(drive_sched) > 1 else ""))
        tail = " Let me know which branch you mean (or all branches) and I can be specific." if branch is None else ""
        return Answer(text=" ".join(parts) + tail, sources=srcs)

    # ---- branch table (built per snapshot in kb.extract_branches) ---------------------------------
    @staticmethod
    def _branch_addr(b: dict) -> str:
        """'807 Ninth St. SE, Dyersville, IA' / 'W. Main St. in Manchester, IA' (street only on the site)."""
        place = b["city"] + (f", {b['state']}" if b["state"] else "") + (f" {b['zip']}" if b["zip"] else "")
        if re.match(r"\d+\s", b["street"]):
            return f"{b['street']}, {place}"
        return f"{b['street']} in {place}"

    @staticmethod
    def _branch_at(b: dict) -> str:
        """'located at 807 Ninth St. SE, ...' or, when the site gives only the street name, 'on W. Main St. in ...'."""
        addr = Dupaco._branch_addr(b)
        if re.match(r"\d+\s", b["street"]):
            return "located at " + addr
        return "on " + addr + " (the website gives the street but not a street number)"

    @staticmethod
    def _branch_label(b: dict) -> str:
        return b["name"].replace("(", "").replace(")", "")

    @staticmethod
    def _branch_closed(view: SnapshotView, b: dict) -> bool:
        return bool(b["closed"]) and (not b["closed_on"] or b["closed_on"] <= view.taken_on)

    def _branch_sources(self, view: SnapshotView, bs: list[dict], first: Optional[list[str]] = None) -> list[Source]:
        urls: list[str] = list(first or [])
        for b in bs:
            for u in sorted(b["sources"], key=lambda u: (not u.rstrip("/").endswith("/contact"), -view.pages[u].authority if u in view.pages else 0)):
                if u not in urls:
                    urls.append(u)
        # newsroom stories that print the branch's street address (opening plans, ribbon cuttings)
        for b in bs[:1]:
            m = re.match(r"(\d+\s+\S+\s+\S+)", b["street"])
            if not m:
                continue
            for u, pg in sorted(view.pages.items(), key=lambda kv: kv[1].content_date or "", reverse=True):
                if pg.kind == "newsroom" and u not in urls and m.group(1) in pg.text:
                    urls.insert(min(len(urls), 2), u)
        for u in view.pages:
            if u.rstrip("/").endswith("/contact") and u not in urls:
                urls.append(u)
        return [Source(url=u, snapshot_id=view.snapshot_id) for u in urls[:6]]

    def _branch_in_question(self, view: SnapshotView, info: dict):
        found = self._branch_in_question_raw(view, info)
        if isinstance(found, list) and len(found) > 1 and len({b["city"] for b in found}) == 1:
            # "on Madison's east side" -> the branch on E. Washington Ave.
            cm = re.search(r"\b(east|west|north|south)(?:ern)?(?:[\s-]+(?:side|end|part))\b", info["ql"])
            if cm:
                letter = cm.group(1)[0]
                narrowed = [b for b in found if letter in kb.street_tokens(b["street"])[:3]]
                if narrowed:
                    return narrowed
        return found

    def _branch_in_question_raw(self, view: SnapshotView, info: dict):
        """Branches named in the question: a list of branch-table rows, '__unknown__' when the question names a
        branch that is not in the table, or None when no particular branch is named."""
        ql = info["ql"]
        q = info.get("raw", "")
        if re.search(r"hy-?vee|headquarter|mailing address|p\.?o\.? box", ql):
            return None                       # Hy-Vee questions go to retrieval; HQ/mailing is not a branch
        names = []
        m = re.search(r"\b([A-Z][\w'.$-]*(?:\s+(?:[A-Z][\w'.-]*|of|the|&))*)\s+(?:[Bb]ranch|[Oo]ffice|[Ll]ocation|[Ll]earning [Ll]ab)\b", q)
        if m:
            names.append(m.group(1))
        else:
            m = re.search(r"\b((?:[a-z0-9'.-]+\s+){1,3}?)(?:branch|office|location|learning lab)\b", ql)
            if m:
                names.append(m.group(1))
        m = re.search(r"\b(?:branch|office|location|learning lab)(?:es|s)?\s+(?:in|at|on|near)\s+(?:the\s+)?([A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){0,3})", q)
        if m:
            names.append(m.group(1))
        drop = {"dupaco", "dupaco's", "the", "your", "our", "a", "an", "my", "new", "nearest", "closest", "main", "general", "all", "each", "every", "any",
                "is", "does", "do", "which", "what", "of", "at", "for", "is", "are", "to", "in", "credit", "union", "can", "you", "give", "me", "tell",
                "lobby", "drive-up", "drive", "up", "hours", "address", "open", "closed", "still", "today", "when", "where", "how", "many", "i", "with", "members", "member",
                "iowa", "ia", "wis", "wisconsin", "illinois", "ill", "il", "there", "have", "has", "local", "nearby", "full-service", "learning", "lab", "s",
                "find", "near", "me", "get", "go", "see", "visit", "use", "closest", "a", "dupacos"}
        toks = [t.lower().strip(".,?'’") for n in names for t in n.split()]
        toks = [t for t in toks if t and t not in drop]
        cand = {t for t in kb.street_tokens(" ".join(toks)) if t not in kb.STREET_GENERIC and t not in drop}
        branches = view.branches
        vocab = {id(b): set(b["city"].lower().split()) | set(b["key"]) | set(b["alias_key"]) for b in branches}
        # street / alias words anywhere in the question ("Which Dubuque branch is on Flint Hill?") narrow a city's
        # branches to the one the member named; the other branches' addresses would be wrong answers
        qst = {t for t in kb.street_tokens(re.sub(r"[^\w\s.'-]", " ", ql)) if t not in kb.STREET_GENERIC and t not in drop and len(t) > 2}

        def narrow(rows):
            by_street = [b for b in rows if (qst - {w for x in rows for w in x["city"].lower().split()}) & (set(b["key"]) | set(b["alias_key"]))]
            return by_street if by_street and len(by_street) < len(rows) else rows
        if cand:
            full = [b for b in branches if cand <= vocab[id(b)]]
            if full:
                return narrow(full) if len(full) > 1 else full
        # a branch city named anywhere in the question ("Is there a Dupaco in Dyersville?")
        cities = sorted({b["city"] for b in branches}, key=len, reverse=True)
        hit = next((c for c in cities if re.search(r"\b" + re.escape(c.lower()) + r"\b", ql)), None)
        if hit:
            inc = [b for b in branches if b["city"] == hit]
            narrowed = [b for b in inc if cand & (set(b["key"]) | set(b["alias_key"]))]
            return narrowed or narrow(inc)
        if cand:
            return "__unknown__"
        return None

    def h_address(self, view: SnapshotView, info: dict) -> Answer:
        branch = self._branch_in_question(view, info)
        if branch == "__unknown__":
            self._definitive = True
            return Answer(text="I couldn't find a Dupaco branch by that name on the website. Please contact Dupaco or use the branch locator on the Contact page.", confident=False)
        if not branch:
            return self.h_retrieve(view, info)
        yes = "Yes. " if re.match(r"\s*(is|are|do|does)\b", info["ql"]) else ""
        open_b = [b for b in branch if not self._branch_closed(view, b)]
        if not open_b:
            b = branch[0]
            return Answer(text=f"Dupaco's {self._branch_label(b)} branch at {self._branch_addr(b)} has closed. "
                               f"From Dupaco's newsroom: {b['closed']}", sources=self._branch_sources(view, [b]))
        if len(open_b) == 1:
            b = open_b[0]
            text = f"{yes}Dupaco's {self._branch_label(b)} branch is {self._branch_at(b)}."
            if b["phone"]:
                text += f" Its phone number is {b['phone']}."
            if b["closed"]:
                text += f" Note from Dupaco's newsroom: {b['closed']}"
            return Answer(text=text, sources=self._branch_sources(view, open_b))
        city = open_b[0]["city"]
        one_city = all(b["city"] == city for b in open_b)
        if one_city and not re.search(r"\bbranches\b|\blocations\b|\boffices\b|\ball\b|\bevery\b|\beach\b|\blist\b|how many|\bboth\b|\bany\b|\bwhich\b", info["ql"]):
            # one branch asked for, several in the city: give one address; another branch's address would be a wrong
            # answer for this member, so the others are named by street or alias only
            b = max(open_b, key=lambda r: (r["street_rank"], len(r["sources"]), -open_b.index(r)))
            others = [x for x in open_b if x is not b]
            names = [re.sub(r"^\d+\s+", "", x["aliases"][0] if x["aliases"] else x["street"]) for x in others]
            names = [n for n in names if n and not re.search(r"\d", n)]
            text = f"{yes}Dupaco's {self._branch_label(b)} branch is {self._branch_at(b)}."
            if names:
                text += f" Dupaco has {len(open_b)} branches in {city}; the other" + (" is" if len(names) == 1 else "s are") + \
                        " on " + ", ".join(names) + "."
            return Answer(text=text, sources=self._branch_sources(view, [b] + others))
        if one_city:
            head = f"{yes}Dupaco has {len(open_b)} branches in {city}: "
        else:
            head = f"{yes}Matching Dupaco branches: "
        items = []
        for b in open_b:
            lab = b["aliases"][0] if b["aliases"] else (b["city"] if b["city"] != city else "")
            items.append((f"{lab}: " if lab else "") + self._branch_addr(b))
        return Answer(text=head + "; ".join(items) + ".", sources=self._branch_sources(view, open_b))

    def _branch_hours(self, view: SnapshotView, branch: list[dict], want_lobby: bool, want_drive: bool) -> Answer:
        open_b = [b for b in branch if not self._branch_closed(view, b)]
        if not open_b:
            b = branch[0]
            return Answer(text=f"Dupaco's {self._branch_label(b)} branch has closed, so it no longer keeps branch hours. "
                               f"From Dupaco's newsroom: {b['closed']}", sources=self._branch_sources(view, [b]))
        with_h = [b for b in open_b if b["lobby"]]
        if not with_h:
            self._definitive = True
            lab = self._branch_label(open_b[0]) if len(open_b) == 1 else open_b[0]["city"]
            if re.search(r"\d", lab):
                lab = open_b[0]["city"]
            return Answer(text=f"The website doesn't post hours for Dupaco's {lab} branch, so I can't confirm them. "
                               "Please contact Dupaco or check the branch locator on the Contact page.", confident=False)
        parts = []
        same = len({(b["lobby"], b["drive_up"]) for b in with_h}) == 1
        groups = [with_h] if same else [[b] for b in with_h]
        for g in groups:
            b = g[0]
            who = (f"Dupaco's {open_b[0]['city']} branches" if len(g) > 1 else f"Dupaco's {self._branch_label(b)} branch ({self._branch_addr(b)})")
            seg = []
            if want_lobby:
                seg.append(f"lobby hours are {b['lobby'].rstrip('.')}.")
            if want_drive:
                seg.append(("Drive-up" if seg else "drive-up") + f" hours are {b['drive_up'].rstrip('.')}.")
            parts.append(f"At {who}, " + " ".join(seg))
        missing = [b for b in open_b if not b["lobby"]]
        if missing:
            parts.append(f"The website doesn't post hours for the other {open_b[0]['city']} branch" + ("es." if len(missing) > 1 else "."))
        return Answer(text=" ".join(parts), sources=self._branch_sources(view, with_h, first=[b["hours_url"] for b in with_h]))

    def h_branch_phone(self, view: SnapshotView, info: dict) -> Optional[Answer]:
        branch = self._branch_in_question(view, info)
        if not isinstance(branch, list):
            return None
        bs = [b for b in branch if b["phone"] and not self._branch_closed(view, b)]
        if not bs:
            return None
        text = " ".join(f"The phone number for Dupaco's {self._branch_label(b)} branch is {b['phone']}." for b in bs[:3])
        return Answer(text=text, sources=self._branch_sources(view, bs))

    # ---- structured rates ----------------------------------------------------------------------
    def h_rates(self, view: SnapshotView, info: dict, fam: str) -> Optional[Answer]:
        ql = info["ql"]
        facts = list(view.by_family.get(fam, []))
        if fam == "ira" or (fam == "certificate" and info["ira"]):
            facts = [f for f in view.by_family.get("ira", []) + view.by_family.get("certificate", []) if f.attrs.get("ira")]
            fam = "ira"
        if fam == "savings":
            prod = None
            for k, name in SAVINGS_PRODUCTS:
                if k in ql:
                    prod = name
                    break
            if prod is None:
                prod = "Share Savings"
            if prod == "Money Market":
                facts = list(view.by_family.get("money_market", []))
            elif prod == "Health Savings":
                facts = list(view.by_family.get("hsa", []))
            else:
                facts = [f for f in facts if prod.lower() in f.section.lower() or prod.lower() in f.label.lower()]
        if fam == "checking":
            if re.search(r"1st rate|first rate", ql):
                facts = [f for f in facts if "1st" in f.section.lower() or "1st" in f.label.lower()]
            elif "free checking" in ql:
                facts = [f for f in facts if "free" in f.section.lower() or "free" in f.label.lower()]
        if fam in ("certificate", "ira"):
            if not info["ira"] and fam == "certificate":
                facts = [f for f in facts if not f.attrs.get("ira")]
            if info["jumbo"]:
                facts = [f for f in facts if f.attrs.get("jumbo")]
            elif re.search(r"\b(standard|regular|non-jumbo)\b", ql) or "jumbo" in info.get("negated", ()):
                facts = [f for f in facts if not f.attrs.get("jumbo")]
            if info["special"] and not info.get("term"):
                facts = [f for f in facts if f.attrs.get("special")]
            if info.get("negated", set()) & {"special", "promo", "promotional"}:
                facts = [f for f in facts if not f.attrs.get("special")]
            if re.search(r"\broth\b", ql):
                facts = [f for f in facts if f.attrs.get("ira") in ("roth", "any")]
            elif "traditional" in ql:
                facts = [f for f in facts if f.attrs.get("ira") in ("traditional", "any")]
            if re.search(r"\b(regular|standard) (certificate|cd)", ql):
                facts = [f for f in facts if not f.attrs.get("special")]
        if fam == "credit_card":
            if info["business"]:
                facts = [f for f in facts if "business" in f.section.lower() or "business" in f.label.lower()]
            elif "platinum" in ql:
                facts = [f for f in facts if "platinum" in f.section.lower() and "business" not in f.section.lower()]
            elif "reward" in ql:
                facts = [f for f in facts if "reward" in f.section.lower() and "business" not in f.section.lower()]
            if not re.search(r"intro|balance transfer|promotional", ql):
                facts = [f for f in facts if not f.attrs.get("intro")]
        if fam == "mortgage":
            if re.search(r"\barm\b|adjustable", ql):
                facts = [f for f in facts if re.search(r"\barm\b|adjustable", (f.section + " " + f.label).lower())]
                if not facts:
                    self._definitive = True
                    return Answer(text="Dupaco's website does not publish a rate for adjustable-rate mortgages. It isn't listed, so please contact a Dupaco mortgage lender for current ARM pricing.", confident=False)
            elif "construction" in ql:
                facts = [f for f in facts if "construction" in (f.section + " " + f.label).lower()]
                if not facts:
                    self._definitive = True
                    return Answer(text="The website describes construction loans but does not publish a construction-loan rate. It isn't listed, so please contact a Dupaco mortgage lender.", confident=False)
            else:
                facts = [f for f in facts if "fixed conventional" in f.label.lower() or "fixed" in f.label.lower()]
                facts = [f for f in facts if f.attrs.get("mtg") == ("refinance" if info["refi"] else "purchase")]
                facts = [f for f in facts if "home equity" not in f.section.lower()]
        # business filter for generic deposit/loan families
        if fam in ("certificate", "checking", "savings", "money_market", "personal_loan", "auto", "ira"):
            if info["business"]:
                facts = [f for f in facts if "business" in f.section.lower() or "/business" in f.url.lower()]
            else:
                facts = [f for f in facts if not ("business" in f.section.lower() or "/business" in f.url.lower())]
        if not facts:
            a = self.h_prose(view, info, family=fam)
            return a or self._no_rate(fam, info)

        # term filter
        term = info.get("term")
        if term:
            tf = [f for f in facts if f.attrs.get("term") == term]
            if not tf and fam in ("auto", "personal_loan", "certificate", "ira", "mortgage", "home_equity_loan", "boat_rv", "atv", "home_improvement"):
                return self._no_rate(fam, info)
            if tf:
                facts = tf
        # model year
        my = info.get("model_year")
        if fam in ("auto", "atv"):
            if my:
                mf = [f for f in facts if f.attrs.get("model_lo", 0) <= my <= f.attrs.get("model_hi", 9999)]
                if mf:
                    facts = mf
            else:
                # newest tier unless the question says older/used
                if not re.search(r"\bolder\b|\bused\b|\b20(1\d|2[0-3])\b", ql):
                    facts = [f for f in facts if f.attrs.get("tier_rank", 0) == 0] or facts
        # balance / loan-amount tier
        amts = info.get("amounts") or []
        tiered = [f for f in facts if "bal_lo" in f.attrs]
        if tiered and amts and fam not in ("savings", "money_market", "hsa", "checking", "certificate", "ira"):
            top = max(f.attrs["bal_hi"] for f in tiered)
            if top < 1e9 and max(amts) > top:
                # the amount is above the largest loan size any published rate covers: the table's rates are not
                # this loan's rate (a jumbo mortgage above the conforming limit)
                self._definitive = True
                what = {"mortgage": "mortgage", "heloc": "home equity line of credit", "home_equity_loan": "home equity loan"}.get(fam, "loan")
                return Answer(text=f"Dupaco's published {what} rates only cover loan amounts up to a set limit, and the amount you "
                                   f"mention is above it. The rate for a larger loan isn't listed there, so please contact Dupaco's "
                                   f"lending team directly and a team member can help.", confident=False)
        if tiered and amts:
            a0 = min(amts)
            tf = [f for f in tiered if f.attrs["bal_lo"] <= a0 <= f.attrs["bal_hi"]]
            if fam == "mortgage":
                # loan "or less" tiers: pick the smallest cap that covers the amount
                caps = sorted({f.attrs["bal_hi"] for f in tiered if f.attrs["bal_hi"] >= max(amts)})
                tf = [f for f in tiered if caps and f.attrs["bal_hi"] == caps[0]]
            if tf:
                facts = tf
        elif tiered and fam in ("savings", "money_market", "hsa", "checking") and not info["superlative"] \
                and not re.search(r"\btiers?\b|\brates\b|\bapys\b|\byields\b|\ball\b|\bany\b|\beach\b|\bevery\b|\blist\b|\bbalances\b|\branges?\b|\bcompare\b", ql):
            # No balance given for a tiered account: listing every tier puts other tiers' rates (wrong for this member)
            # in the answer. Give the first tier that earns, per product; the rest goes to the conflict note.
            facts, info["tier_note"] = self._default_tier(facts)
        rows = pick_rows(facts)
        if not rows:
            return self._no_rate(fam, info)
        # superlative on deposits: keep row with max APY
        if info["superlative"] and fam in ("savings", "money_market", "hsa", "checking", "certificate", "ira") and len(rows) > 1:
            def apy_of(r):
                v = r.get("apy") or r.get("dividend_rate") or r.get("rate")
                try:
                    return float(v[0].value.replace("%", "").split()[0]) if v else -1
                except Exception:
                    return -1
            top = max(rows.values(), key=apy_of)
            rows = {k: v for k, v in rows.items() if v is top}
        # mortgage default: when no term and not a tier question, show 30/20/15 at the smallest-loan tier plus the 30-year range
        if fam == "mortgage" and not term and not amts:
            rows = self._mortgage_headline(rows)
        return self._compose(view, fam, rows, info)

    @staticmethod
    def _default_tier(facts: list[Fact]) -> tuple[list[Fact], str]:
        def earns(f):
            try:
                return float(re.sub(r"[^\d.]", "", f.value.split()[0]) or 0) > 0
            except ValueError:
                return False
        by_prod: dict[str, list[Fact]] = defaultdict(list)
        for f in facts:
            by_prod[row_product(f)].append(f)
        keep, dropped = [], []
        for prod, fs in by_prod.items():
            tiers = sorted({f.attrs["bal_lo"] for f in fs if "bal_lo" in f.attrs})
            if len(tiers) < 2:
                keep.extend(fs)
                continue
            paying = [t for t in tiers if any(earns(f) for f in fs if f.attrs.get("bal_lo") == t and f.metric in ("apy", "dividend_rate", "rate"))]
            pick = (paying or tiers)[0]
            keep.extend(f for f in fs if f.attrs.get("bal_lo", pick) == pick)
            dropped.extend(sorted({f"{kb.strip_footnotes(f.label)}: {f.value}" for f in fs
                                   if f.attrs.get("bal_lo", pick) != pick and f.metric == "apy"}))
        note = ("other balance tiers: " + "; ".join(dropped[:8])) if dropped else ""
        return keep, note

    def _mortgage_headline(self, rows: dict) -> dict:
        out = {}
        # keep all 30-year tiers, plus 20 and 15 year rows
        for rid, mets in rows.items():
            f = next(iter(mets.values()))[0]
            t = f.attrs.get("term")
            if t in (360, 240, 180) and ("bal_hi" not in f.attrs or t == 360):
                out[rid] = mets
        return out or rows

    def _no_rate(self, fam: str, info: dict) -> Answer:
        self._definitive = True
        thing = fam.replace("_", " ")
        return Answer(text=f"Dupaco's website doesn't list a rate for that {thing}. It isn't listed, so please contact Dupaco for details.", confident=False)

    def _row_sort_key(self, rid: tuple, mets: dict) -> tuple:
        f = next(mets[m][0] for m in ("apr", "rate", "apy", "dividend_rate", "min_earn", "min_open", "points", "payment") if m in mets)
        a = f.attrs
        return (a.get("tier_rank", 0), -(a.get("term") or 0) if f.family == "mortgage" else (a.get("term") or 0),
                a.get("bal_lo", -1) if f.family != "mortgage" else -a.get("bal_hi", 0), f.label)

    def _label_text(self, f: Fact, fam: str) -> str:
        lab = kb.strip_footnotes(f.label).replace(" | ", ", ")
        sec = f.section
        if fam in ("savings", "money_market", "hsa", "checking"):
            # label already includes the product name, tier
            parts = [p.strip() for p in f.label.split("|")]
            if len(parts) >= 2 and parts[1] not in ("—", "-", ""):
                return f"{parts[0]} ({parts[1]})"
            return parts[0]
        if fam in ("auto", "atv", "boat_rv", "home_improvement"):
            return lab
        if fam == "personal_loan":
            return f"{lab} personal loan"
        if fam == "credit_card":
            return f"{sec}" + ("" if lab.lower().startswith(sec.lower()[:8]) or lab.lower() in ("standard",) else f" ({lab})")
        if fam in ("certificate", "ira"):
            return lab if "certificate" in lab.lower() or ":" in lab else f"{lab} certificate"
        if fam == "mortgage":
            return lab
        return f"{sec} {lab}".strip()

    def _compose(self, view: SnapshotView, fam: str, rows: dict, info: dict) -> Answer:
        ordered = sorted(rows.items(), key=lambda kv: self._row_sort_key(*kv))
        srcs: list[str] = []
        lines = []
        notes = []
        for rid, mets in ordered:
            anyf = next(mets[m][0] for m in ("apr", "rate", "apy", "dividend_rate", "min_earn", "min_open", "points", "payment") if m in mets)
            label = self._label_text(anyf, fam)
            if fam in ("savings", "money_market", "hsa", "checking", "certificate", "ira"):
                dr = mets.get("dividend_rate", (None,))[0]
                apy = mets.get("apy", (None,))[0]
                bits = []
                if apy:
                    bits.append(f"{fmt_pct(apy.value)} APY")
                if dr and (not apy or dr.value != apy.value):
                    bits.append(f"{fmt_pct(dr.value)} dividend rate")
                if not bits and mets.get("rate"):
                    bits.append(f"{fmt_pct(mets['rate'][0].value)}")
                extra = ""
                if fam in ("certificate", "ira") and mets.get("min_earn"):
                    extra = f" ({mets['min_earn'][0].value} minimum)"
                elif re.search(r"minimum|open", info["ql"]) and mets.get("min_open"):
                    extra = f" (minimum to open {mets['min_open'][0].value})"
                lines.append(f"{label}: {', '.join(bits)}{extra}")
            elif fam == "mortgage":
                r = mets.get("rate", (None,))[0]
                apr = mets.get("apr", (None,))[0]
                bits = []
                if r:
                    bits.append(f"{fmt_pct(r.value)} rate")
                if apr:
                    bits.append(f"{fmt_pct(apr.value)} APR")
                lines.append(f"{label}: {', '.join(bits)}")
            else:
                apr = mets.get("apr", (None,))[0] or mets.get("rate", (None,))[0]
                if apr:
                    lines.append(f"{label}: {fmt_pct(apr.value)} APR")
            for m, (best, agree) in mets.items():
                for f in agree:
                    if f.url not in srcs:
                        srcs.append(f.url)
                # record conflicts for the note
                allf = view.by_key.get(best.key, [])
                others = {f.value for f in allf} - {best.value}
                if others:
                    notes.append(f"{label} {m}: chose {best.value} (effective {best.effective}) over {', '.join(sorted(others))} shown on older/less authoritative pages")
        if not lines:
            return self._no_rate(fam, info)
        intro = {"auto": "Current auto loan rates (as low as): ", "certificate": "Current certificate rates: ", "mortgage": "Current mortgage rates (purchase, 0 points): " if not info["refi"] else "Current refinance rates: ",
                 "personal_loan": "Current personal loan rates (as low as): ", "credit_card": "Current credit card APRs: ", "checking": "Current checking account rates: ",
                 "savings": "", "money_market": "", "hsa": "", "ira": "Current IRA rates: ", "heloc": "", "home_equity_loan": "", "home_improvement": "", "boat_rv": "", "atv": ""}.get(fam, "")
        if len(lines) == 1:
            text = intro + lines[0] + "."
        else:
            text = intro + "; ".join(lines) + "."
        if fam in ("auto",) and len(lines) == 1:
            text = f"The {lines[0].split(':')[0]} auto loan rate is{lines[0].split(':')[1]} (as low as)."
        if info.get("tier_note"):
            text += " Rates differ by balance tier; this is the first tier that earns dividends."
            notes.insert(0, info["tier_note"])
        srcs_sorted = sorted(srcs, key=lambda u: (-view.pages[u].authority if u in view.pages else 0, len(u)))[:6]
        return Answer(text=text, sources=[Source(url=u, snapshot_id=view.snapshot_id) for u in srcs_sorted],
                      conflict_note="; ".join(notes[:4]) or None)

    # ---- retrieval fallback --------------------------------------------------------------------
    _QUOTE_RE = re.compile(r"^\W{0,3}[“\"]|[”\"],? (?:said|says|added|explained|noted)\b|\b(?:said|says)\b")
    _HOWTO_RE = re.compile(r"^(?:how (?:do|can|should|would) i|how to|can i|where (?:do|can) i|what (?:do|should) i|i can'?t|i cannot|i'm unable|what happens)")
    _DEFN_RE = re.compile(r"^(?:what is|what's|what are|tell me about|describe|explain|who (?:is eligible|qualifies|can (?:open|get|have|apply))|is there|does|do you)")

    _AVAIL_RE = re.compile(r"\b(?:do you|does dupaco|can i get|is there|are there)\b.{0,25}\b(?:have|offer|any)\b|\bany [\w ]{0,20}options?\b")
    _ELIG_S_RE = re.compile(r"^\W*if you (?:have|are|live|work|own|use)\b|\beligib|\bqualif|\bmust (?:have|be)\b|\brequire", re.I)

    def retrieve_scored(self, view: SnapshotView, info: dict) -> list:
        """Candidate sentences for an open question, best first: [(score, Sentence)]."""
        toks = info["toks"]
        if not view.bm25 or not toks:
            return []
        weights = expand_query(toks)
        phrase_syn = []        # (question words of the phrase, site words that say the same thing)
        for rx, extra in PHRASE_SYNONYMS:
            m = rx.search(info["ql"])
            if m:
                et = set(tokenize(extra))
                for e in et:
                    weights.setdefault(e, 0.8)
                phrase_syn.append((set(tokenize(m.group(0))) & set(toks), et))
        hits = view.bm25.search(toks, topn=150, weights=weights)
        prods = info.get("products") or []
        if prods:
            # a named product: its own page section is always a candidate, even when other pages repeat the words more
            have = {i for i, _ in hits}
            for d in prods:
                for u in d["urls"]:
                    for st in view.page_sents.get(u, []):
                        if st.section == d["name"] and st.idx not in have:
                            hits.append((st.idx, view.bm25.score_doc(st.idx, st.toks, toks, weights)))
                            have.add(st.idx)
        if not hits:
            return []
        ql = info["ql"]
        qset = set(toks)
        idf = view.bm25.idf
        # a member word the site never uses ("overdraw") weighs as much as its best synonym ("overdraft")
        tw = {t: idf.get(t) or max([idf.get(x, 0.0) for x in SYNONYM_TABLE.get(t, ())] or [0.0]) for t in qset}
        qidf = sum(tw.values()) or 1.0
        top_bm = hits[0][1] or 1.0
        how_many = bool(re.search(r"how many|how much|number of|count", ql))
        definitional = bool(re.search(r"^(what is|what's|what are|tell me about|describe|explain)", ql))
        # how-to and "what is <product>" questions are answered by product and help pages, not by news stories
        howto = bool(self._HOWTO_RE.search(ql) or self._DEFN_RE.search(ql) or self._AVAIL_RE.search(ql))
        wants_quote = bool(re.search(r"\b(say|said|quote|according to)\b", ql))
        # "who is eligible / what do I need": the answer is a condition or requirement sentence
        subj = toks if definitional and len(toks) <= 4 else None
        elig_q = bool(re.search(r"eligib|qualif|who can (?:open|get|have|apply)|what do i need|requirements?\b|do i need to", ql))
        rare = max(qset, key=lambda t: idf.get(t, 0.0)) if qset else None
        dates = [p.content_date for p in view.pages.values() if p.content_date]
        newest = max(dates) if dates else "2026-01-01"
        title_cache: dict[str, set] = {}
        qbigrams = set(zip(toks, toks[1:]))
        prod_secs = {d["name"] for d in prods}
        prod_urls = {u for d in prods for u in d["urls"]}
        sec_toks = {d["name"]: set(d["toks"]) for d in prods}
        # "checking" (the account) and "check" (paper) share a stem; a page called /checking is not about paper checks
        acct = (lambda x: x) if re.search(r"checking", ql) else (lambda x: re.sub(r"(?i)checking", "chkacct", x))
        q_frame = " ".join(re.findall(r"[a-z']+", ql)[:3])
        howto_steps = bool(re.match(r"\s*how (?:do|can|should|would) i\b|\s*how to\b|\s*what (?:are|is) the steps", ql))
        atypes = info.get("atypes")
        if atypes is None:
            atypes = info["atypes"] = answer_types(ql)
        info["_upcoming"] = view.taken_on if (re.search(r"\bwhen (?:does|do|will|is|are|can|should|must)\b|\bnext\b|\bupcoming\b", ql)
                                              and not re.search(r"\b(?:did|was|were|last)\b", ql)) else None
        focus = self._atype_focus(info, idf)
        scored = []
        for i, s in hits:
            sent = view.sentences[i]
            p = sent.page
            st = set(sent.toks)
            in_prod = sent.section in prod_secs and p.url in prod_urls
            if in_prod:
                st |= sec_toks[sent.section]      # a sentence under "### Free Checking" is about Free Checking
            for qw, et in phrase_syn:
                if et <= st or et <= set(tokenize(sent.section)):
                    st |= qw | et                 # "shared branching" (sentence or its heading) answers "other credit unions"
            cover = sum(idf.get(t, 0.0) for t in qset & st) / qidf
            # a member word missing from the sentence but present as a synonym ("add" -> "put") counts at the synonym weight
            cover += sum(0.6 * tw[t] for t in qset - st if any(x in st for x in SYNONYM_TABLE.get(t, ()))) / qidf
            rec = 0.0
            if p.content_date:
                # 0 for the newest page, down to -1.2 for pages four or more years old
                try:
                    yrs = (date.fromisoformat(newest) - date.fromisoformat(p.content_date)).days / 365.0
                except Exception:
                    yrs = 2.0
                rec = -min(1.2, 0.35 * yrs)
            kind = "blog" if "/resources/" in p.path else p.kind     # /business/resources/* are articles, not product pages
            auth = {"product": 0.9, "rates": 0.5, "other": 0.6, "newsroom": 0.3, "blog": 0.0, "landing": -0.3, "calculator": -1.0}.get(kind, 0)
            if howto:
                auth += {"product": 0.7, "other": 0.5, "rates": 0.0, "newsroom": -0.4, "blog": -0.2}.get(kind, 0)
            # the member's words in the same order ("order checks", "monthly fee") count extra
            sb = set(zip(sent.toks, sent.toks[1:]))
            phrase = sum(1 for bg in qbigrams if bg in sb)
            path_ov = len(qset & set(tokenize(acct(re.sub(r"[/#?=_.-]", " ", p.path)))))
            if p.url not in title_cache:
                title_cache[p.url] = set(tokenize(acct(re.sub(r"\|\s*Dupaco\s*$", "", p.title)))) - {"dupaco"}
            ttl = title_cache[p.url]
            title_cov = sum(idf.get(t, 0.0) for t in qset & ttl) / qidf
            is_q = sent.text.rstrip().endswith("?")
            heading_like = (len(sent.text) < 30 or not sent.text.rstrip().endswith((".", "!", "\"", "”", ")", ":"))) and not is_q
            starts_num = bool(re.match(r"^\**\d", sent.text))
            quoted = bool(self._QUOTE_RE.search(sent.text)) and not wants_quote
            score = (s / top_bm) * 1.5 + cover * 4 + rec + auth + min(path_ov, 2) * 0.8 + min(title_cov, 1.0) * 0.8 \
                    + (0.8 if how_many and starts_num else 0) - (0.8 if heading_like else 0) \
                    + (0.6 if definitional and sent.pos <= 3 else 0) \
                    - (1.0 if quoted else 0) + min(phrase, 2) * 0.5 \
                    + (1.2 if elig_q and self._ELIG_S_RE.search(sent.text) else 0) \
                    + (1.0 if in_prod else 0.4 if p.url in prod_urls else 0) \
                    - (1.5 if rare and rare not in st and idf.get(rare, 0) > 4 else 0)
            if definitional and not is_q and subj and sent.toks[:len(subj)] == subj:
                score += 0.5          # "What is Overdraft Privilege?" -> "Overdraft Privilege may cover ..."
            if is_q:
                # an FAQ question: the answer is the next sentence on the page; a close match to the member's question is strong
                plist = view.page_sents.get(p.url, [])
                if sent.pos + 1 < len(plist) and not plist[sent.pos + 1].text.rstrip().endswith("?"):
                    sent = plist[sent.pos + 1]
                    # ...and an FAQ question that asks about more than the member did is a worse match
                    extra = len(set(view.sentences[i].toks) - qset - {x for t in qset for x in SYNONYM_TABLE.get(t, ())}) / max(1, len(set(view.sentences[i].toks)))
                    faq = bool(re.match(r"\W*(?:how|what|who|when|where|why|can|is|do|does|are|should|will|which)\b", view.sentences[i].text, re.I))
                    score += 0.3 + (0.6 if cover >= 0.75 else 0) - (1.5 * extra if faq else 0)
                    # the FAQ asks it the way the member did ("How do I ...")
                    if faq and howto and " ".join(re.findall(r"[a-z']+", view.sentences[i].text.lower())[:3]) == q_frame:
                        score += 0.5
                else:
                    score -= 1.0
            if re.search(r"(?:\.\.\.|…)\W*$", sent.text):
                # a teaser of another article ("... held Sunday, Feb. 25, 2024, Ellen Goodmann Miller, Jeff Gonner...")
                # carries this page's date but not this page's content
                score -= 1.5
            if atypes:
                # the question asks for a phone number / date / amount / person: a sentence (or its heading) that
                # carries one next to the question's key words is the answer; one that doesn't is only context
                score += 0.8 * self._atype_hits(sent.text + " \n " + sent.section, atypes, focus, info) / len(atypes)
            if _FINE_PRINT_RE.search(sent.section) or re.match(r"\W*\d[A-Z]", sent.text):
                score -= 1.0 if not re.search(r"\bterms?\b|disclosure|fine print|condition", ql) else 0.0
            if howto_steps:
                # how-to questions want steps ("endorse your check, log in to the Shine app, tap ... and follow the
                # prompts", a numbered list) over a one-line marketing bullet
                plist = view.page_sents.get(sent.page.url, [])
                nxt = plist[sent.pos + 1].text if sent.pos + 1 < len(plist) and plist[sent.pos + 1].section == sent.section else ""
                if sent.pos > 0 and _STEP_START_RE.match(sent.text) and plist[sent.pos - 1].section == sent.section \
                        and _STEP_START_RE.match(plist[sent.pos - 1].text):
                    nxt = plist[sent.pos - 1].text + " " + nxt       # the step before belongs to the same sequence
                # a cue that is the member's own word ("log in" in "I can't log in") is topic, not a step
                steps = sum(1 for m in _STEP_RE.findall(sent.text + " " + nxt) if not set(tokenize(m)) & qset)
                score += min(steps, 3) * 0.5
                if sent.text.lstrip().startswith(("- ", "* ")) and not steps and len(sent.text) < 120:
                    score -= 0.5
            scored.append((score, sent))
        scored.sort(key=lambda x: -x[0])
        return self._prefer_newer_fact(scored, qset, idf)

    # A figure in a dated article: "$27.7 million in Thank Use since 2016" / "nearly $7 million in fraud losses so far".
    _FIG_RE = re.compile(r"\$\s?\d[\d,.]*\s*(?:million|billion|thousand|[mbk]\b)?|\b\d[\d,.]*\s*(?:million|billion)\b|\b\d{1,3}(?:,\d{3})+\b|\b\d+(?:\.\d+)?\s?%")

    @classmethod
    def _fig_kinds(cls, text: str) -> set:
        kinds = set()
        for m in cls._FIG_RE.finditer(text):
            f = m.group(0).lower()
            if "%" in f:
                kinds.add("pct")
            elif re.search(r"million|billion|\d\s*[mb]\b", f):
                kinds.add("big$" if "$" in f else "big")
            elif "$" in f:
                kinds.add("$")
            else:
                kinds.add("count")
        if ATYPE_TEXT_RE["date"].search(text):
            kinds.add("date")           # "The deadline to apply is ... February 26, 2025" vs the 2026 release
        return kinds

    def _prefer_newer_fact(self, scored: list, qset: set, idf: dict, margin: float = 2.0) -> list:
        """Several dated articles state the same kind of figure about the same thing (running totals, 'so far this
        year' counts): the newest article wins, not the one whose wording happens to score best. 'Same kind' = the
        same figure type ($ millions, %, count), every distinctive question word the leading sentence covers, and
        the question's years. Undated pages (product pages) are left alone."""
        if len(scored) < 2:
            return scored
        top_s, top = scored[0]
        if not top.page.content_date or top.page.kind not in ("newsroom", "blog"):
            return scored
        kinds = self._fig_kinds(top.text)
        if not kinds:
            return scored
        key = {t for t in qset & set(top.toks) if idf.get(t, 0.0) >= 3.0}
        if not key:
            return scored
        key_w = sum(idf.get(t, 0.0) for t in key)
        years = {t for t in qset if re.fullmatch(r"(?:19|20)\d\d", t)}
        best_i, best_d = 0, top.page.content_date
        for i, (sc, st) in enumerate(scored[1:40], start=1):
            if sc < top_s - margin:
                break
            d = st.page.content_date
            if not d or d <= best_d or st.page.kind not in ("newsroom", "blog"):
                continue
            stt = set(st.toks)
            # most of what the leading sentence matched (by idf weight), the same years, the same kind of figure
            if sum(idf.get(t, 0.0) for t in key & stt) < 0.7 * key_w or not years <= stt \
                    or not (self._fig_kinds(st.text) & kinds):
                continue
            best_i, best_d = i, d
        if best_i:
            sc, st = scored[best_i]
            scored = [(top_s + 0.01, st)] + scored[:best_i] + scored[best_i + 1:]
        return scored

    @staticmethod
    def _atype_focus(info: dict, idf: dict) -> set:
        """The question's distinctive words (no question frame, no weak words), for 'value next to key words'."""
        f = {t for t in info["toks"] if t not in PHRASE_GENERIC and t not in WEAK_WORDS and idf.get(t, 0.0) >= 2.0
             and not re.fullmatch(r"\d+", t)}
        return f

    _ABBR_END_RE = re.compile(r"(?:\b(?:jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|ext|st|ave|rd|dr|blvd|no|mr|mrs|ms|dr|inc|co|u\.s)|\b[a-z])\.$", re.I)

    @classmethod
    def _clauses(cls, text: str) -> list:
        """Sentence/list-item pieces; 'Oct. 1-Nov. 12' and 'ext. 210' are not sentence ends."""
        out: list = []
        for line in text.split("\n"):
            for piece in re.split(r"(?<=[.!?;])\s+", line):
                if out and out[-1] and cls._ABBR_END_RE.search(out[-1]) and not out[-1].endswith(";"):
                    out[-1] += " " + piece
                else:
                    out.append(piece)
            out.append("")
        return [c for c in out if c]

    @staticmethod
    def _soft_overlap(focus: set, toks: set) -> set:
        """Focus words found in toks, allowing a shared stem prefix ('apply' ~ 'application', 'awarded' ~ 'awards')."""
        out = focus & toks
        for f in focus - out:
            if len(f) >= 4 and any(len(t) >= 4 and t[:4] == f[:4] and
                                   len(os.path.commonprefix([f, t])) >= 0.75 * min(len(f), len(t)) for t in toks):
                out.add(f)
        return out

    @staticmethod
    def _atype_hits(text: str, atypes: list, focus: set, info: dict) -> int:
        """How many of the asked-for value kinds the text carries in a clause that also names a key question word
        (a person must be named next to the asked-for role: 'Randy Skemp, chair of the board')."""
        n = 0
        clauses = Dupaco._clauses(text)
        role = person_role(info["ql"]) if "person" in atypes else set()
        for a in atypes:
            rx = ATYPE_TEXT_RE[a]
            for c in clauses:
                if not rx.search(c):
                    continue
                if a == "date" and info.get("_upcoming"):
                    ds = Dupaco._explicit_dates(c)
                    if ds and all(d < info["_upcoming"] for d in ds):
                        continue    # "when does it open": a date that has already passed is not the answer
                ct = set(tokenize(c))
                if a == "person":
                    if role and not role & ct:
                        continue
                elif focus and not Dupaco._soft_overlap(focus, ct) and a != "phone":
                    continue
                n += 1
                break
        return n

    _DATE_FULL_RE = re.compile(r"\b(" + _MONTH_ABBR + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d\d)\b", re.I)

    @classmethod
    def _explicit_dates(cls, text: str) -> list:
        """ISO dates written out with a year ('February 25, 2026', 'Jan. 5, 2027')."""
        out = []
        for m in cls._DATE_FULL_RE.finditer(text):
            mon = kb._MONTH_NUM.get(m.group(1).lower().rstrip(".")[:3])
            if mon:
                out.append(f"{m.group(3)}-{mon:02d}-{int(m.group(2)):02d}")
        return out

    def _complete_atype(self, view: SnapshotView, st, text: str, info: dict) -> str:
        """The passage is about the right thing but lacks the asked-for value (the phone number, the date in the
        section heading, the dollar figure two sentences later): add the sentence or heading from the same page that
        carries it next to the most question words."""
        atypes = info.get("atypes") or []
        if not atypes:
            return text
        idf = view.bm25.idf if view.bm25 else {}
        focus = self._atype_focus(info, idf)
        missing = [a for a in atypes if not self._atype_hits(text, [a], focus, info)]
        if not missing:
            return text
        heads = []
        if st.section and st.section not in text:
            heads.append(st.section)
        pool = heads + [x.text for x in view.page_sents.get(st.page.url, []) if x is not st]
        pool += [re.sub(r"[*_]+", "", h).strip() for h in re.findall(r"(?m)^#{1,6}\s+(.+)$", st.page.text)]
        add = []
        for a in missing:
            best, best_c = None, -1.0
            for cand in pool:
                if cand in text or cand in add or _FOOTNOTE_SENT_RE.search(cand) or len(cand) > 400:
                    continue
                if not self._atype_hits(cand, [a], focus, info):
                    continue
                ct = set(tokenize(cand))
                cov = sum(idf.get(t, 1.0) for t in self._soft_overlap(focus, ct))
                if a != "phone" and focus and cov <= 0:
                    continue
                if cov > best_c:
                    best, best_c = cand, cov
            if best:
                add.append(best)
        if not add:
            return text
        extra = " ".join(x.rstrip() if x.rstrip().endswith((".", "!", "?")) else x.rstrip() + "." for x in add)
        return re.sub(r"[*_]{2,}", "", text.rstrip() + " " + extra)

    def h_retrieve(self, view: SnapshotView, info: dict, accept=None) -> Answer:
        """Best passage that is about what was asked (core phrase gate) and, for a question that asks for an amount,
        states one. `accept(sentence, text)` can add a further condition (the fee fallback wants a price)."""
        self._last_hit = None
        scored = self.retrieve_scored(view, info)
        if not scored:
            return Answer(text=REFUSAL, confident=False)
        best_score = scored[0][0]
        thresh = 2.0 if len(view.sentences) > 500 else 1.0
        if best_score < thresh:
            return Answer(text=REFUSAL, confident=False)
        ql = info["ql"]
        pairs = core_pairs(info["raw"])
        qty = bool(_QUANTITY_Q_RE.search(ql))
        focus = set(info["toks"]) & QTY_NOUNS
        qprod = {x for t in info["toks"] if t in PRODUCT_HEAD_NOUNS and t not in (kb.stem("account"),) for x in _variants(t)}
        strong = {t for t in info["toks"] if t not in PHRASE_GENERIC and t not in WEAK_WORDS}
        seen: set = set()
        unpublished = False
        ok: list = []
        fam_q = detect_family(ql)
        fam_rx = dict(FAMILY_RULES).get(fam_q) if fam_q else None
        q_nums = set()
        if info.get("term"):
            q_nums |= {str(info["term"]), str(info["term"] // 12) if info["term"] % 12 == 0 else ""}
        q_nums |= {str(int(a)) for a in info.get("amounts") or [] if a >= 100}
        q_nums.discard("")
        for score, st in scored[:15]:
            if score < thresh or score < best_score - 2.5:
                break
            text = self._passage(view, st)
            if text in seen:
                continue
            seen.add(text)
            ctx_text = tokenize(text)
            ctx_page = tokenize(st.section) + tokenize(re.sub(r"\|\s*Dupaco.*$", "", st.page.title)) \
                + tokenize(re.sub(r"[/#?=_.-]", " ", st.page.path))
            have = set(ctx_text) | set(ctx_page)
            have |= {x for t in list(have) for x in _variants(t)}
            # weak words alone ("a lot", "daily cash flow") are not evidence
            if strong and not (strong & have) and not any(set(SYNONYM_TABLE.get(t, ())) & have for t in strong):
                continue
            # the product asked about (certificate, mortgage, card...) must be what the passage is about
            if qprod and not (qprod & have):
                continue
            # every core phrase must be there: in the passage, in the heading/title, or head in the passage and
            # modifier in the page context ("Shine password" on the Shine help page)
            if not all(pair_in(ctx_text, a, b) or pair_in(tokenize(st.section), a, b)
                       or pair_in(tokenize(st.page.title), a, b)
                       or (_variants(b) & set(ctx_text) and _variants(a) & set(ctx_page)) for a, b in pairs):
                continue
            if accept is not None and not accept(st, text):
                continue
            if fam_rx and q_nums and q_nums & set(re.findall(r"\d[\d,]*", text.replace(",", ""))):
                # the passage repeats the question's term or amount ("60-month", "$10,000"): it must be that
                # product's example, not another product's ("a 60-month home equity loan" for an ATV question)
                ctx = " ".join((text, st.section, st.page.title, re.sub(r"[/#?=_.-]", " ", st.page.path))).lower()
                if not re.search(fam_rx, ctx):
                    continue
            if qty and not self._states_amount(text, focus):
                text = self._extend_for_amount(view, st, text, focus, strong - QTY_NOUNS)
            ok.append((score, st, text))
        chosen = None
        if ok:
            score0, st0, text0 = ok[0]
            if _HEDGE_RE.search(text0) and re.search(r"\b(limit|maximum|max|how much can)\b", ql) and \
                    re.search(r"consult|contact us", text0, re.I):
                unpublished = True
            elif qty and not self._states_amount(text0, focus):
                # about the right thing, but this passage gives no figure. Another passage about the same thing a
                # little further down may ("Credit Coach Loan ... 8.95% APR" after the FAQ answer); if none does
                # ("Maximum deposit limits may apply", "a penalty will be imposed") the amount is not published
                subj = strong - QTY_NOUNS
                later = [x for x in ok[1:] if x[0] >= score0 - 1.5 and self._states_amount(x[2], focus)
                         and not _HEDGE_RE.search(x[2]) and self._amount_about(x[2], subj)]
                if later:
                    chosen = later[0]
                else:
                    unpublished = True
            else:
                chosen = ok[0]
            atypes = info.get("atypes") or []
            if chosen is not None and atypes and len(ok) > 1:
                # prefer a passage that carries the asked-for value (phone, date, amount, person) next to key words
                idf = view.bm25.idf
                fx = self._atype_focus(info, idf)
                need = len(atypes)
                if self._atype_hits(chosen[2], atypes, fx, info) < need:
                    for x in ok:
                        if x[0] < chosen[0] - 1.5:
                            break
                        if self._atype_hits(x[2], atypes, fx, info) >= need and \
                                (not qty or self._states_amount(x[2], focus)):
                            chosen = x
                            break
        if chosen is not None:
            score, st, text = chosen
            text = self._complete_atype(view, st, text, info)
            self._last_hit = st
            # the same sentence on other pages (/checking and /personal/checking-accounts): cite those too, product pages first
            same = self._sentence_pages(view).get(st.text, [])
            srcs = [st.page.url] + sorted((u for u in same if u != st.page.url),
                                          key=lambda u: (-view.pages[u].authority, len(u)))
            return Answer(text=text, sources=[Source(url=u, snapshot_id=view.snapshot_id, quote=st.text[:200]) for u in srcs[:4]])
        if unpublished:
            self._definitive = True
        return Answer(text=REFUSAL, confident=False)

    @staticmethod
    def _amount_about(text: str, subj: set) -> bool:
        """A sentence with a $ or % figure that also names what was asked ('loans up to $1,500 ... 8.95% APR'), not
        some other figure nearby ('a maximum account balance of $1 million' for a mobile-deposit limit)."""
        if not subj:
            return True
        for x in re.split(r"(?<=[.!?;])\s+", text):
            if re.search(r"\$\s?\d|\d\s?%", x):
                xt = set(tokenize(x))
                if subj & (xt | {v for t in xt for v in _variants(t)}):
                    return True
        return False

    def _extend_for_amount(self, view: SnapshotView, st, text: str, focus: set, subj: set = frozenset(), max_more: int = 3) -> str:
        """The passage stops just before the figure ('... earns quarterly dividends as you make payments.' and the
        next sentence has 'a low 8.95% APR'): read on in the same section, a few sentences at most, for the figure."""
        plist = view.page_sents.get(st.page.url, [])
        add = []
        for nx in plist[st.pos + 1:]:
            if nx.section != st.section or nx.text.rstrip().endswith("?") or _FOOTNOTE_SENT_RE.search(nx.text):
                break
            if nx.text in text:
                continue
            add.append(nx.text)
            if self._states_amount(nx.text, focus) and self._amount_about(nx.text, subj):
                out = re.sub(r"[*_]{2,}", "", text.rstrip() + " " + " ".join(add))
                return re.sub(r"(\d(?:\.\d+)?%\s?APR)\d\b", r"\1", out)      # footnote marker: "8.95% APR1"
            if len(add) >= max_more:
                break
        return text

    @staticmethod
    def _label_matches(info: dict, text: str) -> bool:
        """A prose 'label: value' answer passes the same core-phrase test as a retrieved passage."""
        toks = tokenize(text)
        have = set(toks) | {x for t in toks for x in _variants(t)}
        strong = {t for t in info["toks"] if t not in PHRASE_GENERIC and t not in WEAK_WORDS and t not in QTY_NOUNS}
        if strong and not strong & have:
            return False
        return all(pair_in(toks, a, b) for a, b in core_pairs(info["raw"]))

    @staticmethod
    def _states_amount(text: str, focus: set) -> bool:
        """Does the passage give a figure for what was asked? With a focus word ("penalty", "limit"), a sentence that
        names it must carry the figure; "may apply" / "contact us" without a $ or % figure is not one."""
        sents = re.split(r"(?<=[.!?;])\s+", text)
        rate_words = {kb.stem(w) for w in ("rate", "rates", "apr", "apy", "yield", "dividend")}
        rate_q = bool(focus & rate_words)
        if focus:
            # "rate" is answered by an APR/APY/% figure ("a low 8.95% APR"), not only by the word "rate"
            sents = [x for x in sents if set(tokenize(x)) & focus or (rate_q and re.search(r"\d\s?%\s?(?:apr|apy)?", x, re.I))]
        for x in sents:
            if rate_q and not (focus - rate_words) and not re.search(r"\d\s?%", x):
                continue            # a rate is a percentage; "1/12 of the annual rate" is not one
            if _QTY_ANSWER_RE.search(x) and not (_HEDGE_RE.search(x) and not re.search(r"\$\s?\d|\d\s?%", x)):
                return True
        return False

    @staticmethod
    def _sentence_pages(view: SnapshotView) -> dict:
        """sentence text -> page URLs carrying it (built once per snapshot view)."""
        m = getattr(view, "_sent_urls", None)
        if m is None:
            m = defaultdict(list)
            for st in view.sentences:
                m[st.text].append(st.page.url)
            view._sent_urls = m
        return m

    _ITEM_RE = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+(.+)$")

    @staticmethod
    def _list_after(page_text: str, lead: str, max_items: int = 12, max_chars: int = 700) -> list[str]:
        """The bullet or numbered list right after a lead-in sentence ending in ':' on the page, whole items only
        (short items such as 'iesnare.com' are not sentences, so they are read from the page text)."""
        key = re.sub(r"[*_]", "", lead.strip())[-60:]
        plain = re.sub(r"[*_]", "", page_text)
        k = plain.find(key)
        if k < 0:
            return []
        items: list[str] = []
        total = 0
        for line in plain[k + len(key):].split("\n")[1:]:
            if not line.strip():
                continue
            m = Dupaco._ITEM_RE.match(line)
            if not m:
                break
            it = re.sub(r"\s+", " ", m.group(1)).strip()
            if total + len(it) > max_chars or len(items) >= max_items:
                break                       # cut at an item boundary, never inside one
            items.append(it.rstrip(".;"))
            total += len(it)
        return items

    @staticmethod
    def _list_under(page_text: str, heading: str, max_chars: int = 700) -> list[str]:
        """Numbered items under a markdown heading, whole items only, up to max_chars."""
        plain = re.sub(r"[*_]", "", page_text)
        m = re.search(r"^#{1,6}\s+" + re.escape(re.sub(r"[*_]", "", heading)) + r"\s*$", plain, re.M)
        if not m:
            return []
        items: list[str] = []
        total = 0
        for line in plain[m.end():].split("\n"):
            if re.match(r"\s*#{1,6}\s", line):
                break
            im = re.match(r"\s*\d{1,2}[.)]\s+(.+)$", line)
            if not im:
                continue
            it = re.sub(r"\s+", " ", im.group(1)).strip().rstrip(".;")
            if total + len(it) > max_chars:
                break
            items.append(it)
            total += len(it)
        return items

    @staticmethod
    def _passage(view: SnapshotView, best) -> str:
        """The hit plus following sentences from the same page; a sentence ending in ':' pulls in its list items."""
        plist = view.page_sents.get(best.page.url, [])
        text = best.text
        if Dupaco._ITEM_RE.match(text) and re.match(r"\s*\d", text) and best.section:
            # one step of a numbered list ("8. Enter your password ..."): give the whole list under its heading
            steps = Dupaco._list_under(best.page.text, best.section)
            if len(steps) > 1:
                return re.sub(r"[*_]{2,}", "", f"{best.section}: " + " ".join(f"{i + 1}. {x}." for i, x in enumerate(steps)))
        if plist and best.pos > 0 and _STEP_START_RE.match(text):
            # a step that continues the one before it ("Select Submit a Travel Notice." after "Go to My Settings,
            # then Card Security."): start the passage at the first step of the sequence
            k = best.pos
            while k > 0 and best.pos - k < 3 and plist[k - 1].section == best.section and _STEP_START_RE.match(plist[k - 1].text) \
                    and not plist[k - 1].text.rstrip().endswith("?"):
                k -= 1
            if k < best.pos:
                text = " ".join(x.text for x in plist[k:best.pos]) + " " + text
        if plist:
            j = best.pos + 1
            while j < len(plist) and len(text) < 320:
                nxt = plist[j].text
                if nxt.endswith("?") and len(nxt) < 60:
                    break
                if plist[j].section != best.section or _FOOTNOTE_SENT_RE.search(nxt):
                    break           # next heading, or a rate-table footnote with effective dates
                listing = text.rstrip().endswith(":")
                if listing:
                    items = Dupaco._list_after(best.page.text, text)
                    if items:
                        text += " " + "; ".join(items) + "."
                        break
                text += " " + nxt
                j += 1
                if listing:
                    # list items follow: take up to a few of them
                    k = 0
                    while j < len(plist) and k < 3 and len(text) < 480 and not plist[j].text.endswith("?") \
                            and plist[j].section == best.section and not _FOOTNOTE_SENT_RE.search(plist[j].text):
                        text += "; " + plist[j].text if not text.rstrip().endswith(":") else " " + plist[j].text
                        j += 1
                        k += 1
                    break
                if best.text.rstrip().endswith((".", "!", "\"", "”")) and len(best.text) >= 30 and j > best.pos + 1 \
                        and not text.rstrip().endswith(":"):
                    break
        text = re.sub(r"[*_]{2,}", "", text)
        text = re.sub(r"\s+", " ", text).strip().strip("_* ")
        return re.sub(r"(^|(?<=[.!?] ))\d(?=[A-Z][a-z])", "", text)      # footnote markers: "1Unlimited ATM access"

    # ---- conflicts -----------------------------------------------------------------------------
    def _build_conflicts(self) -> list[Conflict]:
        out: list[Conflict] = []
        if not self.kb.views:
            return out
        latest = self.kb.views[-1]
        RATE_METRICS = ("apy", "apr", "rate", "dividend_rate")

        def rows_of(view):
            g: dict[tuple, list[Fact]] = defaultdict(list)
            for f in view.facts:
                if f.metric in RATE_METRICS:
                    g[group_id(f)].append(f)
            rows: dict[tuple, dict] = defaultdict(dict)
            for gid, fs in g.items():
                rows[gid[:-1]][gid[-1]] = fs
            return rows

        def primary(mets: dict) -> str:
            for m in ("apy", "apr", "rate", "dividend_rate"):
                if m in mets:
                    return m
            return next(iter(mets))

        done = set()
        # (a) within the latest snapshot: the same row shows different values on different pages
        for rid, mets in rows_of(latest).items():
            pm = primary(mets)
            if len({f.value for f in mets[pm]}) < 2:
                continue
            best = max(mets[pm], key=fact_rank)
            values, srcs = [], []
            for m in (pm,) + tuple(k for k in mets if k != pm):
                for f in self._one_per_value(mets[m]):
                    if f.value not in values:
                        values.append(f.value)
                    srcs.append(Source(url=f.url, snapshot_id=f.snapshot_id, quote=f"{f.label}: {f.value}"))
            res_bits = [best.value] + [max(mets[m], key=fact_rank).value for m in mets if m != pm]
            out.append(Conflict(topic=self._topic(best), values=values, sources=srcs[:6],
                                resolution=" / ".join(res_bits),
                                reason=f"the page with the newest effective date and highest authority ({best.kind} page {best.url}, effective {best.effective}) wins; the other pages are stale copies"))
            done.add(rid)
        # (b) across snapshots: the winning value changed over time (old value may still be quoted elsewhere)
        hist: dict[tuple, list[tuple[str, dict]]] = defaultdict(list)
        for v in self.kb.views:
            for rid, mets in rows_of(v).items():
                hist[rid].append((v.taken_on, {m: max(fs, key=fact_rank) for m, fs in mets.items()}))
        for rid, seq in hist.items():
            if rid in done or seq[-1][0] != latest.taken_on:
                continue
            pm = primary(seq[-1][1])
            prim_vals = []
            for _, mets in seq:
                if pm in mets and mets[pm].value not in prim_vals:
                    prim_vals.append(mets[pm].value)
            if len(prim_vals) < 2:
                continue
            values = list(prim_vals)
            srcs = []
            for _, mets in seq:
                for m, f in mets.items():
                    if f.value not in values:
                        values.append(f.value)
                if pm in mets:
                    srcs.append(Source(url=mets[pm].url, snapshot_id=mets[pm].snapshot_id, quote=f"{mets[pm].label}: {mets[pm].value}"))
            cur = seq[-1][1]
            res_bits = [cur[pm].value] + [f.value for m, f in cur.items() if m != pm]
            out.append(Conflict(topic=self._topic(cur[pm]), values=values, sources=srcs[-5:],
                                resolution=" / ".join(res_bits),
                                reason=f"value changed between snapshots; the newest snapshot ({latest.snapshot_id}, effective {cur[pm].effective}) is current"))
        return out

    @staticmethod
    def _one_per_value(fs: list[Fact]) -> list[Fact]:
        seen, out = set(), []
        for f in sorted(fs, key=fact_rank, reverse=True):
            if f.value not in seen:
                seen.add(f.value)
                out.append(f)
        return out

    @staticmethod
    def _topic(f: Fact) -> str:
        lab = kb.strip_footnotes(f.label).replace(" | ", " ")
        metric = {"apy": "APY", "apr": "APR", "rate": "rate", "dividend_rate": "dividend rate"}.get(f.metric, f.metric)
        fam = f.family.replace("_", " ")
        sec = f.section
        if fam in ("savings", "money market", "hsa", "checking", "certificate", "ira", "credit card"):
            return f"{lab} {metric}" if sec.lower()[:6] in lab.lower() else f"{sec} {lab} {metric}"
        return f"{fam} {lab} {metric}"


def build():
    return Dupaco()
