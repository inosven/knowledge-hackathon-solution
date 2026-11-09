"""
Knowledge base: ingest snapshots into page versions, structured facts and a sentence index.

Everything here is deterministic and uses only the standard library.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

# ----------------------------------------------------------------------------- text cleaning

_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_WS_RE = re.compile(r"[ \t]+")
_FOOTNOTE_RE = re.compile(r"(%|\bAPR|\bAPY)(\d)\b")
MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_LONG_DATE_RE = re.compile(rf"\b({MONTHS}|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+(\d{{1,2}}),\s+(20\d\d)\b", re.I)
_EFFECTIVE_RE = re.compile(r"effective\s+(?:as\s+of\s+)?(\d{1,2})/(\d{1,2})/(20\d\d)", re.I)
_EFFECTIVE_LONG_RE = re.compile(rf"effective\s+(?:as\s+of\s+|date:?\s*)?({MONTHS})\s+(\d{{1,2}}),?\s+(20\d\d)", re.I)
_MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTHS.split("|"))}
_MONTH_NUM.update({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12})


def parse_long_date(m: re.Match) -> Optional[str]:
    try:
        mon = _MONTH_NUM[m.group(1).lower()]
        return date(int(m.group(3)), mon, int(m.group(2))).isoformat()
    except Exception:
        return None


def strip_footnotes(s: str) -> str:
    """'5.39% APR1' -> '5.39% APR', '0.20%1' -> '0.20%'."""
    s = _FOOTNOTE_RE.sub(r"\1", s)
    return s.strip()


def clean_markdown(md: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (plain text with links flattened, [(anchor_text, url), ...])."""
    links = [(m.group(1).strip(), m.group(2).strip()) for m in _LINK_RE.finditer(md)]
    text = _IMG_RE.sub(" ", md)
    text = _LINK_RE.sub(lambda m: m.group(1), text)
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\*", "*").replace("\\#", "#")
    text = text.replace("&amp;#45;", "-").replace("&#45;", "-").replace("&amp;", "&").replace("&nbsp;", " ")
    text = text.replace(" ", " ")
    # cut off "related articles" blocks on newsroom/blog pages
    # (only up to the next "## " heading: product pages put their FAQ after the related-articles block)
    for marker in ("## Handpicked for You", "## Related Articles", "## You May Also Like"):
        i = text.find(marker)
        if i > 200:
            # the block ends at the next heading of any level ("### The Fine Print" holds card fee disclosures)
            mj = re.compile(r"\n#{1,4} ").search(text, i + len(marker))
            j = mj.start() if mj else -1
            text = text[:i] + ("\n" + text[j + 1:] if j > 0 else "")
    text = _WS_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), links


# ----------------------------------------------------------------------------- page classification

def page_kind(url: str) -> str:
    u = url.lower()
    path = re.sub(r"^https?://[^/]+", "", u).split("?")[0].rstrip("/")
    if path in ("/rates", "/business-rates") or path.endswith("/rates"):
        return "rates"
    if "/calculators/" in path or path.endswith("-calculator") or "calculator" in path:
        return "calculator"
    if "/newsroom/" in path:
        return "newsroom"
    if "/learn/blog/" in path or re.search(r"/20\d\d/\d\d/", path):
        return "blog"
    if path.endswith("-lp") or "switch-and-earn" in path or "-offer" in path or "/hubfs/" in path:
        return "landing"
    if path.startswith("/personal") or path.startswith("/business") or path in ("/checking", "/savings-account", "/contact", "/about", "/about/join") \
            or path.startswith("/savings") or path.startswith("/connect") or path.startswith("/online-and-mobile-banking") or path.startswith("/checking"):
        return "product"
    return "other"


AUTHORITY = {"rates": 100, "product": 80, "other": 55, "landing": 40, "newsroom": 35, "blog": 30, "calculator": 15}


# ----------------------------------------------------------------------------- data model

@dataclass
class PageVer:
    url: str
    snapshot_id: str
    taken_on: str
    fetched_at: str
    title: str
    text: str
    kind: str
    authority: int
    content_date: Optional[str]      # publication date (newsroom/blog) or latest 'effective' date
    links: list[tuple[str, str]]
    body_hash: str
    path: str = ""


@dataclass
class Fact:
    section: str          # heading above the table, e.g. "Share Savings"
    label: str            # row label (non-metric columns joined with ' | ')
    metric: str           # apy / dividend_rate / apr / rate / min_open / min_earn / points / payment
    value: str            # cleaned display value, e.g. "5.39%"
    attrs: dict
    effective: Optional[str]
    url: str
    snapshot_id: str
    taken_on: str
    authority: int
    kind: str
    key: str = ""
    family: str = ""


# ----------------------------------------------------------------------------- label parsing

def _money(s: str) -> float:
    return float(s.replace(",", "").replace("$", ""))


def parse_label_attrs(label: str, section: str) -> dict:
    a: dict = {}
    L = label.lower().replace("–", "-").replace("—", "-")
    S = section.lower()
    # term
    m = re.search(r"(\d+)\s*-?\s*(?:month|months|mo)\b", L)
    if m:
        a["term"] = int(m.group(1))
    else:
        m = re.search(r"(\d+)\s*-?\s*(?:year|yr)\b", L)
        if m:
            a["term"] = int(m.group(1)) * 12
    # model years: "2024-27 Model Year", "Model Year 2026-23", "2016 Model Year (and older)"
    m = re.search(r"(20\d\d)\s*-\s*(\d{2,4})", L)
    if m and "model" in L:
        lo, hi = int(m.group(1)), int(m.group(2))
        if hi < 100:
            hi += 2000
        a["model_lo"], a["model_hi"] = min(lo, hi), max(lo, hi)
    else:
        m = re.search(r"(20\d\d)\s*model year\s*\(and older\)", L)
        if m:
            a["model_lo"], a["model_hi"] = 0, int(m.group(1))
    # balance / loan amount tiers
    m = re.search(r"\$([\d,]+(?:\.\d+)?)\s*-\s*\$?([\d,]+(?:\.\d+)?)", L)
    if m:
        a["bal_lo"], a["bal_hi"] = _money(m.group(1)), _money(m.group(2))
    else:
        m = re.search(r"\$([\d,]+(?:\.\d+)?)\s*\+", L)
        if m:
            a["bal_lo"], a["bal_hi"] = _money(m.group(1)), float("inf")
        else:
            m = re.search(r"\$([\d,]+(?:\.\d+)?)\s*or less", L)
            if m:
                a["bal_lo"], a["bal_hi"] = 0.0, _money(m.group(1))
    a["jumbo"] = "jumbo" in L or "jumbo" in S
    a["special"] = "special" in L or "special" in S
    if "roth" in L:
        a["ira"] = "roth"
    elif "traditional" in L:
        a["ira"] = "traditional"
    elif "ira" in S:
        a["ira"] = "any"
    if "purchase" in L or "purchase" in S:
        a["mtg"] = "purchase"
    elif "refinance" in L or "refinance" in S:
        a["mtg"] = "refinance"
    if "variable" in L or "variable" in S:
        a["variable"] = True
    if "introductory" in L or "intro" in L:
        a["intro"] = True
    return a


def _fam(s: str) -> str:
    s = s.lower()
    if re.search(r"\d+-year fixed|fixed conventional|purchase:|refinance:|\barm\b|adjustable", s):
        return "mortgage"
    if "atv" in s or "snowmobile" in s or "jet ski" in s:
        return "atv"
    if "boat" in s or re.search(r"\brvs?\b", s) or "camper" in s:
        return "boat_rv"
    if "auto" in s or "motorcycle" in s or "vehicle" in s or "model year" in s:
        return "auto"
    if "ira" in s and "certificate" not in s:
        return "ira"
    if "certificate" in s or re.search(r"\b\d+ month\b", s) and "loan" not in s:
        return "certificate"
    if "heloc" in s or "line of credit" in s:
        return "heloc"
    if "home equity" in s:
        return "home_equity_loan"
    if "home improvement" in s:
        return "home_improvement"
    if "home loan" in s or "mortgage" in s:
        return "mortgage"
    if "personal loan" in s or "signature loan" in s:
        return "personal_loan"
    if "visa" in s or "credit card" in s or "mastercard" in s or re.search(r"\bcards?\b", s):
        return "credit_card"
    if "checking" in s:
        return "checking"
    if "money market" in s:
        return "money_market"
    if "health savings" in s or "hsa" in s:
        return "hsa"
    if "savings" in s or "club" in s:
        return "savings"
    if "business" in s and "loan" in s:
        return "business_loan"
    return "other"


def family_of(section: str, label: str, url: str) -> str:
    fl, fs = _fam(label), _fam(section)
    if fl == "other":
        return fs
    if fs == "other":
        return fl
    if fl != fs:
        if fs in ("atv", "boat_rv") and fl == "auto":
            return fs
        if fs in ("certificate", "ira") and fl in ("certificate", "ira"):
            return fl
        return fl
    return fl


def classify_header(h: str) -> str:
    h = h.lower().strip()
    if "apy" in h or "annual percentage yield" in h:
        return "apy"
    if "dividend rate" in h:
        return "dividend_rate"
    if "apr" in h or "annual percentage rate" in h:
        return "apr"
    if h == "rate" or h.startswith("rate") or h.endswith(" rate"):
        return "rate"
    if "min" in h and "open" in h:
        return "min_open"
    if "min" in h and ("earn" in h or "balance" in h):
        return "min_earn"
    if "points" in h:
        return "points"
    if "payment" in h:
        return "payment"
    if "term" in h:
        return "term"
    return "label"


_METRICS = {"apy", "dividend_rate", "apr", "rate", "min_open", "min_earn", "points", "payment"}


def norm_label(s: str) -> str:
    s = s.lower().replace("–", "-").replace("—", "-")
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9$%.+-]+", " ", s)
    s = re.sub(r"\bmonths?\b", "mo", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ----------------------------------------------------------------------------- table extraction

def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def extract_tables(text: str) -> list[tuple[str, list[str], list[list[str]], Optional[str], int]]:
    """Yield (section_heading, headers, rows, effective_date_after_table, line_index)."""
    lines = text.split("\n")
    out = []
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].lstrip().startswith("|") and i + 1 < n and re.match(r"^\s*\|?\s*:?-{2,}", lines[i + 1]):
            headers = _split_row(lines[i])
            j = i + 2
            rows = []
            while j < n and lines[j].lstrip().startswith("|"):
                r = _split_row(lines[j])
                if len(r) == len(headers):
                    rows.append(r)
                j += 1
            # heading: nearest previous non-empty, non-table, short line
            k = i - 1
            section = ""
            while k >= 0:
                cand = lines[k].strip()
                if cand and not cand.startswith("|") and len(cand) < 90 and "effective" not in cand.lower() \
                        and not cand.lower().startswith(("open my", "see details", "learn more", "apply", "compare")):
                    section = cand.strip("#* ").strip()
                    break
                k -= 1
            # a bare heading like "Regular" or "Special" needs its parent ("Certificates")
            if _fam(section) == "other":
                kk = k - 1
                steps = 0
                while kk >= 0 and steps < 60:
                    cand = lines[kk].strip()
                    if cand and not cand.startswith("|") and len(cand) < 60:
                        cand_clean = cand.strip("#* ").strip()
                        if _fam(cand_clean) != "other":
                            section = f"{cand_clean} {section}".strip()
                            break
                    kk -= 1
                    steps += 1
            # effective date within next 4 lines
            eff = None
            for e in range(j, min(n, j + 4)):
                m = _EFFECTIVE_RE.search(lines[e])
                if m:
                    try:
                        eff = date(int(m.group(3)), int(m.group(1)), int(m.group(2))).isoformat()
                    except Exception:
                        eff = None
                    break
            out.append((section, headers, rows, eff, i))
            i = j
        else:
            i += 1
    return out


def facts_from_page(p: PageVer) -> list[Fact]:
    facts: list[Fact] = []
    tables = extract_tables(p.text)
    for section, headers, rows, eff, _ in tables:
        kinds = [classify_header(h) for h in headers]
        rate_cols = [i for i, k in enumerate(kinds) if k in ("apy", "apr", "rate", "dividend_rate")]
        if not rate_cols or not any("%" in r[i] for r in rows for i in rate_cols):
            facts.extend(_fee_table_facts(p, section, headers, rows, eff))
            continue
        label_idx = [i for i, k in enumerate(kinds) if k not in _METRICS]
        # tier rank for model-year tables (newest tier first as it appears)
        tier_rank: dict[str, int] = {}
        for r in rows:
            lab = " | ".join(r[i] for i in label_idx) if label_idx else section
            attrs = parse_label_attrs(lab, section)
            if "model_lo" in attrs:
                mk = (attrs["model_lo"], attrs["model_hi"])
                if mk not in tier_rank:
                    tier_rank[mk] = len(tier_rank)
        for r in rows:
            lab = " | ".join(r[i] for i in label_idx) if label_idx else section
            if not lab.strip():
                lab = section
            attrs = parse_label_attrs(lab, section)
            fam = family_of(section, lab, p.url)
            if "model_lo" in attrs:
                attrs["tier_rank"] = tier_rank[(attrs["model_lo"], attrs["model_hi"])]
            for i, k in enumerate(kinds):
                if k not in _METRICS:
                    continue
                val = strip_footnotes(r[i])
                if re.search(r"\bAPR\b", val) and k in ("apy", "rate", "dividend_rate"):
                    k = "apr"          # the cell's own unit beats a sloppy column header
                elif re.search(r"\bAPY\b", val) and k in ("apr", "rate"):
                    k = "apy"
                val = re.sub(r"\s*\b(APR|APY)\b\s*", "", val).strip().replace("–", "-").replace("—", "-")
                mv = re.match(r"^(\$?\d[\d,]*(?:\.\d+)?%?(?:\s*-\s*\$?\d[\d,]*(?:\.\d+)?%?)?)", val)
                if mv:
                    val = re.sub(r"\s*-\s*", " - ", mv.group(1))
                if not val or val in ("—", "-", "–", "N/A", "n/a"):
                    continue
                f = Fact(section=section, label=lab, metric=k, value=val, attrs=attrs, effective=eff or p.content_date,
                         url=p.url, snapshot_id=p.snapshot_id, taken_on=p.taken_on, authority=p.authority, kind=p.kind, family=fam)
                f.key = make_key(f)
                facts.append(f)
    return facts


_PROSE_RE = re.compile(r"^[\-\*\s]*([A-Za-z0-9$][^|\n;]{2,70}?)\s*(?:\bis\b|\bof\b|\bat\b|:|—|–)?\s*(?:up to\s+)?(\$\s?\d[\d,]*(?:\.\d+)?|\d{1,2}(?:\.\d+)?\s?%)(?:\s*(APY|APR))?(?![\w.])[^\n;]{0,80}$")


def prose_facts_from_page(p: PageVer) -> list[Fact]:
    """Key/value facts written as prose, e.g. '6-month certificate APY: 2.95%' or 'late payment fee—up to $35.00'."""
    facts = []
    fee_page = bool(re.search(r"fee schedule|fees and charges", p.text, re.I))
    for line in p.text.split("\n"):
        if line.lstrip().startswith("|") or len(line) > 4000:
            continue
        long_line = len(line) > 600       # card disclosures: one long paragraph, fees separated by ';'
        for seg in re.split(r";\s*|\.\s+(?=[A-Z])", line):
            m = _PROSE_RE.match(seg.strip())
            if not m or long_line and not re.search(r"\bfees?\b|\bcharge\b|\bpenalty\b", m.group(1), re.I):
                continue
            label = re.sub(r"^(other fees include:|fees include:)\s*", "", m.group(1).strip(" *-_"), flags=re.I)
            label = re.sub(r"\s*\b(APY|APR)\s*:?$", "", label).strip(" :")
            val = m.group(2).replace(" ", "")
            unit = (m.group(3) or "").lower()
            ll = label.lower()
            if len(label) > 70 or "effective" in ll or ll.startswith(("rates are", "the ", "a ", "an ", "if ", "for ", "with ", "when ", "you ", "we ", "your ")):
                continue
            if not re.search(r"[a-z]{3}", ll) or len(ll.split()) > 10:
                continue
            if "%" in val:
                metric = "apy" if ("apy" in m.group(1).lower() or unit == "apy" or "yield" in ll) else "apr" if ("apr" in m.group(1).lower() or unit == "apr") else "rate"
            else:
                metric = "fee" if (fee_page or re.search(r"\bfees?\b|charge|penalty", ll)) else "amount"
            attrs = parse_label_attrs(label, p.title)
            fam = _fam(label)
            if fam == "other" and metric == "fee":
                fam = "fee"
            if fam == "other" and metric == "amount":
                continue
            expired = bool(re.search(r"offer (ended|expired)|expired|ended (on )?\d", seg.lower()))
            attrs["expired"] = expired
            if metric == "fee":
                # pricing wording after the amount ("or 3.00% of the amount of each cash advance, whichever is greater")
                tail = seg.strip()[m.end(3) if m.group(3) else m.end(2):].strip(" ,")
                if re.match(r"(?:or|per|each|for|of)\b", tail, re.I):
                    attrs["detail"] = tail.rstrip(" .")
            f = Fact(section=p.title or "", label=label, metric=metric, value=val, attrs=attrs, effective=p.content_date,
                     url=p.url, snapshot_id=p.snapshot_id, taken_on=p.taken_on, authority=p.authority - (30 if expired else 0), kind=p.kind, family=fam)
            f.key = make_key(f)
            facts.append(f)
    return facts


_FEE_VAL_RE = re.compile(r"^(Free|No charge|\$\s?\d[\d,]*(?:\.\d+)?|\d{1,2}(?:\.\d+)?\s?%)", re.I)


def _fee_table_facts(p: PageVer, section: str, headers: list[str], rows: list[list[str]], eff: Optional[str]) -> list[Fact]:
    """Two-column 'service | price' tables (fee schedules)."""
    out = []
    if len(headers) < 2 or p.kind == "calculator":
        return out
    cand = [(r[0].strip(), r[1].strip()) for r in rows if len(r) >= 2 and r[0].strip() and r[1].strip()]
    feeish = "fee schedule" in p.text.lower() or sum(1 for l, _ in cand if re.search(r"fee|charge|cost|per item|each|per month", l, re.I)) >= max(1, len(cand) // 3)
    if not feeish:
        return out
    if len(cand) < 2 or sum(1 for _, v in cand if _FEE_VAL_RE.match(v)) < max(2, len(cand) // 2):
        return out
    pairs = cand
    if _FEE_VAL_RE.match(headers[1].strip()) and headers[0].strip():
        pairs = [(headers[0].strip(), headers[1].strip())] + cand
    group = section
    for label, val in pairs:
        m = _FEE_VAL_RE.match(val)
        if not m:
            group = label if len(label) < 60 else group
            continue
        v = m.group(1).replace(" ", "")
        v = "Free" if v.lower() in ("free", "no charge") else v
        attrs = {"detail": val[len(m.group(0)):].strip(" *"), "group": group}
        f = Fact(section=group or section, label=label, metric="fee", value=v, attrs=attrs, effective=eff or p.content_date,
                 url=p.url, snapshot_id=p.snapshot_id, taken_on=p.taken_on, authority=p.authority + 10, kind=p.kind, family="fee")
        f.key = f"fee|{norm_label(label)}"
        out.append(f)
    return out


def make_key(f: Fact) -> str:
    a = f.attrs
    if f.family in ("auto", "atv") and "tier_rank" in a:
        lab = f"tier{a['tier_rank']} {a.get('term', '')}mo"
        return f"{f.family}|{lab}|{f.metric}"
    sec = norm_label(f.section)
    lab = norm_label(f.label)
    # mortgage/business pages repeat the same product under different headings; normalise a little
    sec = re.sub(r"\b(home loans?|rates?)\b", "", sec).strip(" -")
    return f"{f.family}|{sec}|{lab}|{f.metric}"


# ----------------------------------------------------------------------------- page-level extraction

def content_date_of(text: str, kind: str) -> Optional[str]:
    effs = []
    for m in _EFFECTIVE_RE.finditer(text):
        try:
            effs.append(date(int(m.group(3)), int(m.group(1)), int(m.group(2))).isoformat())
        except Exception:
            pass
    for m in _EFFECTIVE_LONG_RE.finditer(text):
        d = parse_long_date(m)
        if d:
            effs.append(d)
    if kind in ("newsroom", "blog"):
        head = text[:1500]
        m = _LONG_DATE_RE.search(head)
        if m:
            d = parse_long_date(m)
            if d:
                return d
    if effs:
        return max(effs)
    return None


def make_pagever(url: str, title: str, md: str, fetched_at: str, snapshot_id: str, taken_on: str) -> PageVer:
    text, links = clean_markdown(md)
    kind = page_kind(url)
    body_hash = hashlib.md5(re.sub(r"\s+", " ", text).encode("utf-8")).hexdigest()
    path = re.sub(r"^https?://[^/]+", "", url).split("?")[0].rstrip("/").lower()
    return PageVer(url=url, snapshot_id=snapshot_id, taken_on=taken_on, fetched_at=fetched_at, title=title or "",
                   text=text, kind=kind, authority=AUTHORITY.get(kind, 50), content_date=content_date_of(text, kind),
                   links=links, body_hash=body_hash, path=path)


# ----------------------------------------------------------------------------- sentence index (BM25)

_TOKEN_RE = re.compile(r"[a-z0-9$%][a-z0-9$%.'-]*")
STOP = set("""a an the of to and or in on at for with by from is are was were be been it its this that these those you your we our
us as can do does how what which who when where why will would should could may i me my about any all get give tell
me know please there their they them has have had not no into than then so if up out more most some such only also
""".split())


# Members write "sixty-month", "five year", "7/1 ARM", "CD"; the site writes "60 Month", "5 Year", "7/1", "Certificate".
# Canonical number-word table (solution.parse_question reuses it for term slots).
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
             "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "seventeen": 17, "eighteen": 18,
             "twenty": 20, "thirty": 30, "thirty-six": 36, "forty-eight": 48, "sixty": 60}
NUM_WORD_ALT = "|".join(sorted(NUM_WORDS, key=len, reverse=True))   # longest first: "thirty-six" before "thirty"
_UNIT = r"(?:month|months|mo|mos|year|years|yr|yrs|day|days)\b"
_NUM_WORD_RE = re.compile(r"\b(" + NUM_WORD_ALT + r")[\s-]+(?=" + _UNIT + ")")
# 7/1, 5/5, 10/1 (ARM ratios) but not any part of a date such as 08/31/2026 or 8/31/26.
_RATIO_RE = re.compile(r"(?<![\d/])\b(\d{1,2})\s*/\s*(\d{1,2})\b(?!\s*/\s*\d)")
ALIAS = {"cd": "certificate", "cds": "certificate"}
_LOGIN_RE = re.compile(r"\b(?:log|sign)(?:ging)?[- ]?(?:in|on)\b|\blogins?\b")   # "log in" / "sign in" / "login" are one word


def normalize_text(s: str) -> str:
    """Lower-case and rewrite spellings so that the member's words match the website's."""
    s = s.lower().replace("–", "-").replace("—", "-").replace("’", "'")
    s = _NUM_WORD_RE.sub(lambda m: str(NUM_WORDS[m.group(1)]) + " ", s)
    s = _RATIO_RE.sub(r"\1x\2", s)
    s = _LOGIN_RE.sub("login", s)
    return s


def tokenize(s: str) -> list[str]:
    s = normalize_text(s)
    toks = _TOKEN_RE.findall(s)
    expanded = []
    for t in toks:
        expanded.append(t)
        if "-" in t.strip("-"):
            expanded.extend(t.split("-"))
    out = []
    for t in expanded:
        t = t.strip(".'-")
        if not t or t in STOP:
            continue
        if t.endswith("'s"):
            t = t[:-2]
        out.append(stem(ALIAS.get(t, t)))
    return out


_SUFFIXES = ("izations", "ization", "ations", "ation", "ized", "izes", "ize", "ings", "ing", "tion", "sion", "ness", "ments", "ment", "ed", "es", "ly", "s")


def stem(t: str) -> str:
    """Very light suffix stripping, enough to match 'recognized' with 'recognition' and plurals."""
    if len(t) <= 3 or not t.isalpha():
        return t
    if len(t) == 4 and t.endswith("s") and t[-2] not in "su":
        return t[:-1]                    # short plurals: fees -> fee, atms -> atm, kids -> kid
    for suf in _SUFFIXES:
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            if suf == "s" and t.endswith("ss"):
                return t
            t = t[: -len(suf)]
            if suf in ("tion", "sion") and t.endswith("i"):
                t = t[:-1]
            if suf in ("ing", "ings", "ed") and len(t) >= 4 and t[-1] == t[-2] and t[-1] not in "lsz" and t[-1].isalpha():
                t = t[:-1]               # referred / referring -> refer, planned -> plan
            break
    if len(t) > 5 and t.endswith("e"):
        t = t[:-1]
    return t


@dataclass
class Sentence:
    text: str
    page: PageVer
    toks: list[str]
    pos: int = 0
    section: str = ""     # markdown heading the sentence sits under
    idx: int = -1         # position in SnapshotView.sentences / the BM25 corpus


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.avgdl = (sum(len(d) for d in docs) / self.N) if self.N else 1.0
        self.df: Counter = Counter()
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.dl = [len(d) for d in docs]
        for i, d in enumerate(docs):
            c = Counter(d)
            for t, f in c.items():
                self.df[t] += 1
                self.postings[t].append((i, f))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in self.df.items()}

    def search(self, q: list[str], topn: int = 50, weights: Optional[dict[str, float]] = None) -> list[tuple[int, float]]:
        """BM25 over the query terms; `weights` (term -> factor) lets synonyms count less than the member's own words."""
        scores: dict[int, float] = defaultdict(float)
        terms = set(q) | (set(weights) if weights else set())
        for t in terms:
            if t not in self.postings:
                continue
            idf = self.idf[t] * (weights.get(t, 1.0) if weights else 1.0)
            for i, f in self.postings[t]:
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / denom
        return sorted(scores.items(), key=lambda x: -x[1])[:topn]

    def score_doc(self, i: int, doc: list[str], q: list[str], weights: Optional[dict[str, float]] = None) -> float:
        """BM25 score of one document (doc i, tokens `doc`) for the query, same formula as search()."""
        c = Counter(doc)
        total = 0.0
        for t in set(q) | (set(weights) if weights else set()):
            f = c.get(t, 0)
            if not f or t not in self.idf:
                continue
            idf = self.idf[t] * (weights.get(t, 1.0) if weights else 1.0)
            total += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return total


def split_sentences(text: str) -> list[str]:
    return [s for s, _ in split_sentences_h(text)]


_PDF_KEEP_LINE_RE = re.compile(r"^\s*(?:#|\||[-*•]\s|\d{1,2}[.)]\s)|\$\s?\d")


def unwrap_lines(text: str) -> str:
    """PDF text keeps the page's hard line wraps ("Go to My Settings,\nthen Card Security.\nSelect Submit\na Travel
    Notice."), so a sentence is cut into fragments too short to index. Join the wrapped lines of a block back into
    one paragraph; headings, table rows, list items and price lines ("Rush fee $25.00") stay on their own line."""
    out: list[str] = []
    run: list[str] = []
    for line in text.split("\n"):
        ln = line.strip()
        if not ln or _PDF_KEEP_LINE_RE.search(line) or "•" in ln:
            if run:
                out.append(" ".join(run))
                run = []
            out.append(line)
            continue
        if run:
            prev = run[-1]
            last = prev.split()[-1] if prev.split() else ""
            # a wrapped sentence: the line before stops mid-sentence ("Go to My Settings," / "build stronger savings
            # and feel") or this line carries on in lower case ("a Travel Notice."); a run of title-case lines
            # ("Danielle Gratton" / "Chief Financial Officer") is a list, not prose
            mid = not re.search(r"[.!?:]['\"”’)]*$", prev) and (re.search(r"[,;(-]$", prev) or re.match(r"[a-z]", last))
            if not (mid or re.match(r"[a-z(]", ln)):
                out.append(" ".join(run))
                run = []
        run.append(ln)
    if run:
        out.append(" ".join(run))
    return "\n".join(out)


def split_sentences_h(text: str, wrapped: bool = False) -> list[tuple[str, str]]:
    """Sentences with the markdown heading they sit under ("" before the first heading). `wrapped`: the text is a
    PDF with hard line wraps (see unwrap_lines)."""
    out = []
    heading = ""
    if wrapped:
        text = unwrap_lines(text)
    for para in text.split("\n"):
        para = para.strip()
        if not para or para.startswith("|"):
            continue
        hm = re.match(r"#{1,6}\s+(.+)", para)
        if hm:
            heading = re.sub(r"[*_]+", "", hm.group(1)).strip()
        para = para.strip("#*> ").strip()
        if len(para) < 25:
            continue
        # split on sentence end followed by capital/quote
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"“(])", para)
        buf = ""
        for s in parts:
            s = s.strip()
            if not s:
                continue
            if len(buf) + len(s) < 160:
                buf = (buf + " " + s).strip()
            else:
                if buf:
                    out.append((buf, heading))
                buf = s
        if buf:
            out.append((buf, heading))
    return out


# ----------------------------------------------------------------------------- branch table

STATE_ABBR = {"iowa": "IA", "ia": "IA", "ia.": "IA", "wis.": "WI", "wis": "WI", "wisconsin": "WI", "wi": "WI",
              "ill.": "IL", "ill": "IL", "illinois": "IL", "il": "IL"}
_STREET_ABBR = {"street": "st", "road": "rd", "roads": "rd", "avenue": "ave", "av": "ave", "boulevard": "blvd", "highway": "hwy",
                "drive": "dr", "lane": "ln", "court": "ct", "parkway": "pkwy", "east": "e", "west": "w", "north": "n",
                "south": "s", "so": "s", "southeast": "se", "southwest": "sw", "northeast": "ne", "northwest": "nw",
                "first": "1st", "second": "2nd", "third": "3rd", "ninth": "9th"}
STREET_GENERIC = {"st", "rd", "ave", "blvd", "hwy", "dr", "ln", "ct", "pkwy", "way", "e", "w", "n", "s", "se", "sw", "ne", "nw",
                  "business", "plaza", "the", "of", "and", "at"}
_STREET_SUFFIX = r"(?:St|Street|Road|Rd|Ave|Avenue|Blvd|Boulevard|Way|Lane|Ln|Drive|Dr|Court|Ct|Parkway|(?:Highway|Hwy\.?)\s+\d+)"
_ADDR_RE = re.compile(r"\b(\d{2,6}\s+(?:[A-Z0-9][\w'’.]*\s+){0,4}?" + _STREET_SUFFIX + r"\b\.?(?:\s+(?:SE|SW|NE|NW|[NSEW])\b\.?)?)")
_DATELINE_RE = re.compile(r"^([A-Z][A-Z .]{2,30}),\s*(Iowa|Wis\.|Wisconsin|Ill\.|Illinois)\s*[—–-]+\s*")
_CONTACT_BRANCH_RE = re.compile(r"###\s*Dupaco Credit Union\s*\n+\s*Dupaco Branch\s*\n+\s*([^\n]+)")
_CONTACT_ADDR_RE = re.compile(r"^(.+?),\s*([A-Za-z .]+?),\s*([A-Z]{2}),?\s*(\d{5})")
_BRANCH_LIST_RE = re.compile(r"Select a Branch\s*([^\n]+)")
_BRANCH_LIST_ITEM_RE = re.compile(r"([A-Z][a-z]+(?: [A-Z][a-z]+)?): ")
_PHONE_RE = re.compile(r"phone number (?:for [^.]{0,60}? )?is (\(\d{3}\) \d{3}-\d{4}|\d{3}-\d{3}-\d{4})")
_HOURS_BLOCK_RE = re.compile(r"\*\*Lobby\*\*\s*(.+?)\s*\*\*Drive[- ]?up\*\*\s*(.+?)(?=\*\*[A-Z]|$|\n\s*#)", re.S)
_DAY_LINE_RE = re.compile(r"^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)[-–—\w ]*:\s*\S")
# sentence split that does not break inside "1090 E. Hickman Road" or "807 Ninth St. SE"
_SENT_SPLIT_RE = re.compile(r"(?<![A-Z]\.)(?<!\bSt\.)(?<!Ave\.)(?<!Rd\.)(?<!Dr\.)(?<!Hwy\.)(?<!Blvd\.)(?<=[.!?])\s+(?=[A-Z“\"])")
_ARTICLE_END_RE = re.compile(r"\*\*ABOUT DUPACO|\n\s*#{3,4}\s*(?:\n|\[)|\n\s*\\?# \\?# \\?#")
_NOT_BRANCH_RE = re.compile(r"innovation lab|voices building|headquarter|operations center|coworking|purchased the|plans to build|groundbreaking", re.I)


def street_tokens(s: str) -> list[str]:
    toks = re.findall(r"[a-z0-9][a-z0-9'’-]*", s.lower().replace(".", " "))
    return [_STREET_ABBR.get(t, t) for t in toks]


def street_key(s: str) -> set[str]:
    """Distinctive street words: no house number, suffixes or compass letters ('807 Ninth St. SE' -> {'9th'})."""
    toks = street_tokens(s)
    if toks and toks[0].isdigit() and len(toks) > 1:
        toks = toks[1:]
    return {t for t in toks if t not in STREET_GENERIC}


def _clean_hours(s: str) -> str:
    lines = [ln.strip() for ln in re.split(r"\n+|(?<=\.)\s+(?=(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))", s) if ln.strip()]
    keep = []
    for ln in lines:
        if not _DAY_LINE_RE.match(ln):
            break
        keep.append(ln)
    s = " ".join(keep)
    s = re.sub(r"\s+", " ", s).strip().replace("–", "-").replace("—", "-")
    return re.sub(r"\s+(?=(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)[-\w]*:)", "; ", s)


def _city_title(s: str) -> str:
    return " ".join(w.capitalize() for w in s.strip().split())


def _closing_date(sent: str) -> Optional[str]:
    m = _LONG_DATE_RE.search(sent)
    return parse_long_date(m) if m else None


def extract_branches(pages: dict) -> list[dict]:
    """Build the branch table for one snapshot from the pages that name branch locations.

    Sources, strongest first: the /contact locator list (street, city, state, zip); newsroom releases about a
    branch (street with number, phone, Lobby/Drive-up hours, closures); the "Select a Branch" City: Street list
    (every branch, street name only). Rows merge on (city, distinctive street words).
    """
    rows: list[dict] = []

    def find(city: str, key: set, context: str = "") -> Optional[dict]:
        cands = [r for r in rows if r["city"].lower() == city.lower()]
        ctx = street_key(context) if context else set()
        for r in cands:
            if key and (key & r["key"] or key & r["alias_key"]):
                return r
        for r in cands:
            if ctx and (ctx & r["key"] or ctx & r["alias_key"]):
                return r
        return None

    def new_row(city: str) -> dict:
        r = {"name": _city_title(city), "city": _city_title(city), "street": "", "street_rank": 0, "state": "", "zip": "",
             "phone": "", "lobby": "", "drive_up": "", "hours_date": "", "hours_url": "", "closed": "", "closed_on": "",
             "aliases": [], "key": set(), "alias_key": set(), "sources": []}
        rows.append(r)
        return r

    def set_street(r: dict, street: str, rank: int) -> None:
        if rank > r["street_rank"]:
            r["street"], r["street_rank"] = street.strip().rstrip(","), rank
        r["key"] |= street_key(street)

    def add_src(r: dict, url: str) -> None:
        if url not in r["sources"]:
            r["sources"].append(url)

    # 1. "Select a Branch" City: Street list (complete list, street names only)
    for p in pages.values():
        for m in _BRANCH_LIST_RE.finditer(p.text):
            body = m.group(1)
            items = list(_BRANCH_LIST_ITEM_RE.finditer(body))
            for i, it in enumerate(items):
                city = it.group(1)
                hint = body[it.end(): items[i + 1].start() if i + 1 < len(items) else len(body)].strip()
                if not hint:
                    continue
                alias = ""
                if "," in hint:
                    alias, hint = [x.strip() for x in hint.split(",", 1)]
                hy = re.search(r"\bHy-?Vee\b", hint, re.I)
                if hy:
                    alias = alias or "Hy-Vee"
                    hint = (hint[:hy.start()] + hint[hy.end():]).strip()
                key = street_key(hint) | street_key(alias)
                r = find(city, key)
                if r is None:
                    r = new_row(city)
                set_street(r, hint, 1)
                if not set(street_tokens(hint)) & {"st", "rd", "ave", "blvd", "hwy", "dr", "ln", "way", "ct", "pkwy"}:
                    alias = alias or hint          # 'Edgewood', 'San Marnan': a branch name more than a street
                if alias and alias not in r["aliases"]:
                    r["aliases"].append(alias)
                    r["alias_key"] |= street_key(alias)
                add_src(r, p.url)

    # 2. /contact locator list (authoritative street, state, zip for the branches it shows)
    for p in pages.values():
        if not p.url.rstrip("/").endswith("/contact"):
            continue
        for m in _CONTACT_BRANCH_RE.finditer(p.text):
            a = _CONTACT_ADDR_RE.match(m.group(1).strip())
            if not a:
                continue
            street, city, st, z = a.group(1), a.group(2), a.group(3), a.group(4)
            r = find(city, street_key(street)) or new_row(city)
            set_street(r, street, 3)
            r["state"], r["zip"] = st, z
            add_src(r, p.url)

    # 3. newsroom releases: branch street address, phone, hours, closures
    news = sorted((p for p in pages.values() if p.kind == "newsroom" and "/tag/" not in p.url and "/author/" not in p.url),
                  key=lambda p: (p.content_date or "", p.url))
    for p in news:
        body = _ARTICLE_END_RE.split(p.text, 1)[0]      # the release itself, not the related-article previews
        paras = [x.strip() for x in body.split("\n\n") if x.strip() and not x.strip().endswith("...")]
        dl_city, dl_state = "", ""
        for para in paras[:12]:
            d = _DATELINE_RE.match(para)
            if d:
                dl_city, dl_state = _city_title(d.group(1)), STATE_ABBR.get(d.group(2).lower(), "")
                break
        closing = bool(re.search(r"\bclos(e|es|ing|ure)\b", p.title, re.I))
        primary = None
        for para in paras:
            if para.startswith("**") or para.startswith("#"):
                continue
            for sent in _SENT_SPLIT_RE.split(para):
                if not re.search(r"\bbranch|learning lab", sent, re.I) or _NOT_BRANCH_RE.search(sent):
                    continue
                for am in _ADDR_RE.finditer(sent):
                    street = am.group(1).rstrip(".") + ("." if am.group(1).endswith(".") else "")
                    tail = sent[am.end():am.end() + 40]
                    cm = re.match(r",?\s*(?:in|,)?\s*([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b(?:,\s*(Iowa|Wis\.|Ill\.|Illinois|Wisconsin))?", tail)
                    city = dl_city
                    if cm and (re.match(r",?\s*in\s", tail) or re.match(r",\s*[A-Z]", tail)) and cm.group(1) not in ("Dupaco", "Highway", "Interstate"):
                        city = cm.group(1)
                    if not city:
                        continue
                    r = find(city, street_key(street), sent)
                    if r is None:
                        r = new_row(city)
                    set_street(r, street, 2)
                    if not r["state"]:
                        st = cm.group(2) if cm and cm.group(2) else None
                        r["state"] = STATE_ABBR.get(st.lower(), "") if st else (dl_state if city == dl_city else "")
                    add_src(r, p.url)
                    if primary is None:
                        primary = r
                    if closing and re.search(r"last day|will close|clos(e|ed|ing)", sent, re.I):
                        r["closed"], r["closed_on"] = sent.strip(), _closing_date(sent) or ""
        if closing and primary is None:
            # closure release without a street number ('branch inside the South Locust St. Hy-Vee store')
            for r in rows:
                names = [a for a in r["aliases"] if a]
                if any(re.search(r"\b" + re.escape(a) + r"\b", p.title, re.I) for a in names):
                    sent = next((s for para in paras for s in _SENT_SPLIT_RE.split(para)
                                 if re.search(r"\bwill close\b|last day", s, re.I)), "")
                    if sent:
                        r["closed"], r["closed_on"] = sent.strip(), _closing_date(sent) or ""
                        add_src(r, p.url)
        if primary is None:
            continue
        pm = _PHONE_RE.search(body)
        if pm:
            primary["phone"] = pm.group(1)
        hm = _HOURS_BLOCK_RE.search(body)
        if hm:
            lobby, drive = _clean_hours(hm.group(1)), _clean_hours(hm.group(2))
            if ("a.m." in lobby or "p.m." in lobby) and (p.content_date or "") >= primary["hours_date"]:
                primary.update(lobby=lobby, drive_up=drive, hours_date=p.content_date or "", hours_url=p.url)

    # state for rows that only came from the City: Street list: the state the site most often writes after the city
    for r in rows:
        if r["state"]:
            continue
        cnt: Counter = Counter()
        pat = re.compile(r"\b" + re.escape(r["city"]) + r",?\s+(Iowa|IA|Wis\.|Wisconsin|WI|Ill\.|Illinois|IL)\b")
        for p in pages.values():
            for m in pat.finditer(p.text):
                cnt[STATE_ABBR[m.group(1).lower()]] += 1
        if cnt:
            r["state"] = cnt.most_common(1)[0][0]
    # display names: city alone when unique, otherwise city plus the branch's street or alias
    per_city = Counter(r["city"] for r in rows)
    for r in rows:
        r["key"] = frozenset(r["key"])
        r["alias_key"] = frozenset(r["alias_key"])
        if per_city[r["city"]] > 1:
            label = r["aliases"][0] if r["aliases"] else re.sub(r"^\d+\s+", "", r["street"])
            r["name"] = f"{r['city']} ({label})"
    rows.sort(key=lambda r: (r["city"], r["street"]))
    return rows


# ----------------------------------------------------------------------------- "About Dupaco" facts

# Press releases end with an "ABOUT DUPACO COMMUNITY CREDIT UNION" paragraph whose numbers change from year to
# year (members, assets, employees, counties, branch count). The most recently dated release is current.
_ABOUT_PARA_RE = re.compile(r"Dupaco Community Credit Union is a not-for-profit, member-owned financial cooperative[^\n]*")
ABOUT_FIELDS = {
    "hq": re.compile(r"headquartered in ([A-Z][a-z]+, [A-Z][a-z]+)"),
    "counties": re.compile(r"serves residents in (\d+) counties throughout ([^.]+)\."),
    "founded": re.compile(r"(?:Founded|Chartered) in (\d{4})(?: by ((?:ten|\d+) [A-Z][\w ]+? employees))?"),
    "members": re.compile(r"membership (?:has grown to|totals) ((?:more than|almost|over|nearly|about) )?([\d,]{5,})"),
    "assets": re.compile(r"assets (?:exceeding|of nearly|of more than|of) (\$[\d.]+ billion)"),
    "employees": re.compile(r"(?:It|Dupaco) has ((?:over|more than|nearly|almost|about) )?(\d[\d,]*) employees"),
    "recognition": re.compile(r"(?:It|Dupaco) is a (Forbes [^.]+)\."),
}
_CEO_RES = (
    re.compile(r"(.{0,40})\b(?:President(?:,| and| &|/) ?CEO|CEO and President)\b,? ([A-Z][a-z]+(?: [A-Z]\.)? [A-Z][a-z]+)"),
    re.compile(r"(.{0,40})\b([A-Z][a-z]+ [A-Z][a-z]+), (?:Dupaco(?:’s|'s)? )?(?:President(?:,| and| &|/) ?CEO|CEO)\b"),
)
ABOUT_GROWING = ("members", "assets")


def _about_num(v: str) -> float:
    m = re.search(r"[\d,]+(?:\.\d+)?", v)
    return float(m.group(0).replace(",", "")) if m else 0.0


_FORMER_RE = re.compile(r"\b(former|long-?time|retired|late|then|interim|emeritus)\b", re.I)


def extract_about(pages: dict) -> dict:
    """{field: {"value", "detail", "sentence", "url", "date", "urls"}} from the newest press-release boilerplate,
    plus "ceo" from "President and CEO <Name>" mentions (most recent page wins, former/long-time titles skipped)."""
    found: dict[str, list] = defaultdict(list)
    for p in pages.values():
        for m in _ABOUT_PARA_RE.finditer(p.text):
            para = m.group(0)
            for field, rx in ABOUT_FIELDS.items():
                fm = rx.search(para)
                if not fm:
                    continue
                if field in ("members", "employees"):
                    value, detail = fm.group(2), (fm.group(1) or "").strip()
                elif field == "counties":
                    value, detail = fm.group(1), fm.group(2).strip()
                elif field == "founded":
                    value, detail = fm.group(1), (fm.group(2) or "").strip()
                else:
                    value, detail = fm.group(1), ""
                sent = next((x for x in re.split(r"(?<=\.)\s+(?=[A-Z])", para) if fm.group(0)[:20] in x), fm.group(0))
                found[field].append((p.content_date or "", p.authority, p.url, value, detail, sent.strip()))
    out: dict[str, dict] = {}
    for field, hits in found.items():
        hits.sort(reverse=True)
        if field in ABOUT_GROWING and hits[0][0]:
            # Member and asset totals only grow. A release that is newer but quotes a smaller total reused an older
            # boilerplate, so among recent releases (120 days) take the largest figure, newest first on ties.
            try:
                cut = (date.fromisoformat(hits[0][0]) - timedelta(days=120)).isoformat()
                recent = [h for h in hits if h[0] >= cut]
                hits = sorted(recent, key=lambda h: (_about_num(h[3]), h[0]), reverse=True) + [h for h in hits if h[0] < cut]
            except ValueError:
                pass
        d, _, url, value, detail, sent = hits[0]
        urls = [h[2] for h in hits if h[3] == value]
        out[field] = {"value": value, "detail": detail, "sentence": sent, "url": url, "date": d, "urls": urls}
    names: dict[str, list] = defaultdict(list)
    for p in pages.values():
        for rx in _CEO_RES:
            for m in rx.finditer(p.text):
                name = m.group(2)
                lead = re.split(r"[.!?]\s", m.group(1))[-1]      # only the same sentence says "former"/"long-time"
                if _FORMER_RE.search(lead) or re.search(r"\b(Dupaco|President|Credit|Union|Chief)\b", name):
                    continue
                names[name].append((p.content_date or "", p.url, bool(re.search(re.escape(name), p.title))))
    if names:
        best = max(names, key=lambda n: (max(h[0] for h in names[n]), len(names[n])))
        hs = sorted(names[best], key=lambda h: (not h[2], h[0] == "", [-ord(c) for c in h[0]]))
        urls = []
        for h in hs:
            if h[1] not in urls:
                urls.append(h[1])
        out["ceo"] = {"value": best, "detail": "", "sentence": "", "url": urls[0], "date": max(h[0] for h in names[best]), "urls": urls}
    return out


# ----------------------------------------------------------------------------- product sections

# A heading on a product page that names a product ("### 1st Rate Checking", "### Health Savings Accounts").
_PRODUCT_HEAD_RE = re.compile(r"(checking|savings|accounts?|cards?|loans?|certificates?|club|market|iras?|visa|protection|privilege|"
                              r"mortgages?|line of credit|heloc|insurance|program)$", re.I)
PRODUCT_GENERIC = {"account", "dupaco", "your", "our", "find", "kind", "access", "manag", "create", "custom", "using", "use", "who", "on",
                   "frequently", "asked", "question", "about", "little", "backup", "grow", "with", "flexibl", "flexibility", "power",
                   "build", "skip", "extra", "home", "for"}


def product_sections(sentences: list) -> dict:
    """{heading: {"name", "toks", "urls"}} for product-name headings on product (or help) pages."""
    out: dict[str, dict] = {}
    for st in sentences:
        p = st.page
        h = st.section
        if not h or p.kind not in ("product", "other") or "/resources/" in p.path or len(h.split()) > 5:
            continue
        if not _PRODUCT_HEAD_RE.search(h) or re.match(r"\d+\\?\.\s|frequently", h, re.I):
            continue
        d = out.get(h)
        if d is None:
            toks = frozenset(tokenize(h)) - {"account"}
            if not toks or toks <= PRODUCT_GENERIC:
                continue
            d = out[h] = {"name": h, "toks": toks, "urls": []}
        if p.url not in d["urls"]:
            d["urls"].append(p.url)
    return out


class SnapshotView:
    """Everything known about the website as of one snapshot."""

    def __init__(self, snapshot_id: str, taken_on: str):
        self.snapshot_id = snapshot_id
        self.taken_on = taken_on
        self.pages: dict[str, PageVer] = {}
        self.facts: list[Fact] = []
        self.prose: list[Fact] = []
        self.by_key: dict[str, list[Fact]] = defaultdict(list)
        self.by_family: dict[str, list[Fact]] = defaultdict(list)
        self.sentences: list[Sentence] = []
        self.bm25: Optional[BM25] = None
        self.links: Counter = Counter()           # (anchor_lower, url) -> count
        self.link_pages: dict[tuple, set] = defaultdict(set)
        self.page_sents: dict[str, list] = {}
        self.boiler: set[str] = set()
        self.branches: list[dict] = []           # one dict per branch, see extract_branches
        self.about: dict[str, dict] = {}         # "About Dupaco" facts, see extract_about
        self.products: dict[str, dict] = {}      # product-name headings on product pages, see product_sections

    def add_page(self, p: PageVer) -> None:
        self.pages[p.url] = p

    def finalize(self) -> None:
        for p in self.pages.values():
            for f in facts_from_page(p):
                self.facts.append(f)
                self.by_key[f.key].append(f)
                self.by_family[f.family].append(f)
            self.prose.extend(prose_facts_from_page(p))
            for a, u in p.links:
                self.links[(a.lower().strip(), u)] += 1
                self.link_pages[(a.lower().strip(), u)].add(p.path)
        # sentence index with boilerplate removal (a sentence on many pages is navigation/footer)
        sent_pages: Counter = Counter()
        per_page: list[tuple[PageVer, list[str]]] = []
        for p in self.pages.values():
            ss = split_sentences_h(p.text, wrapped=p.url.lower().split("?")[0].endswith(".pdf"))
            per_page.append((p, ss))
            for s in {x for x, _ in ss}:
                sent_pages[s] += 1
        thresh = max(8, int(0.03 * len(self.pages)))
        self.boiler = {s for s, c in sent_pages.items() if c >= thresh and not re.search(r"\d\d branch office", s)}
        docs = []
        for p, ss in per_page:
            seen = set()
            plist = []
            for s, head in ss:
                if s in self.boiler or s in seen:
                    continue
                seen.add(s)
                toks = tokenize(s)
                if len(toks) < 2:
                    continue
                sent = Sentence(s, p, toks, len(plist), head, len(self.sentences))
                plist.append(sent)
                self.sentences.append(sent)
                docs.append(toks)
            self.page_sents[p.url] = plist
        self.bm25 = BM25(docs) if docs else None
        self.products = product_sections(self.sentences)
        self.branches = extract_branches(self.pages)
        self.about = extract_about(self.pages)


class KnowledgeBase:
    def __init__(self):
        self.views: list[SnapshotView] = []

    def ingest(self, snapshot) -> None:
        v = SnapshotView(snapshot.snapshot_id, snapshot.taken_on)
        for pg in snapshot.pages:
            v.add_page(make_pagever(pg.url, pg.title, pg.markdown, pg.fetched_at, snapshot.snapshot_id, snapshot.taken_on))
        v.finalize()
        self.views.append(v)
        self.views.sort(key=lambda x: x.taken_on)

    def view_for(self, as_of: Optional[str]) -> Optional[SnapshotView]:
        if not self.views:
            return None
        if not as_of:
            return self.views[-1]
        as_of = as_of[:10]
        cands = [v for v in self.views if v.taken_on <= as_of]
        return cands[-1] if cands else self.views[0]
