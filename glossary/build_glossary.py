"""
Build a German medical glossary from transcript files in this repo.

Inputs:
- transcripts_dir: transcript files (.txt/.md/.json)
- manual_overrides: optional mapping for lay terms
- blacklist_path: remove generic terms/phrases

Outputs:
- public/glossary.csv
- public/glossary.json
- public/wikidata_cache.json (search/entity cache)
- public/glossary.todo.csv (top N items with missing lay_terms)
- public/blacklist_suggestions.csv (top N frequent non-medical tokens)

Wikidata:
- Uses labels/aliases/descriptions in German with retries/backoff.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import requests
from rapidfuzz import fuzz

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

GERMAN_STOPWORDS: Set[str] = {
    "und", "oder", "aber", "auch", "dann", "denn", "dass", "das", "der", "die", "ein", "eine", "einen",
    "einem", "einer", "ich", "wir", "sie", "ihr", "ihnen", "du", "er", "es", "ist", "sind", "war", "waren",
    "haben", "hat", "hatte", "hatten", "nicht", "kein", "keine", "mit", "auf", "für", "im", "in", "am", "an",
    "zu", "vom", "von", "als", "bei", "so", "wie", "was", "wo", "wann", "warum", "wieso", "bitte", "danke",
    "okay", "genau", "ja", "nein", "mal", "noch", "schon", "jetzt",
}

MED_SUFFIXES = (
    "itis", "ose", "ämie", "aemie", "pathie", "kardie", "tomie", "omie", "ektomie", "skopie",
    "embolie", "thrombose", "infarkt", "syndrom", "stenose", "insuffizienz", "hypertonie", "hypotonie",
)

MED_SUBSTRINGS = (
    "pankreat", "cholezyst", "append", "koronar", "arter", "vene", "myokard", "vorhoffl",
    "tachy", "brady", "dyspno", "synkop", "angina", "tropon", "bilirub", "creatin",
)

LAY_HINT_SUBSTRINGS = (
    "schmerz", "entzünd", "atem", "luft", "herz", "bauch", "blinddarm", "galle", "niere", "leber", "lunge",
    "übel", "brechen", "fieber", "schwindel", "blutdruck", "zucker", "wasser",
)

ABBREV_OK = {
    "ekg", "rr", "spo2", "ct", "mrt", "crp", "hba1c", "inr", "ptt", "aptt", "bga", "ck", "ckmb",
    "ldh", "ast", "alt", "ggt", "ntpro", "nt-pro", "ntprobnp", "nt-probnp",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--cache_json", required=True)
    ap.add_argument("--manual_overrides", default="")
    ap.add_argument("--blacklist_path", default="glossary/blacklist.txt")

    ap.add_argument("--out_todo_csv", default="public/glossary.todo.csv")
    ap.add_argument("--todo_n", type=int, default=20)

    ap.add_argument("--out_blacklist_suggestions_csv", default="public/blacklist_suggestions.csv")
    ap.add_argument("--blacklist_suggestions_n", type=int, default=50)

    ap.add_argument("--max_terms", type=int, default=300)
    ap.add_argument("--min_count", type=int, default=2)
    ap.add_argument("--max_ngram", type=int, default=4)

    ap.add_argument("--sleep_s", type=float, default=0.2)
    ap.add_argument("--max_file_mb", type=float, default=5.0)
    ap.add_argument("--http_timeout_s", type=float, default=30.0)
    ap.add_argument("--http_retries", type=int, default=6)
    return ap.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8", errors="ignore") or "{}")


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_text_from_json(text: str) -> str:
    try:
        obj = json.loads(text)
    except Exception:
        return text

    if isinstance(obj, dict):
        for k in ("transcript", "text", "content", "body"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v
        segs = obj.get("segments")
        if isinstance(segs, list):
            parts = []
            for s in segs:
                if isinstance(s, dict):
                    t = s.get("text")
                    if isinstance(t, str) and t.strip():
                        parts.append(t.strip())
            if parts:
                return " ".join(parts)
        return text

    if isinstance(obj, list):
        parts = []
        for it in obj:
            if isinstance(it, str) and it.strip():
                parts.append(it.strip())
            elif isinstance(it, dict):
                t = it.get("text") or it.get("transcript")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
        return " ".join(parts) if parts else text

    return text


def clean_transcript(raw: str) -> str:
    t = raw or ""
    t = re.sub(r"(?s)\A---.*?---\s*", "", t)
    t = re.sub(r"\[\s*\d{1,2}:\d{2}(?::\d{2})?\s*\]", " ", t)
    t = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", t)
    t = re.sub(r"(?m)^\s*#{1,6}\s+", "", t)
    t = re.sub(r"(?m)^\s*[-*]\s+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def iter_transcript_texts(transcripts_dir: Path, max_file_mb: float) -> Iterable[str]:
    exts = {".txt", ".md", ".json"}
    for p in transcripts_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        try:
            if p.stat().st_size > max_file_mb * 1024 * 1024:
                continue
        except Exception:
            continue

        raw = read_text_file(p)
        if p.suffix.lower() == ".json":
            raw = extract_text_from_json(raw)
        cleaned = clean_transcript(raw)
        if cleaned:
            yield cleaned


def word_tokens(text: str) -> List[str]:
    return re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿÄÖÜäöüß\-]+", text or "")


def load_blacklist(path: str) -> Tuple[Set[str], Set[str]]:
    p = Path(path)
    if not p.exists():
        return set(), set()

    phrase_bl: Set[str] = set()
    token_bl: Set[str] = set()

    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        s = re.sub(r"\s+", " ", s).lower()
        phrase_bl.add(s)
        if " " not in s:
            token_bl.add(s)

    return phrase_bl, token_bl


def is_blacklisted(term_lc: str, phrase_bl: Set[str], token_bl: Set[str]) -> bool:
    t = re.sub(r"\s+", " ", (term_lc or "").strip().lower())
    if not t:
        return False
    if t in phrase_bl:
        return True
    tokens = t.split()
    return any(tok in token_bl for tok in tokens)


def is_medicalish_word(tok: str) -> bool:
    t = (tok or "").strip()
    if not t:
        return False
    tl = t.lower()

    if tl in GERMAN_STOPWORDS:
        return False
    if tl in ABBREV_OK:
        return True
    if t.isupper() and 2 <= len(t) <= 10:
        return True
    if any(tl.endswith(s) for s in MED_SUFFIXES):
        return True
    if any(s in tl for s in MED_SUBSTRINGS):
        return True
    if any(ch.isdigit() for ch in t) and re.fullmatch(r"\d+[a-zA-Z%]+", t):
        return True

    # Capitalized words are too broad; keep only longer proper nouns if needed.
    if t[0].isupper() and len(t) >= 9:
        return True

    return False


def is_candidate_ngram(tokens: List[str]) -> bool:
    if not tokens:
        return False
    if all(t.lower() in GERMAN_STOPWORDS for t in tokens):
        return False

    joined = " ".join(tokens).strip()
    if len(joined) < 3 or len(joined) > 90:
        return False

    if not any(is_medicalish_word(t) for t in tokens):
        return False

    if all(re.fullmatch(r"\d+", t) for t in tokens):
        return False

    return True


def extract_candidates(
    texts: Iterable[str],
    max_ngram: int,
    phrase_bl: Set[str],
    token_bl: Set[str],
) -> Dict[str, int]:
    max_ngram = int(max(1, max_ngram))
    counts: Dict[str, int] = {}

    for txt in texts:
        toks = word_tokens(txt)

        for w in toks:
            if is_candidate_ngram([w]):
                k = w.lower()
                if not is_blacklisted(k, phrase_bl, token_bl):
                    counts[k] = counts.get(k, 0) + 1

        for n in range(2, max_ngram + 1):
            for i in range(0, max(0, len(toks) - n + 1)):
                ng = toks[i : i + n]
                if not is_candidate_ngram(ng):
                    continue
                k = " ".join(x.lower() for x in ng)
                if is_blacklisted(k, phrase_bl, token_bl):
                    continue
                counts[k] = counts.get(k, 0) + 1

    return counts


def extract_token_frequencies(texts: Iterable[str]) -> Dict[str, int]:
    """
    Raw token frequencies from transcripts (for blacklist suggestions).
    """
    counts: Dict[str, int] = {}
    for txt in texts:
        for tok in word_tokens(txt):
            tl = tok.lower()
            if not tl:
                continue
            counts[tl] = counts.get(tl, 0) + 1
    return counts


def generate_blacklist_suggestions(
    token_counts: Dict[str, int],
    phrase_bl: Set[str],
    token_bl: Set[str],
    n: int,
) -> pd.DataFrame:
    """
    Suggest frequent non-medical tokens that are not already blacklisted.
    """
    rows: List[Dict[str, Any]] = []
    for tok, freq in token_counts.items():
        if freq <= 0:
            continue
        if tok in GERMAN_STOPWORDS:
            continue
        if len(tok) < 3:
            continue
        already = tok in token_bl or tok in phrase_bl
        if already:
            continue
        if tok in ABBREV_OK:
            continue
        if is_medicalish_word(tok):
            continue
        rows.append(
            {
                "token": tok,
                "freq": int(freq),
                "why": "frequent_non_medical",
            }
        )

    rows.sort(key=lambda r: r["freq"], reverse=True)
    df = pd.DataFrame(rows[: max(0, int(n))])
    return df


@dataclass(frozen=True)
class WikidataEntity:
    qid: str
    label_de: str
    description_de: str
    aliases_de: List[str]


def _sleep_with_jitter(seconds: float) -> None:
    if seconds <= 0:
        return
    time.sleep(seconds + random.random() * min(0.25, seconds))


def http_get_json(
    session: requests.Session,
    params: Dict[str, Any],
    timeout_s: float,
    retries: int,
    base_sleep_s: float,
) -> Dict[str, Any]:
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            r = session.get(WIKIDATA_API, params=params, timeout=timeout_s)
            if r.status_code in (429, 500, 502, 503, 504):
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    _sleep_with_jitter(float(retry_after))
                else:
                    _sleep_with_jitter(base_sleep_s * (2 ** (attempt - 1)))
                continue

            r.raise_for_status()
            try:
                return r.json()
            except Exception as e:
                last_exc = e
                _sleep_with_jitter(base_sleep_s * (2 ** (attempt - 1)))
                continue

        except requests.RequestException as e:
            last_exc = e
            _sleep_with_jitter(base_sleep_s * (2 ** (attempt - 1)))
            continue

    raise RuntimeError(f"Wikidata request failed after {retries} retries: {last_exc}") from last_exc


def wikidata_search(
    session: requests.Session,
    term: str,
    timeout_s: float,
    retries: int,
    base_sleep_s: float,
) -> Optional[str]:
    params = {"action": "wbsearchentities", "search": term, "language": "de", "limit": 5, "format": "json"}
    data = http_get_json(session, params=params, timeout_s=timeout_s, retries=retries, base_sleep_s=base_sleep_s)
    results = data.get("search") or []
    if not results:
        return None
    return str(results[0].get("id") or "") or None


def wikidata_get_entity(
    session: requests.Session,
    qid: str,
    timeout_s: float,
    retries: int,
    base_sleep_s: float,
) -> Optional[WikidataEntity]:
    params = {
        "action": "wbgetentities",
        "ids": qid,
        "props": "labels|aliases|descriptions",
        "languages": "de",
        "format": "json",
    }
    data = http_get_json(session, params=params, timeout_s=timeout_s, retries=retries, base_sleep_s=base_sleep_s)
    ent = (data.get("entities") or {}).get(qid)
    if not isinstance(ent, dict):
        return None

    label = ((ent.get("labels") or {}).get("de") or {}).get("value") or ""
    desc = ((ent.get("descriptions") or {}).get("de") or {}).get("value") or ""
    aliases = [
        a.get("value")
        for a in ((ent.get("aliases") or {}).get("de") or [])
        if isinstance(a, dict) and a.get("value")
    ]
    aliases = [str(a) for a in aliases]
    return WikidataEntity(qid=qid, label_de=str(label), description_de=str(desc), aliases_de=aliases)


def best_qid_for_term(
    session: requests.Session,
    term: str,
    cache: Dict[str, Any],
    sleep_s: float,
    timeout_s: float,
    retries: int,
) -> Optional[str]:
    key = f"search:{term.lower()}"
    if key in cache:
        return cache[key] or None

    qid = wikidata_search(session, term, timeout_s=timeout_s, retries=retries, base_sleep_s=sleep_s)
    cache[key] = qid or ""
    _sleep_with_jitter(sleep_s)
    return qid


def get_entity_cached(
    session: requests.Session,
    qid: str,
    cache: Dict[str, Any],
    sleep_s: float,
    timeout_s: float,
    retries: int,
) -> Optional[WikidataEntity]:
    key = f"entity:{qid}"
    if key in cache:
        raw = cache[key]
        if not raw:
            return None
        return WikidataEntity(
            qid=qid,
            label_de=raw.get("label_de", ""),
            description_de=raw.get("description_de", ""),
            aliases_de=list(raw.get("aliases_de", [])),
        )

    ent = wikidata_get_entity(session, qid, timeout_s=timeout_s, retries=retries, base_sleep_s=sleep_s)
    cache[key] = (
        {"label_de": ent.label_de, "description_de": ent.description_de, "aliases_de": ent.aliases_de} if ent else ""
    )
    _sleep_with_jitter(sleep_s)
    return ent


def looks_technical(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return False
    tl = t.lower()
    if tl in ABBREV_OK:
        return True
    if any(tl.endswith(x) for x in MED_SUFFIXES):
        return True
    if any(x in tl for x in MED_SUBSTRINGS):
        return True
    if " " not in t and len(t) >= 9 and t[0].isupper():
        return True
    return False


def looks_lay(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return False
    tl = t.lower()
    if any(x in tl for x in LAY_HINT_SUBSTRINGS):
        return True
    if " " in t and len(t) >= 6:
        return True
    return False


def load_manual_overrides(path: str) -> Dict[str, str]:
    p = Path(path) if path else None
    if not p or not p.exists():
        return {}
    df = pd.read_csv(p, comment="#")
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        term = str(row.get("term_in_corpus") or "").strip().lower()
        lay = str(row.get("preferred_lay_term") or "").strip()
        if term and lay:
            out[term] = lay
    return out


def write_todo_csv(df: pd.DataFrame, out_path: str, todo_n: int) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _is_empty(v: Any) -> bool:
        s = str(v or "").strip()
        return s == ""

    todo = df[df["lay_terms"].apply(_is_empty)].copy()
    if todo.empty:
        pd.DataFrame(columns=["term_in_corpus", "freq", "qid", "label_de", "description_de"]).to_csv(out, index=False)
        return

    todo = todo.sort_values(["freq", "term_in_corpus"], ascending=[False, True], kind="mergesort")
    todo = todo.head(max(0, int(todo_n)))
    todo[["term_in_corpus", "freq", "qid", "label_de", "description_de"]].to_csv(out, index=False, encoding="utf-8")


def main() -> int:
    args = parse_args()

    transcripts_dir = Path(args.transcripts_dir)
    if not transcripts_dir.exists():
        raise SystemExit(f"Missing transcripts dir: {transcripts_dir}")

    phrase_bl, token_bl = load_blacklist(args.blacklist_path)

    texts = list(iter_transcript_texts(transcripts_dir, max_file_mb=float(args.max_file_mb)))
    if not texts:
        raise SystemExit(f"No transcripts found under: {transcripts_dir}")

    # Blacklist suggestions (raw token frequency)
    token_freq = extract_token_frequencies(texts)
    bl_df = generate_blacklist_suggestions(
        token_counts=token_freq,
        phrase_bl=phrase_bl,
        token_bl=token_bl,
        n=int(args.blacklist_suggestions_n),
    )
    bl_out = Path(args.out_blacklist_suggestions_csv)
    bl_out.parent.mkdir(parents=True, exist_ok=True)
    bl_df.to_csv(bl_out, index=False, encoding="utf-8")

    # Glossary candidates
    cand = extract_candidates(
        texts=texts,
        max_ngram=int(args.max_ngram),
        phrase_bl=phrase_bl,
        token_bl=token_bl,
    )
    items = [(t, c) for t, c in cand.items() if c >= int(args.min_count)]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[: int(args.max_terms)]

    cache_path = Path(args.cache_json)
    cache = load_json(cache_path)
    if not isinstance(cache, dict):
        cache = {}

    overrides = load_manual_overrides(args.manual_overrides)

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "FSPtranskript-GlossaryBot/1.1 (+https://github.com/protokolFSP/FSPtranskript)"}
    )

    rows: List[Dict[str, Any]] = []

    for idx, (term_lc, freq) in enumerate(items, start=1):
        try:
            qid = best_qid_for_term(
                session=session,
                term=term_lc,
                cache=cache,
                sleep_s=float(args.sleep_s),
                timeout_s=float(args.http_timeout_s),
                retries=int(args.http_retries),
            )
        except Exception as e:
            print(f"[WARN] search failed for term='{term_lc}': {e}")
            qid = None

        if not qid:
            rows.append(
                {
                    "term_in_corpus": term_lc,
                    "freq": freq,
                    "qid": "",
                    "label_de": "",
                    "description_de": "",
                    "technical_terms": "",
                    "lay_terms": overrides.get(term_lc, ""),
                    "aliases_de": "",
                    "source": "",
                }
            )
            continue

        try:
            ent = get_entity_cached(
                session=session,
                qid=qid,
                cache=cache,
                sleep_s=float(args.sleep_s),
                timeout_s=float(args.http_timeout_s),
                retries=int(args.http_retries),
            )
        except Exception as e:
            print(f"[WARN] entity failed for qid='{qid}' term='{term_lc}': {e}")
            ent = None

        if not ent:
            rows.append(
                {
                    "term_in_corpus": term_lc,
                    "freq": freq,
                    "qid": qid,
                    "label_de": "",
                    "description_de": "",
                    "technical_terms": "",
                    "lay_terms": overrides.get(term_lc, ""),
                    "aliases_de": "",
                    "source": "Wikidata",
                }
            )
            continue

        syns = [ent.label_de] + ent.aliases_de
        syns = [s for s in syns if isinstance(s, str) and s.strip()]
        seen: Set[str] = set()
        syns2: List[str] = []
        for s in syns:
            sl = s.lower()
            if sl in seen:
                continue
            seen.add(sl)
            syns2.append(s)

        technical = [s for s in syns2 if looks_technical(s)]
        lay = [s for s in syns2 if looks_lay(s) and not looks_technical(s)]

        preferred_lay = overrides.get(term_lc, "")
        if not preferred_lay and lay:
            lay_sorted = sorted(lay, key=lambda s: (len(s.split()), -fuzz.partial_ratio(s.lower(), term_lc)))
            preferred_lay = lay_sorted[0]
        lay_out = ([preferred_lay] if preferred_lay else []) + [x for x in lay if x != preferred_lay]

        rows.append(
            {
                "term_in_corpus": term_lc,
                "freq": freq,
                "qid": ent.qid,
                "label_de": ent.label_de,
                "description_de": ent.description_de,
                "technical_terms": " | ".join(technical),
                "lay_terms": " | ".join(lay_out),
                "aliases_de": " | ".join([x for x in syns2 if x != ent.label_de]),
                "source": "Wikidata",
            }
        )

        if idx % 25 == 0:
            print(f"[INFO] processed {idx}/{len(items)} terms...")

    df = pd.DataFrame(rows).sort_values(["freq", "term_in_corpus"], ascending=[False, True], kind="mergesort")

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False, encoding="utf-8")

    payload = []
    for _, r in df.iterrows():
        payload.append(
            {
                "term": r.get("term_in_corpus", ""),
                "freq": int(r.get("freq", 0) or 0),
                "qid": r.get("qid", ""),
                "label_de": r.get("label_de", ""),
                "description_de": r.get("description_de", ""),
                "lay_terms": [x.strip() for x in str(r.get("lay_terms", "")).split("|") if x.strip()],
                "technical_terms": [x.strip() for x in str(r.get("technical_terms", "")).split("|") if x.strip()],
                "aliases_de": [x.strip() for x in str(r.get("aliases_de", "")).split("|") if x.strip()],
                "source": r.get("source", ""),
            }
        )

    save_json(out_json, payload)
    save_json(cache_path, cache)

    # TODO CSV: missing lay terms
    write_todo_csv(df=df, out_path=str(args.out_todo_csv), todo_n=int(args.todo_n))

    print(f"Wrote: {out_csv} | {out_json} | {cache_path} | {args.out_todo_csv} | {args.out_blacklist_suggestions_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
