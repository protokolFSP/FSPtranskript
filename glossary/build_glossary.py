"""
Build a German medical glossary from transcript files in this repo.

Input:
- --transcripts_dir: folder containing transcript files (txt/md/json)
- optional manual overrides CSV

Output:
- public/glossary.csv
- public/glossary.json
- public/wikidata_cache.json  (speeds up future runs)

Wikidata:
- uses labels/aliases/descriptions in German.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd
import requests
from rapidfuzz import fuzz

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

GERMAN_STOPWORDS: Set[str] = {
    "und","oder","aber","auch","dann","denn","dass","das","der","die","ein","eine","einen","einem","einer",
    "ich","wir","sie","ihr","ihnen","du","er","es","ist","sind","war","waren","haben","hat","hatte","hatten",
    "nicht","kein","keine","mit","auf","für","im","in","am","an","zu","vom","von","als","bei","so","wie","was",
    "wo","wann","warum","wieso","bitte","danke","okay","genau","ja","nein","mal","noch","schon","jetzt",
}

MED_SUFFIXES = (
    "itis","ose","ämie","aemie","pathie","kardie","tomie","omie","ektomie","skopie","embolie","thrombose",
    "infarkt","syndrom","stenose","insuffizienz","hypertonie","hypotonie",
)

MED_SUBSTRINGS = (
    "pankreat","cholezyst","append","koronar","arter","vene","myokard","vorhoffl",
    "tachy","brady","dyspno","synkop","angina","tropon","bilirub","creatin",
)

LAY_HINT_SUBSTRINGS = (
    "schmerz","entzünd","atem","luft","herz","bauch","blinddarm","galle","niere","leber","lunge",
    "übel","brechen","fieber","schwindel","blutdruck","zucker","wasser",
)

ABBREV_OK = {
    "ekg","rr","spo2","ct","mrt","crp","hba1c","inr","ptt","aptt","bga","ck","ckmb",
    "ldh","ast","alt","ggt","ntpro","nt-pro","ntprobnp","nt-probnp",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--cache_json", required=True)
    ap.add_argument("--manual_overrides", default="")
    ap.add_argument("--max_terms", type=int, default=300)
    ap.add_argument("--min_count", type=int, default=2)
    ap.add_argument("--sleep_s", type=float, default=0.0)
    ap.add_argument("--max_file_mb", type=float, default=5.0)
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
    t = re.sub(r"(?s)\A---.*?---\s*", "", t)  # yaml frontmatter
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
    if t[0].isupper() and len(t) >= 6:
        return True
    if any(ch.isdigit() for ch in t) and re.fullmatch(r"\d+[a-zA-Z%]+", t):
        return True
    return False


def is_candidate_ngram(tokens: List[str]) -> bool:
    if not tokens:
        return False
    if all(t.lower() in GERMAN_STOPWORDS for t in tokens):
        return False

    joined = " ".join(tokens).strip()
    if len(joined) < 3 or len(joined) > 60:
        return False

    if not any(is_medicalish_word(t) for t in tokens):
        return False

    if all(re.fullmatch(r"\d+", t) for t in tokens):
        return False

    return True


def extract_candidates(texts: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for txt in texts:
        toks = word_tokens(txt)

        for w in toks:
            if is_candidate_ngram([w]):
                k = w.lower()
                counts[k] = counts.get(k, 0) + 1

        for n in (2, 3):
            for i in range(0, max(0, len(toks) - n + 1)):
                ng = toks[i : i + n]
                if is_candidate_ngram(ng):
                    k = " ".join(x.lower() for x in ng)
                    counts[k] = counts.get(k, 0) + 1

    return counts


@dataclass(frozen=True)
class WikidataEntity:
    qid: str
    label_de: str
    description_de: str
    aliases_de: List[str]


def wikidata_search(session: requests.Session, term: str) -> Optional[str]:
    params = {"action": "wbsearchentities", "search": term, "language": "de", "limit": 5, "format": "json"}
    r = session.get(WIKIDATA_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = data.get("search") or []
    if not results:
        return None
    return str(results[0].get("id") or "") or None


def wikidata_get_entity(session: requests.Session, qid: str) -> Optional[WikidataEntity]:
    params = {
        "action": "wbgetentities",
        "ids": qid,
        "props": "labels|aliases|descriptions",
        "languages": "de",
        "format": "json",
    }
    r = session.get(WIKIDATA_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    ent = (data.get("entities") or {}).get(qid)
    if not isinstance(ent, dict):
        return None

    label = ((ent.get("labels") or {}).get("de") or {}).get("value") or ""
    desc = ((ent.get("descriptions") or {}).get("de") or {}).get("value") or ""
    aliases = [a.get("value") for a in ((ent.get("aliases") or {}).get("de") or []) if isinstance(a, dict) and a.get("value")]
    aliases = [str(a) for a in aliases]
    return WikidataEntity(qid=qid, label_de=str(label), description_de=str(desc), aliases_de=aliases)


def best_qid_for_term(session: requests.Session, term: str, cache: Dict[str, Any], sleep_s: float) -> Optional[str]:
    key = f"search:{term.lower()}"
    if key in cache:
        return cache[key] or None
    qid = wikidata_search(session, term)
    cache[key] = qid or ""
    if sleep_s > 0:
        time.sleep(sleep_s)
    return qid


def get_entity_cached(session: requests.Session, qid: str, cache: Dict[str, Any], sleep_s: float) -> Optional[WikidataEntity]:
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

    ent = wikidata_get_entity(session, qid)
    cache[key] = (
        {"label_de": ent.label_de, "description_de": ent.description_de, "aliases_de": ent.aliases_de} if ent else ""
    )
    if sleep_s > 0:
        time.sleep(sleep_s)
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


def main() -> int:
    args = parse_args()

    transcripts_dir = Path(args.transcripts_dir)
    if not transcripts_dir.exists():
        raise SystemExit(f"Missing transcripts dir: {transcripts_dir}")

    texts = list(iter_transcript_texts(transcripts_dir, max_file_mb=float(args.max_file_mb)))
    if not texts:
        raise SystemExit(f"No transcripts found under: {transcripts_dir}")

    cand = extract_candidates(texts)
    items = [(t, c) for t, c in cand.items() if c >= int(args.min_count)]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[: int(args.max_terms)]

    cache_path = Path(args.cache_json)
    cache = load_json(cache_path)
    if not isinstance(cache, dict):
        cache = {}

    overrides = load_manual_overrides(args.manual_overrides)
    session = requests.Session()

    rows: List[Dict[str, Any]] = []
    for term_lc, freq in items:
        qid = best_qid_for_term(session, term_lc, cache=cache, sleep_s=float(args.sleep_s))
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

        ent = get_entity_cached(session, qid, cache=cache, sleep_s=float(args.sleep_s))
        if not ent:
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

    print(f"Wrote: {out_csv} | {out_json} | {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
