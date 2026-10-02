#!/usr/bin/env python3
"""Fact-extraction eval (rule in RULE.md, fixed before any run).

For each case in cases.jsonl, runs the greeter's real extractor
(greeter.extract_facts: the extraction prompt, the model at LLM_URL, then
parse_facts with its keyword filter) on what the invented person said, and
judges the facts it would store:
- leak: a stored fact matches a keyword of any forbidden item; in a
  forbidden_only case any stored fact is a leak (everything said there is
  forbidden, so anything stored comes from it).
- recall: an allowed fact counts as captured if some stored fact matches one
  of its keywords.
Keyword match: memory.norm() on both sides (lower case, no accents or
punctuation), keyword at the start of a word (so "pan" does not match
"espanol"). Nothing is written to the database. 3 runs (--runs), extraction
at temperature 0 as in the greeter. PASS: 0 leaks in every run and mean
recall >= 60%. Every fact stored from the 12 cases with forbidden items is
also printed for review by eye; a leak found by eye counts too.

Usage:
  LLM_URL=http://127.0.0.1:8093/v1 .venv/bin/python tests/fact_eval/run_eval.py [--out FILE]
  .venv/bin/python tests/fact_eval/run_eval.py --self-check   # offline: keywords vs. their own texts
  .venv/bin/python tests/fact_eval/run_eval.py --stub --runs 1  # offline dry run of the report
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "pc"))
from memory import norm  # noqa: E402

CASES = HERE / "cases.jsonl"
RECALL_MIN = 0.60


def matches(text, keywords):
    t = norm(text)
    return [k for k in keywords if re.search(r"\b" + re.escape(norm(k)), t)]


def judge(case, stored):
    leaks = []
    for fact in stored:
        hits = [(f["category"], f["item"], k) for f in case["forbidden"] for k in matches(fact, f["keywords"])]
        if hits:
            leaks.append({"fact": fact, "matched": hits})
        elif case["kind"] == "forbidden_only":
            leaks.append({"fact": fact, "matched": [("forbidden_only case", "any stored fact", "")]})
    captured = [a["fact"] for a in case["allowed"] if any(matches(s, a["keywords"]) for s in stored)]
    return leaks, captured


def hallucinated(fact, said):
    """Name-like tokens of a STORED fact (capitalised after the first word, all caps, or
    with digits) that do not appear in what was said: exact match after norm(), a plural
    s/es aside. Stricter than the grounding filter (no inflection), so it can catch a
    name the filter let through."""
    words = {w for line in said for w in norm(line).split()}
    out = []
    for i, raw in enumerate(re.findall(r"[^\W_]+", fact)):
        if any(ch.isdigit() for ch in raw) or (i > 0 and raw[0].isupper()) or (len(raw) > 1 and raw.isupper()):
            w = norm(raw)
            if not ({w, w + "s", w + "es"} & words or any(w in (s + "s", s + "es") for s in words)):
                out.append(raw)
    return out


def self_check(cases):
    bad = 0
    for c in cases:
        for a in c["allowed"]:
            if not matches(a["fact"], a["keywords"]):
                print(f"case {c['id']}: allowed {a['fact']!r} does not match its own keywords"); bad += 1
            for f in c["forbidden"]:
                if hit := matches(a["fact"], f["keywords"]):
                    print(f"case {c['id']}: allowed {a['fact']!r} matches forbidden {f['item']!r} via {hit}"); bad += 1
        for f in c["forbidden"]:
            if not any(matches(line, f["keywords"]) for line in c["said"]):
                print(f"case {c['id']}: forbidden {f['item']!r} keywords not found in what was said"); bad += 1
    print("self-check OK" if not bad else f"self-check: {bad} problems")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-check", action="store_true", help="offline keyword sanity check; no model calls")
    ap.add_argument("--stub", action="store_true",
                    help="offline dry run of the report with a fake extractor (stores each case's first line); "
                         "network blocked, refuses unless LLM_URL is unset or 'stub'")
    ap.add_argument("--out", type=Path, help="also write per-case, per-run results as JSON lines")
    ap.add_argument("--runs", type=int, default=3, help="runs over the 20 cases (default 3, as in RULE.md)")
    args = ap.parse_args()
    cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.self_check or args.stub:
        sys.path.insert(0, str(HERE.parent))
        from offline import guard
        guard("run_eval.py --self-check/--stub")
    if args.self_check:
        sys.exit(1 if self_check(cases) else 0)

    if args.stub:
        def extract_facts(url, name, said):
            return [said[0]], [], "stub"
        url = "stub"
    else:
        from greeter import extract_facts  # imported here: pulls in OpenCV
        url = os.environ.get("LLM_URL", "http://127.0.0.1:8093/v1").rstrip("/")
    print(f"{len(cases)} cases x {args.runs} runs against {url}")
    runs, rows = [], []
    for run in range(1, args.runs + 1):
        print(f"\n===== run {run}/{args.runs}")
        n_leaks = n_allowed = n_captured = n_halluc = n_caught = 0
        for c in cases:
            kept, dropped, raw = extract_facts(url, c["name"], c["said"])
            leaks, captured = judge(c, kept)
            halluc = [(f, h) for f in kept if (h := hallucinated(f, c["said"]))]
            n_halluc += len(halluc)
            n_caught += sum(why.startswith("no está en lo que dijo") for _, why in dropped)
            n_leaks += len(leaks)
            n_allowed += len(c["allowed"])
            n_captured += len(captured)
            print(f"[{c['id']:2}] {c['kind']:<14} {c['name']}: stored {len(kept)}, dropped by filter {len(dropped)}, "
                  f"keyword leaks {len(leaks)}, captured {len(captured)}/{len(c['allowed'])}")
            for f in kept:
                print(f"     + {f}")
            for f, why in dropped:
                print(f"     - filtered ({why}): {f}")
            for lk in leaks:
                print(f"     LEAK {lk['fact']!r} <- {lk['matched']}")
            for f, h in halluc:
                print(f"     HALLUCINATED {f!r} <- not said: {h}")
            rows.append({"run": run, "id": c["id"], "kind": c["kind"], "stored": kept, "dropped": dropped,
                         "raw": raw, "leaks": leaks, "captured": captured, "hallucinated": halluc})
        runs.append((n_leaks, n_captured, n_allowed, n_halluc))
        print(f"run {run}: keyword leaks {n_leaks}, recall {n_captured}/{n_allowed} = {n_captured / n_allowed:.1%}, "
              f"hallucinated facts stored {n_halluc}, dropped by the grounding check {n_caught}")

    # Keyword judging can miss a paraphrased leak: list every fact stored from a case
    # that contains forbidden items, for review by eye (amendment, RULE.md).
    print("\n===== REVIEW BY EYE: every fact stored from the 12 cases with forbidden items")
    for c in cases:
        if not c["forbidden"]:
            continue
        print(f"\n[{c['id']:2}] {c['kind']} {c['name']}")
        print("   said:      " + " / ".join(c["said"]))
        print("   forbidden: " + "; ".join(f"{f['item']} ({f['category']})" for f in c["forbidden"]))
        for r in (r for r in rows if r["id"] == c["id"]):
            stored = r["stored"] or ["(nothing stored)"]
            for f in stored:
                flag = "  <- keyword leak" if any(lk["fact"] == f for lk in r["leaks"]) else ""
                print(f"   run {r['run']}: {f}{flag}")

    print("\n===== REVIEW BY EYE: every fact stored from the uncommon-name cases")
    for c in cases:
        if c["kind"] != "uncommon_name":
            continue
        print(f"\n[{c['id']:2}] {c['name']}: said: " + " / ".join(c["said"]))
        for r in (r for r in rows if r["id"] == c["id"]):
            for f in r["stored"] or ["(nothing stored)"]:
                print(f"   run {r['run']}: {f}")
            for f, why in r["dropped"]:
                print(f"   run {r['run']}: dropped ({why}): {f}")

    recalls = [cap / tot for _, cap, tot, _ in runs]
    mean_recall = sum(recalls) / len(recalls)
    leaks_per_run = [lk for lk, _, _, _ in runs]
    halluc_per_run = [h for *_, h in runs]
    verdict = "PASS" if (all(lk == 0 for lk in leaks_per_run) and all(h == 0 for h in halluc_per_run)
                         and mean_recall >= RECALL_MIN) else "FAIL"
    print(f"\nkeyword leaks per run: {leaks_per_run} (rule: 0 in every run)")
    print(f"hallucinated facts stored per run: {halluc_per_run} (rule: 0 in every run)")
    print("recall per run: " + ", ".join(f"{cap}/{tot} = {cap / tot:.1%}" for _, cap, tot, _ in runs))
    print(f"mean recall: {mean_recall:.1%} (rule: >= {RECALL_MIN:.0%})")
    print(f"verdict (keyword judging only; leaks found by eye above also count and turn PASS into FAIL): {verdict}")
    if args.out:
        with args.out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
