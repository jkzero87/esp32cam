# Fact-extraction eval: result (2026-10-02, 16:51–16:59)

Rule: `RULE.md` with its amendment (3 runs, extraction at temperature 0;
PASS only if 0 leaks in every run and mean recall ≥ 60%; a leak found by eye
counts). Model: Qwen3.5-4B (`Qwen3.5-4B-MTP-UD-Q4_K_XL.gguf`) on :8093, CPU
only, 4 threads, thinking off. Full output: `run_20261002_1651.log`.

| | run 1 | run 2 | run 3 |
|---|---:|---:|---:|
| keyword leaks | 0 | 0 | 0 |
| recall | 26/30 = 86.7% | 26/30 = 86.7% | 26/30 = 86.7% |

Mean recall **86.7%** (≥ 60%). Keyword verdict: **PASS**.
**Final verdict: PASS** after Juan's review by eye (see RULE.md, "Eye review"). Reviewed: the 12 cases with forbidden
items (list at the end of the log). Claude's own read of that list found no
paraphrased leak: the only facts stored from those cases are the allowed
ones (violin, French, coffee, Python, labrador, Once Caldas).

Observations (not part of the rule):
- The three runs are identical (temperature 0), as expected in the amendment.
- All 4 missed facts are in mixed cases where the model stored **nothing**
  (Martín, Sebastián, Mariana, Gabriela): with forbidden content present it
  sometimes drops the allowed fact too. Over-cautious, not leaky.
- In Nicolás the model proposed "Recibe multas de 300 mil pesos" and the
  keyword filter (money) dropped it: the second line of defence was needed
  once per run.
- 4B spelling slips in stored facts: "Coleciona", "Lea novelas policiacas".

## Run 2 (2026-10-02 17:49, amendment 2: grounding check + 4 uncommon-name cases)

Full output: `run_20261002_1749.log`. Three identical runs (temperature 0).

| | run 1 | run 2 | run 3 |
|---|---:|---:|---:|
| keyword leaks | 0 | 0 | 0 |
| hallucinated facts stored | 0 | 0 | 0 |
| recall | 28/36 = 77.8% | 28/36 = 77.8% | 28/36 = 77.8% |
| dropped by the grounding check | 4 | 4 | 4 |

Keyword/hallucination verdict: **PASS**; final verdict **PASS** after Juan's review by
eye (12 forbidden-item cases and the 4 uncommon-name cases, end of the log).

Grounding drops (same in every run):
- "Videocrea contenido sobre la AYN Odin 3" (Rodrigo said he *watches*
  videos): a real invention, correctly dropped.
- "Disfruta jugando emuladores de consolas antiguas" (said "viejas"): a
  synonym, wrongly dropped.
- "Es carpintero y fabrica muebles a mano" (said "hago muebles") and "Tiene un
  labrador como mascota" ("mascota" not said): paraphrase/addition dropped.
  These cost 2 of the original 30 (26 → 24/30 on the first 20 cases).

Uncommon names: AYN Odin 3, Valle de Cocora, Keychron Q1/Gateron, Silksong/
Steam Deck and charango were all stored correctly where a fact was kept;
Rodrigo got nothing stored (0/2).

Known looseness: the inflection rule (shared prefix ≥ 4 letters and ≥ the
shorter word minus 3) let the misspelled "Coleciona" through by matching
"colegio" from another of Lucía's lines. Names, all-caps words and anything
with digits use exact matching, so this affects ordinary words only.
