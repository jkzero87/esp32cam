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
**Final verdict pending Juan's review by eye** of the 12 cases with forbidden
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
