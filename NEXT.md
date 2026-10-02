# Next (written 2026-10-02, 18:52, for 2026-10-03)

Everything must stop by 18:55 (PC off ~19:00): `tools/stop_at.sh HH:MM PID PATTERN`.

## State at the end of 2026-10-02
- Camera OK (192.168.20.71, 240x240). `cam.people`: juan (consent recorded
  2026-10-02 15:38), 1 fact ("Esperando el partido de la selección").
  `data/gallery/juan.npy` (20 embeddings). `pc/status.py` shows all of it.
- Fact eval on the 4B: run 2 (`tests/fact_eval/run_20261002_1749.log`) final
  **PASS** after Juan's eye review (0 leaks, 0 hallucinated facts, recall
  77.8%). Since then (not re-evaluated): facts forced to third person
  (prompt + `memory.third_person`).
- Greeter fixes today: Ctrl+C during the save is ignored (3rd force-quits),
  grounding check drops facts with words not said, system prompt says it is a
  camera on a desk with no body.

## 1. First job: fact eval on the 27B (:8092)
Run the same eval (RULE.md as amended twice, 24 cases, 3 runs) against the
27B on :8092, only when es-eval is not using it (es-eval is closed):

```sh
LLM_URL=http://127.0.0.1:8092/v1 .venv/bin/python tests/fact_eval/run_eval.py --out data/fact_eval_27b.jsonl \
  | tee tests/fact_eval/run_$(date +%Y%m%d_%H%M)_27b.log
```

The 27B becomes the greeter default (`LLM_URL`) **only if** it passes RULE.md
**and** Juan's eye review. Note: `NEXT_27B.md` was referenced for this step but
does not exist; the command above is the plan. Compare with the 4B: leaks,
hallucinations, recall, reply latency.

## 2. Live test (c): unknown flow + forget
Not done today. Only the "no answer" path was verified (18:08, nothing
stored). Run with an empty temporary gallery: "sí" → name "prueba" → talk →
"olvídame" → "sí"; check with `pc/status.py --gallery-dir /tmp/empty_gallery`
after each step (README, "Phase 4").

## 3. One live conversation (memory)
Check that stored facts come out in third person (fix made after today's
live tests) and that the greeting uses a stored fact (verified with the 4B at
18:07, 1 fact in the prompt).

## 4. Known 4B weaknesses to compare against the 27B
- Slang: "chelas" (beers) not understood.
- Invented details in replies (e.g. "veo y escucho").
- No inference of durable traits ("esperando el partido de la selección" →
  never "es hincha de la Selección").
