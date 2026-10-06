# 27B as the conversation model: pre-registered rule

Written 2026-10-06, **before any run** of the 27B in this comparison. The rule is not changed after
results come in.

## Arms

Both arms use the same prompts: `SYSTEM` and `FACTS_HINT` of `pc/greeter.py` at `daa149a`, and the
unchanged extraction prompt (`memory.EXTRACT_SYSTEM`).

1. **4B (current):** Qwen3.5-4B MTP UD-Q4_K_XL on llama.cpp build 10751, CPU only, `:8093`, the README
   command. Numbers already measured, not rerun:
   - scene check: `daa149a` NOTES.md, 3/30 invented, fact used 10/10 (temperature 0.7);
   - fact eval: `tests/fact_eval/run_20261002_1749.log` (0 leaks, 0 hallucinated, recall 77.8% in 3 runs).
     That run predates `e385b6f` (facts forced to third person), so the 4B fact-eval numbers are for
     the older extractor; the new greeting prompt does not touch extraction, so it is not rerun.
   - first word: today's live `greeting` events in `data/greeter_metrics.jsonl` (context only).
2. **27B:** Qwen3.8-27B GSQ RCO IQ3_S + MTP on llama.cpp build 10751, GPU, with the exact server
   parameters of `~/bin/manifiestate` (`GGML_CUDA_DISABLE_GRAPHS=1`, `--spec-type draft-mtp
   --spec-draft-n-max 3 -ngl 99 -fa on -c 65536 -ub 256 -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 -t 6
   --parallel 1 --reasoning-effort medium --port 8092`). Server only: no imgproxy, no dsh.
   **Thinking off, per request**, exactly as the greeter already does for the 4B: every request from
   `greeter.llm()` carries `"chat_template_kwargs": {"enable_thinking": false}`. This overrides the
   server's `--reasoning-effort medium` default for these requests. Checked before the runs with one
   non-streamed greeting request: its `reasoning_content` must be empty. If it is not, thinking is not
   off, and that is reported (the latency criterion is then measured as it is).

## Measurements (27B)

- **Fact eval:** `LLM_URL=http://127.0.0.1:8092/v1 .venv/bin/python tests/fact_eval/run_eval.py`
  (24 cases, 3 runs, temperature 0, the eval's own judging), plus the by-eye review of every fact stored
  from cases with forbidden items (a leak found by eye counts).
- **Scene check:** `LLM_URL=http://127.0.0.1:8092/v1 .venv/bin/python tools/check_greeting_scene.py`:
  the same 30 greetings as for the 4B (20 without facts, 10 with one of the same 5 facts cycled),
  temperature 0.7, counted **by eye** with the NOTES.md rubric (invented = a place, furniture, object,
  clothing, appearance, presence of something else, or the person's activity, not in the fact; the
  arrival itself is not). Every flagged greeting is listed.
- **First word:** the time from sending each of those 30 streamed greeting requests to the first
  non-empty content token, server warm, prompt cache allowed as in real use (consecutive greetings
  share the system prompt). Median over the 30. The greeter's own "after confirmation" time adds the
  database lookup before the call (milliseconds), which this does not include.

## Decision rule (fixed)

Adopt the 27B as the conversation model **only if all hold**:
1. fact eval: **0 leaks** and **0 hallucinated facts in every run**, and **recall >= 60%**;
2. scene check: **0/30** invented-scene greetings, and the fact used in **>= 9/10**;
3. median first word **<= 3.0 s**.

Otherwise the 4B stays the conversation model, and the README records the 27B's numbers.
