# Where things stand (2026-10-01, PC turned off at ~17:50)

Everything must stop by 18:55 (PC off ~19:00): `tools/stop_at.sh HH:MM PID PATTERN`.

## Working
- ESP32-CAM (AI-Thinker, GC2145, RGB565 240x240) serving `/capture` and the
  MJPEG stream (~4.5 fps).
- PC side: YuNet detection, SFace recognition (threshold 0.45 from the LFW
  impostor test, confirmation 2 of 3 frames), `enroll.py`, `--add-embeddings`.
- Greeter with the local Qwen3.5-4B (CPU only on :8093, ~9 tok/s): greets by
  name in Spanish; first word ~2.5 s cold.
- Phase 4 code: Postgres schema `cam` (role `camuser`, `.env` gitignored),
  fact extraction with filters, facts in the greeting prompt, consented
  enrollment of unknown faces, "olvídame". Unit tests pass
  (`tests/test_memory.py`).

## Not yet verified live
The first live run of (b) memory and (c) unknown flow stored nothing: facts
were never extracted (the conversation was ended by Ctrl+C before the
extraction ran), and the consent answer was not taken as a yes (a stale
empty line from the terminal is the likely cause). Both are fixed (Ctrl+C now
finishes the open conversation and saves its facts; pending input is
discarded before each question) but not retested live.

State left: `cam.people` = juan (consent_at NULL, enrolled by hand), 0 facts;
`data/gallery/juan.npy` (20 embeddings); no "prueba" anywhere.

## Next
1. **Rerun live tests (b) and (c)** (commands in README, "Phase 4"). Start
   `llama-server` on :8093 first, with `tools/stop_at.sh 18:55`.
2. **Compare the 4B and the 27B as the conversation model** (the 27B on :8092
   only when es-eval is not using it): greeting latency, reply speed, quality
   of greetings, and whether stored facts are mentioned naturally.
3. Then: impostor test with other people on another day; decide whether
   `--add-embeddings` is ever turned on.
