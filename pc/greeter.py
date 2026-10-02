#!/usr/bin/env python3
"""Greet recognized people in the terminal through a local LLM (Spanish), with memory.

Reads the ESP32-CAM MJPEG stream (CAM_IP, default 192.168.20.71) and runs the
same recognition as capture_detect.py --recognize: YuNet + SFace, cosine
>= 0.45, a person confirmed after 2 of the last 3 frames, an unknown face
after 3 of 3. The model is reached at LLM_URL (OpenAI-compatible, default
http://127.0.0.1:8093/v1).

- Known person, not seen for --absent-minutes (default 30; never seen in this
  run counts as absent; leaving counts only after 5 s unconfirmed, so with
  --absent-minutes 0 stepping out of view for 5 s and back greets again):
  the model greets them by name. Their last 5 facts
  from cam.facts go into the system prompt so the greeting can mention one.
  Replies are typed in the terminal; an empty line (or 3 minutes of silence)
  ends the conversation. Afterwards the model extracts up to 3 short durable
  facts from what the person said; facts about health, money, passwords/IDs
  or third parties are dropped. Only those facts are stored, never the
  transcript.
- "olvídame" / "bórrame" in a conversation, after a yes/no confirmation,
  deletes the person's row, facts and gallery file.
- Unknown face: asks "Hola, no te conozco. ¿Quieres que te recuerde la próxima
  vez?". Only on a clear yes it asks the name, records consent_at and builds
  the gallery from the embeddings of the unknown face seen during this exchange
  (keep if cosine < 0.9 to all kept, up to 20). On no, or no answer within
  30 s, it stores nothing and does not ask again for 10 minutes.

--gallery-dir chooses the gallery (default data/gallery). --metrics appends
timings and counts (no text) to data/greeter_metrics.jsonl. Stops at --until
(default 18:50) or on Ctrl+C: the first ends the open conversation and saves its
facts; while saving, a second is ignored with a notice and a third force-quits
(that conversation's memory is lost).
"""
import argparse
import json
import os
import queue
import select
import signal
import sys
import termios
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_detect import (GALLERY, GREEN, MODEL, RECOGNIZE_AT, ROOT, SFACE, WINDOW, YELLOW,  # noqa: E402
                            Confirmer, identify, show, stream_frames)
from memory import (Memory, extraction_messages, is_clear_yes, name_slug, parse_facts,  # noqa: E402
                    wants_forget)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

METRICS = ROOT / "data" / "greeter_metrics.jsonl"
UNKNOWN_QUESTION = "Hola, no te conozco. ¿Quieres que te recuerde la próxima vez?"
UNKNOWN_COOLDOWN_S = 600
ANSWER_TIMEOUT_S = 30
SILENCE_END_S = 180
LEFT_AFTER_S = 5  # unconfirmed this long = left, so detection flicker never re-greets
DIVERSITY, CAP = 0.9, 20
SYSTEM = ("Eres un asistente amable: una cámara sobre un escritorio, con una voz que habla por el terminal. "
          "No tienes cuerpo: no puedes cocinar, traer, comprar ni hacer nada físico. Si te piden algo así, "
          "dilo con amabilidad y sigue conversando; nunca prometas ni ofrezcas acciones que no puedes hacer "
          "(tampoco tienes internet ni noticias). No inventes lo que ves: solo sabes quién ha llegado. "
          "Hablas en español, de forma cercana, informal y breve: una o dos frases como máximo. "
          "Acabas de ver llegar a {name}. Salúdale por su nombre.")
FACTS_HINT = ("\nCosas que recuerdas de {name} de conversaciones anteriores (puedes mencionar una, "
              "con naturalidad, si encaja; no las enumeres):\n{facts}")


def metric(enabled, **rec):
    if enabled:
        METRICS.parent.mkdir(parents=True, exist_ok=True)
        with METRICS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now().isoformat(timespec="milliseconds"), **rec}) + "\n")


STOPPING = threading.Event()  # set on Ctrl+C / --until: open conversations end and save their facts
SAVING_MSG = "(guardando… espera unos segundos)"


class CtrlC:
    """SIGINT handler. 1st Ctrl+C: KeyboardInterrupt, so the main loop stops and the
    open conversation ends and saves its facts. While saving (self.saving), the 2nd
    is ignored with a notice; the 3rd force-quits and the memory is lost."""

    def __init__(self, exit_fn=os._exit):
        self.saving = False
        self.ignored = 0
        self.exit_fn = exit_fn

    def __call__(self, signum=None, frame=None):
        if not self.saving:
            raise KeyboardInterrupt
        self.ignored += 1
        if self.ignored == 1:
            print(f"\n{SAVING_MSG} Otro Ctrl+C fuerza la salida y se pierde la memoria de esta conversación.",
                  flush=True)
            return
        print("\n(salida forzada: la memoria de esta conversación se ha perdido)", flush=True)
        self.exit_fn(130)


def read_line(prompt, timeout, fresh=False):
    """input() with a timeout; None on timeout, end of input or shutdown.
    FRESH discards anything typed before the prompt (a stale Enter must not
    answer a yes/no question)."""
    if fresh and sys.stdin.isatty():
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    print(prompt, end="", flush=True)
    deadline = time.monotonic() + timeout
    while not STOPPING.is_set():
        left = deadline - time.monotonic()
        if left <= 0:
            break
        ready, _, _ = select.select([sys.stdin], [], [], min(left, 0.5))
        if ready:
            line = sys.stdin.readline()
            return None if line == "" else line.rstrip("\n")
    print(flush=True)
    return None


def load_gallery(gallery_dir):
    """{name: L2-normalised embeddings [k, 128]} from GALLERY_DIR/*.npy."""
    out = {}
    for path in sorted(Path(gallery_dir).glob("*.npy")):
        emb = np.load(path).astype(np.float32).reshape(-1, 128)
        out[path.stem] = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return out


def select_diverse(feats, diversity=DIVERSITY, cap=CAP):
    """Same rule as enroll.py: keep a feature if its cosine to every kept one is < DIVERSITY."""
    kept, units = [], []
    for f in feats:
        u = f / np.linalg.norm(f)
        if len(kept) < cap and all(float(u @ k) < diversity for k in units):
            kept.append(f)
            units.append(u)
    return kept


def llm(url, messages, stream_to_terminal=True, max_tokens=120, temperature=0.7):
    """One chat completion. Returns (text, first-word time from call, tokens, tokens/s)."""
    body = json.dumps({"messages": messages, "stream": True, "max_tokens": max_tokens, "temperature": temperature,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"{url}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    text, first, n_tok, tps = "", None, 0, None
    if stream_to_terminal:
        print("asistente> ", end="", flush=True)
    with urllib.request.urlopen(req, timeout=120) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
            if delta:
                if first is None and delta.strip():
                    first = time.perf_counter() - t0
                n_tok += 1
                text += delta
                if stream_to_terminal:
                    print(delta, end="", flush=True)
            if t := chunk.get("timings"):
                n_tok, tps = t.get("predicted_n", n_tok), t.get("predicted_per_second")
    if stream_to_terminal:
        print(flush=True)
    return text, first, n_tok, tps or n_tok / max(time.perf_counter() - t0, 1e-9)


def extract_facts(url, name, said):
    """What NAME said -> (kept facts, dropped [(fact, reason)], raw model output).
    Temperature 0: extraction is not conversation (the chat keeps 0.7)."""
    raw, _, _, _ = llm(url, extraction_messages(name, said), stream_to_terminal=False, max_tokens=200,
                       temperature=0)
    return (*parse_facts(raw), raw)


class State:
    """Shared between the camera loop (main thread) and the conversation thread."""

    def __init__(self, gallery_dir):
        self.lock = threading.Lock()
        self.gallery_dir = Path(gallery_dir)
        self.gallery = load_gallery(gallery_dir)
        self.last_seen = {}
        self.collector = None          # list of raw features of the unknown face during an exchange
        self.unknown_busy = False
        self.unknown_cooldown_until = 0.0


def person_conversation(st, mem, args, url, name, t_confirm):
    display = name.capitalize()
    started = datetime.now(timezone.utc)
    mem.person_id(name)
    facts = mem.recent_facts(name)
    system = SYSTEM.format(name=display)
    if facts:
        system += FACTS_HINT.format(name=display, facts="\n".join(f"- {f}" for f in facts))
    print(f"\n[{datetime.now():%H:%M:%S}] {display} confirmado; hechos en el prompt: {len(facts)}"
          + "".join(f"\n   · {f}" for f in facts), flush=True)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"({display} acaba de llegar.)"}]
    t_call = time.perf_counter()
    text, first, n_tok, tps = llm(url, messages)
    first_total = (t_call - t_confirm) + (first or 0)  # confirmation -> first word, incl. the DB lookup
    print(f"   [primera palabra {first_total:.2f} s tras la confirmación; {n_tok} tokens a {tps:.1f} tok/s]",
          flush=True)
    metric(args.metrics, event="greeting", name=name, facts_in_prompt=len(facts),
           latency_first_word_s=round(first_total or -1, 3), tokens=n_tok, tokens_per_s=round(tps, 2))
    messages.append({"role": "assistant", "content": text})

    said, end = [], "empty_line"
    while True:
        reply = read_line("tú> ", SILENCE_END_S)
        if reply is None or not reply.strip():
            end = "shutdown" if STOPPING.is_set() else "silence" if reply is None else "empty_line"
            print(f"(fin de la conversación: {end})", flush=True)
            break
        if wants_forget(reply):
            answer = read_line("asistente> ¿Seguro que quieres que te olvide? Borraré tu perfil, lo que recuerdo "
                               "de ti y tu cara. (sí/no)\ntú> ", 60, fresh=True)
            if answer is not None and is_clear_yes(answer):
                removed = mem.forget(name)
                with st.lock:
                    st.gallery.pop(name, None)
                    st.last_seen.pop(name, None)
                print(f"asistente> Hecho, {display}. Ya no te recuerdo.\n"
                      f"   [borrado: {removed['people']} fila en cam.people, {removed['facts']} hechos, "
                      f"galería {'borrada' if removed['gallery_file'] else 'no existía'}]", flush=True)
                metric(args.metrics, event="forget", name=name, **removed)
                return
            print("asistente> Vale, no borro nada.", flush=True)
            continue
        said.append(reply)
        messages.append({"role": "user", "content": reply})
        text, first, n_tok, tps = llm(url, messages)
        print(f"   [primera palabra {first:.2f} s; {n_tok} tokens a {tps:.1f} tok/s]", flush=True)
        metric(args.metrics, event="reply", name=name, latency_first_word_s=round(first or -1, 3),
               tokens=n_tok, tokens_per_s=round(tps, 2))
        messages.append({"role": "assistant", "content": text})

    if not said:
        metric(args.metrics, event="facts", name=name, stored=0, dropped=0, end=end, user_lines=0)
        return
    print("   [extrayendo hechos…]", flush=True)
    kept, dropped, _ = extract_facts(url, display, said)
    n = mem.add_facts(name, kept, started)
    print(f"   [memoria: {n} hecho(s) guardado(s)" + "".join(f"\n    + {f}" for f in kept)
          + "".join(f"\n    - descartado ({why}): {f}" for f, why in dropped) + "]", flush=True)
    metric(args.metrics, event="facts", name=name, stored=n, dropped=len(dropped), end=end, user_lines=len(said))


def unknown_exchange(st, mem, args):
    with st.lock:
        st.collector = []
    t0 = time.monotonic()
    try:
        print(f"\nasistente> {UNKNOWN_QUESTION}", flush=True)
        answer = read_line("tú> ", ANSWER_TIMEOUT_S, fresh=True)
        if answer is None or not is_clear_yes(answer):
            print("asistente> Vale, no guardo nada." if answer is not None else "(sin respuesta: no guardo nada)",
                  flush=True)
            metric(args.metrics, event="unknown_declined", answered=answer is not None,
                   answer_empty=answer is not None and not answer.strip())
            return False
        raw_name = read_line("asistente> ¡Genial! ¿Cómo te llamas?\ntú> ", 60, fresh=True)
        name = name_slug(raw_name or "")
        if not name:
            print("asistente> No entendí el nombre; no guardo nada.", flush=True)
            return False
        if (st.gallery_dir / f"{name}.npy").exists():
            print(f"asistente> Ya conozco a alguien llamado {name}; no guardo nada.", flush=True)
            return False
        with st.lock:
            feats = list(st.collector)
        kept = select_diverse(feats)
        if not kept:
            print("asistente> No te vi bien la cara; no guardo nada.", flush=True)
            return False
        st.gallery_dir.mkdir(parents=True, exist_ok=True)
        np.save(st.gallery_dir / f"{name}.npy", np.vstack(kept).astype(np.float32))
        pid = mem.person_id(name, consent=True)
        with st.lock:
            st.gallery = load_gallery(st.gallery_dir)
            st.last_seen[name] = time.monotonic()
        print(f"asistente> Encantado, {name.capitalize()}. La próxima vez te reconoceré.\n"
              f"   [guardado: cam.people id={pid} con consent_at; {len(kept)} embeddings de {len(feats)} "
              f"caras vistas en {time.monotonic() - t0:.0f} s -> {st.gallery_dir / (name + '.npy')}]", flush=True)
        metric(args.metrics, event="enrolled", embeddings=len(kept), faces_seen=len(feats))
        return True
    finally:
        with st.lock:
            st.collector = None


def conversation_worker(events, st, mem, args, url):
    while True:
        kind, name, t_confirm = events.get()
        if kind == "stop" or STOPPING.is_set():
            return
        try:
            if kind == "unknown":
                ok = unknown_exchange(st, mem, args)
                with st.lock:
                    st.unknown_busy = False
                    if not ok:
                        st.unknown_cooldown_until = time.monotonic() + UNKNOWN_COOLDOWN_S
            else:
                person_conversation(st, mem, args, url, name, t_confirm)
        except Exception as e:
            print(f"\n(error: {e})", flush=True)
            with st.lock:
                st.unknown_busy = False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--absent-minutes", type=float, default=30,
                    help="greet a person again only after this long unseen (default 30)")
    ap.add_argument("--gallery-dir", type=Path, default=GALLERY, help="gallery directory (default data/gallery)")
    ap.add_argument("--until", default="18:50", help="stop at this clock time today (default 18:50)")
    ap.add_argument("--preview", action="store_true", help="show the stream with names in a window, 2x")
    ap.add_argument("--metrics", action="store_true",
                    help="append timings/counts (no text) to data/greeter_metrics.jsonl")
    args = ap.parse_args()

    url = os.environ.get("LLM_URL", "http://127.0.0.1:8093/v1").rstrip("/")
    cam = f"http://{os.environ.get('CAM_IP', '192.168.20.71')}:81/stream"
    now = datetime.now()
    stop_at = datetime.strptime(args.until, "%H:%M").replace(year=now.year, month=now.month, day=now.day)
    st = State(args.gallery_dir)
    mem = Memory(args.gallery_dir)
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    det = cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320), 0.7, 0.3, 50,
                                    cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU)
    rec = cv2.FaceRecognizerSF.create(str(SFACE), "", cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU)
    print(f"greeter: {cam} -> {url}; gallery {args.gallery_dir} ({', '.join(st.gallery) or 'empty'}); "
          f"threshold {RECOGNIZE_AT}; greet after {args.absent_minutes:g} min absent; stops at {args.until} "
          f"or Ctrl+C. Reply + Enter; empty line ends a conversation.", flush=True)

    ctrl_c = CtrlC()
    signal.signal(signal.SIGINT, ctrl_c)
    events = queue.Queue()
    worker = threading.Thread(target=conversation_worker, args=(events, st, mem, args, url), daemon=True)
    worker.start()
    confirmer = Confirmer()
    if args.preview:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        for img, _ in stream_frames(cam, timeout=10):
            if datetime.now() >= stop_at:
                print(f"\n{args.until} reached: stopping", flush=True)
                break
            h, w = img.shape[:2]
            det.setInputSize((w, h))
            _, faces = det.detect(img)
            faces = [] if faces is None else faces
            with st.lock:
                gallery = st.gallery
            ids = [identify(rec, gallery, img, f, RECOGNIZE_AT) for f in faces] if gallery else \
                [("unknown", 0.0, None) for _ in faces]
            confirmed, unknown_confirmed = confirmer.update(
                {n for n, _, _ in ids if n != "unknown"}, any(n == "unknown" for n, _, _ in ids))
            mono = time.monotonic()
            with st.lock:
                if st.collector is not None and len(faces) == 1 and ids[0][0] == "unknown":
                    feat = ids[0][2] if ids[0][2] is not None else \
                        rec.feature(rec.alignCrop(img, faces[0])).ravel().astype(np.float32)
                    st.collector.append(feat)
                for name in confirmed:
                    gap = mono - st.last_seen[name] if name in st.last_seen else None
                    if gap is None or gap >= max(args.absent_minutes * 60, LEFT_AFTER_S):
                        events.put(("person", name, time.perf_counter()))
                    st.last_seen[name] = mono
                if (unknown_confirmed and not confirmed and not st.unknown_busy
                        and mono >= st.unknown_cooldown_until):
                    st.unknown_busy = True
                    events.put(("unknown", None, time.perf_counter()))
            if args.preview:
                labels = [(f"{n}{'*' if n in confirmed else ''} {s:.2f}", n != "unknown") for n, s, _ in ids]
                show(img, faces, "confirmed: " + (", ".join(sorted(confirmed)) or "-"),
                     GREEN if confirmed else YELLOW, labels or None)
    except KeyboardInterrupt:
        print("\nstopped (Ctrl+C)", flush=True)
    finally:
        ctrl_c.saving = True  # from here on Ctrl+C no longer interrupts the save (see CtrlC)
        STOPPING.set()  # an open conversation ends now and still saves its facts
        events.put(("stop", None, None))
        if args.preview:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        if worker.is_alive():
            print(f"(cerrando) {SAVING_MSG}", flush=True)
            worker.join(timeout=90)
    return 0


if __name__ == "__main__":
    sys.exit(main())
