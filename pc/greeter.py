#!/usr/bin/env python3
"""Greet recognized people in the terminal through a local LLM (Spanish).

Reads the ESP32-CAM MJPEG stream (CAM_IP, default 192.168.20.71) and runs the
same recognition as capture_detect.py --recognize: YuNet + SFace, cosine
>= 0.45, a person confirmed after 2 of the last 3 frames, an unknown face
after 3 of 3.

- A confirmed person not seen for at least --absent-minutes (default 30; never
  seen in this run counts as absent) starts a short conversation. The model,
  at LLM_URL (OpenAI-compatible, default http://127.0.0.1:8093/v1), greets them
  by name. Type a reply and press Enter to continue; an empty line ends it.
- A confirmed unknown face gets one question per run: "Hola, no te conozco.
  ¿Quieres que te recuerde la próxima vez?". The answer is read and not used
  yet (profiles and memory come later).

No conversation text is written to disk. --metrics writes one JSON line per
event or reply with timings and token counts only (no text) to
data/greeter_metrics.jsonl. Stops at --until (default 18:50) or on Ctrl+C.
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_detect import (GREEN, MODEL, RECOGNIZE_AT, ROOT, SFACE, WINDOW, YELLOW,  # noqa: E402
                            Confirmer, identify, load_gallery, show, stream_frames)

import cv2  # noqa: E402

METRICS = ROOT / "data" / "greeter_metrics.jsonl"
UNKNOWN_QUESTION = "Hola, no te conozco. ¿Quieres que te recuerde la próxima vez?"
SYSTEM = ("Eres un asistente doméstico amable que ve la entrada de la casa a través de una cámara. "
          "Hablas en español, de forma cercana, informal y breve: una o dos frases como máximo. "
          "Acabas de ver llegar a {name}. Salúdale por su nombre.")


def metric(enabled, **rec):
    if enabled:
        METRICS.parent.mkdir(parents=True, exist_ok=True)
        with METRICS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now().isoformat(timespec="milliseconds"), **rec}) + "\n")


def chat(url, messages, t_ref):
    """Stream one reply, printing it as it arrives. Returns (text, seconds from
    t_ref to the first word, generated tokens, generation tokens/s)."""
    body = json.dumps({"messages": messages, "stream": True, "max_tokens": 120, "temperature": 0.7,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"{url}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    text, first, n_tok, tps, t_start = "", None, 0, None, time.perf_counter()
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
                    first = time.perf_counter() - t_ref
                n_tok += 1
                text += delta
                print(delta, end="", flush=True)
            if t := chunk.get("timings"):
                n_tok, tps = t.get("predicted_n", n_tok), t.get("predicted_per_second")
    print(flush=True)
    if tps is None:
        tps = n_tok / max(time.perf_counter() - t_start, 1e-9)
    return text, first, n_tok, tps


def conversation(events, args, url):
    """Worker thread: handle one event at a time; reads the user's replies."""
    asked_unknown = False
    while True:
        kind, name, t_confirm = events.get()
        if kind == "stop":
            return
        if kind == "unknown":
            if asked_unknown:
                continue
            asked_unknown = True
            print(f"\nasistente> {UNKNOWN_QUESTION}", flush=True)
            metric(args.metrics, event="unknown_asked", latency_s=round(time.perf_counter() - t_confirm, 3))
            try:
                input("tú> ")
            except EOFError:
                pass
            print("(De momento no guardo perfiles nuevos.)", flush=True)
            continue
        display = name.capitalize()
        print(f"\n[{datetime.now():%H:%M:%S}] {display} confirmado", flush=True)
        messages = [{"role": "system", "content": SYSTEM.format(name=display)},
                    {"role": "user", "content": f"({display} acaba de llegar.)"}]
        try:
            text, first, n_tok, tps = chat(url, messages, t_confirm)
        except Exception as e:
            print(f"\n(LLM no disponible en {url}: {e})", flush=True)
            continue
        print(f"   [primera palabra {first:.2f} s tras la confirmación; {n_tok} tokens a {tps:.1f} tok/s]")
        metric(args.metrics, event="greeting", name=name, latency_first_word_s=round(first or -1, 3),
               tokens=n_tok, tokens_per_s=round(tps, 2))
        messages.append({"role": "assistant", "content": text})
        turn = 0
        while True:
            try:
                reply = input("tú> ").strip()
            except EOFError:
                break
            if not reply:
                print("(fin de la conversación)", flush=True)
                break
            turn += 1
            messages.append({"role": "user", "content": reply})
            t_sent = time.perf_counter()
            try:
                text, first, n_tok, tps = chat(url, messages, t_sent)
            except Exception as e:
                print(f"\n(LLM no disponible: {e})", flush=True)
                break
            print(f"   [primera palabra {first:.2f} s; {n_tok} tokens a {tps:.1f} tok/s]")
            metric(args.metrics, event="reply", name=name, turn=turn, latency_first_word_s=round(first or -1, 3),
                   tokens=n_tok, tokens_per_s=round(tps, 2))
            messages.append({"role": "assistant", "content": text})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--absent-minutes", type=float, default=30,
                    help="greet a person again only after this long unseen (default 30)")
    ap.add_argument("--until", default="18:50", help="stop at this clock time today (default 18:50)")
    ap.add_argument("--preview", action="store_true", help="show the stream with names in a window, 2x")
    ap.add_argument("--metrics", action="store_true",
                    help="append timings/token counts (no text) to data/greeter_metrics.jsonl")
    args = ap.parse_args()

    url = os.environ.get("LLM_URL", "http://127.0.0.1:8093/v1").rstrip("/")
    cam = f"http://{os.environ.get('CAM_IP', '192.168.20.71')}:81/stream"
    stop_at = datetime.strptime(args.until, "%H:%M").replace(
        year=datetime.now().year, month=datetime.now().month, day=datetime.now().day)
    gallery = load_gallery()
    if not gallery:
        sys.exit("no embeddings in data/gallery/ (run pc/enroll.py first)")
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    det = cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320), 0.7, 0.3, 50,
                                    cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU)
    rec = cv2.FaceRecognizerSF.create(str(SFACE), "", cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU)
    print(f"greeter: {cam} -> {url}; known: {', '.join(gallery)}; threshold {RECOGNIZE_AT}; "
          f"greet after {args.absent_minutes:g} min absent; stops at {args.until} or Ctrl+C. "
          f"Reply with text + Enter, empty line ends a conversation.", flush=True)

    events = queue.Queue()
    worker = threading.Thread(target=conversation, args=(events, args, url), daemon=True)
    worker.start()
    confirmer, last_seen, unknown_sent = Confirmer(), {}, False
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
            ids = [identify(rec, gallery, img, f, RECOGNIZE_AT) for f in faces]
            confirmed, unknown_confirmed = confirmer.update(
                {n for n, _, _ in ids if n != "unknown"}, any(n == "unknown" for n, _, _ in ids))
            now = time.monotonic()
            for name in confirmed:
                if name not in last_seen or now - last_seen[name] >= args.absent_minutes * 60:
                    events.put(("person", name, time.perf_counter()))
                last_seen[name] = now
            if unknown_confirmed and not confirmed and not unknown_sent:
                unknown_sent = True
                events.put(("unknown", None, time.perf_counter()))
            if args.preview:
                labels = [(f"{n}{'*' if n in confirmed else ''} {s:.2f}", n != "unknown") for n, s, _ in ids]
                show(img, faces, "confirmed: " + (", ".join(sorted(confirmed)) or "-"),
                     GREEN if confirmed else YELLOW, labels or None)
    except KeyboardInterrupt:
        print("\nstopped (Ctrl+C)", flush=True)
    finally:
        events.put(("stop", None, None))
        if args.preview:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
