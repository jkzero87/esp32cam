#!/usr/bin/env python3
"""Run YuNet face detection on the PC (CPU) on frames from the ESP32-CAM.

Two frame sources (camera address from cam_ip(): CAM_IP in the environment or .env):
  --source capture (default): every --interval seconds (default 1; 0 = as fast
      as possible) fetch one JPEG from http://<CAM_IP>/capture.
  --source stream: read the MJPEG stream at http://<CAM_IP>:81/stream and
      process every frame as it arrives (--interval is ignored).

Prints one line per frame: timestamp, t (seconds since the measured part
started; negative during warm-up), frame wait time ("fetch": the /capture
request, or the time spent receiving the next stream part), detection time,
number of faces, best score (0 if none), then score and box (x, y, w, h) of
each face. Stops after --seconds and prints a summary: frames per second,
detection rate, mean score, mean fetch and detection times.

--save writes frames that contain at least one face, with the boxes drawn, to
data/frames/. Frames without a face are never written, and nothing is written
to disk without --save.

--recognize compares every detected face (aligned and embedded with SFace)
with every embedding of every person in data/gallery/*.npy (from enroll.py):
the face gets the name with the highest cosine similarity, or "unknown" if
that is below 0.45 (OpenCV documents 0.363; 0.45 is chosen from the LFW
impostor trade-off in the README). A person is "confirmed" when they match
in at least 2 of the last 3 frames; an unknown face when it is seen in 3 of 3.
--add-embeddings (off by default): when a person is confirmed and a face
matches them at >= 0.6 from a new view (cosine < 0.9 to every kept
embedding), append it to data/gallery/NAME.npy, up to 20.

--preview shows each frame in a window, scaled up 2x, with the face boxes,
scores, t and the running count of frames with a face. Press q to stop early.
--warmup N first runs N seconds that are shown (labelled WARM-UP) but not
measured, printed as "(warm-up)", or saved; the measured --seconds follow
immediately in the same window.
"""
import argparse
import os
import re
import statistics
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# The opencv-python wheel ships only Qt's X11 (xcb) plugin; on Wayland use XWayland.
if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "face_detection_yunet_2026may.onnx"
FRAMES = ROOT / "data" / "frames"
SFACE = ROOT / "models" / "face_recognition_sface_2021dec.onnx"
GALLERY = ROOT / "data" / "gallery"


def cam_ip():
    """The camera's LAN address: CAM_IP from the environment, else CAM_IP in
    .env (gitignored, the one place it is configured). Exits if neither is set."""
    ip = os.environ.get("CAM_IP")
    env = ROOT / ".env"
    if not ip and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("CAM_IP="):
                ip = line.split("=", 1)[1].strip()
    if not ip:
        sys.exit("CAM_IP is not set: add CAM_IP=<camera IP> to .env or export it")
    return ip
# Same identity if cosine similarity >= 0.363 (or norm-L2 distance <= 1.128):
# OpenCV tutorial "DNN-based Face Detection And Recognition" (tutorial_dnn_face),
# samples/dnn/face_detect.py, and opencv_zoo models/face_recognition_sface/sface.py.
COSINE_SAME = 0.363
# Used threshold: 0.45, from the LFW impostor test (README, "Trade-off"): 0 of
# 2,000 strangers named juan vs 4 at 0.363, keeping 88.7% of own walk-by frames.
RECOGNIZE_AT = 0.45
CONFIRM_WINDOW, CONFIRM_NEED, UNKNOWN_NEED = 3, 2, 3
ADD_MIN_SIM, ADD_DIVERSITY, GALLERY_CAP = 0.6, 0.9, 20
WINDOW = "ESP32-CAM faces (q to quit)"
GREEN, YELLOW = (0, 255, 0), (0, 215, 255)
CONTENT_LENGTH = re.compile(rb"^content-length:\s*(\d+)", re.I)


def decode(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode {len(data)} bytes as an image")
    return img


def capture_frames(url, interval, timeout):
    """Yield (image, seconds the request took), one request per INTERVAL."""
    k, t_start = 0, time.monotonic()
    while True:
        t0 = time.perf_counter()
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read()
        yield decode(data), time.perf_counter() - t0
        k += 1
        yield None, max(0.0, t_start + k * interval - time.monotonic())  # pacing request


def stream_frames(url, timeout):
    """Yield (image, seconds spent receiving it) for each part of an MJPEG stream.

    The firmware sends '--boundary', then 'Content-Type', 'Content-Length' and
    'X-Timestamp' headers, a blank line, and exactly Content-Length JPEG bytes."""
    with urllib.request.urlopen(url, timeout=timeout) as r:
        while True:
            t0 = time.perf_counter()
            length = None
            while True:
                line = r.readline()
                if not line:
                    raise ConnectionError("stream closed by the camera")
                if m := CONTENT_LENGTH.match(line):
                    length = int(m.group(1))
                elif line in (b"\r\n", b"\n") and length is not None:
                    break
            data = r.read(length)
            if len(data) != length:
                raise ConnectionError(f"short read: {len(data)}/{length} bytes")
            yield decode(data), time.perf_counter() - t0


def draw_faces(img, faces, labels=None):
    """Box each face; label it with its score, or with LABELS[i] = (text, known)."""
    for i, f in enumerate(faces):
        x, y, bw, bh = (int(v) for v in f[:4])
        text, color = f"{float(f[14]):.2f}", GREEN
        if labels:
            text, known = labels[i]
            color = GREEN if known else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(img, text, (x, max(y - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def load_gallery():
    """{name: L2-normalised embeddings [k, 128]} from data/gallery/*.npy."""
    gallery = {}
    for path in sorted(GALLERY.glob("*.npy")):
        emb = np.load(path).astype(np.float32).reshape(-1, 128)
        gallery[path.stem] = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return gallery


def embed(rec, img, face):
    """SFace feature of an aligned face (raw, as enroll.py stores it)."""
    return rec.feature(rec.alignCrop(img, face)).ravel().astype(np.float32)


def identify(rec, gallery, img, face, threshold=COSINE_SAME, feat=None):
    """(best name or 'unknown', best cosine over every embedding of every person, raw feature).

    The threshold defaults to the documented 0.363 so impostor_test.py can sweep
    it; --recognize and greeter.py pass RECOGNIZE_AT."""
    feat = embed(rec, img, face) if feat is None else feat
    unit = feat / np.linalg.norm(feat)
    best_name, best = "unknown", -1.0
    for name, emb in gallery.items():
        sim = float((emb @ unit).max())
        if sim > best:
            best_name, best = name, sim
    return (best_name if best >= threshold else "unknown"), best, feat


class Confirmer:
    """Confirm a person after CONFIRM_NEED of the last CONFIRM_WINDOW frames
    match them, and an unknown face after UNKNOWN_NEED of the last frames."""

    def __init__(self):
        self.history = []  # per frame: (set of matched names, any unknown face)

    def update(self, names, unknown):
        self.history = (self.history + [(set(names), bool(unknown))])[-CONFIRM_WINDOW:]
        counts = {}
        for frame_names, _ in self.history:
            for n in frame_names:
                counts[n] = counts.get(n, 0) + 1
        confirmed = {n for n, c in counts.items() if c >= CONFIRM_NEED}
        unknown_confirmed = sum(u for _, u in self.history) >= UNKNOWN_NEED
        return confirmed, unknown_confirmed


def maybe_add(gallery, name, feat, sim):
    """--add-embeddings: append FEAT to NAME's gallery if it is a confident, new view."""
    if sim < ADD_MIN_SIM or len(gallery[name]) >= GALLERY_CAP:
        return None
    unit = feat / np.linalg.norm(feat)
    if float((gallery[name] @ unit).max()) >= ADD_DIVERSITY:
        return None
    path = GALLERY / f"{name}.npy"
    raw = np.load(path).astype(np.float32).reshape(-1, 128)
    np.save(path, np.vstack([raw, feat[None, :]]))
    gallery[name] = np.vstack([gallery[name], unit[None, :]])
    return len(gallery[name])


def show(img, faces, label, color, labels=None):
    """Scale 2x, draw boxes at the new scale and the status line; return True on q."""
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    draw_faces(big, [list(f[:4] * 2) + list(f[4:]) for f in faces], labels)
    cv2.rectangle(big, (0, 0), (big.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(big, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imshow(WINDOW, big)
    return (cv2.waitKey(1) & 0xFF) == ord("q")


def wait(until, preview):
    """Sleep until monotonic time UNTIL; with a window, keep it responsive. True on q."""
    while (left := until - time.monotonic()) > 0:
        if not preview:
            time.sleep(left)
            return False
        if (cv2.waitKey(max(1, min(50, int(left * 1000)))) & 0xFF) == ord("q"):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=("capture", "stream"), default="capture",
                    help="poll /capture (default) or read the MJPEG stream on port 81")
    ap.add_argument("--seconds", type=float, default=30, help="how long to measure (default 30)")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="capture only: seconds between requests (default 1; 0 = back to back)")
    ap.add_argument("--score", type=float, default=0.9,
                    help="YuNet score threshold (default 0.9, as in the opencv_zoo demo)")
    ap.add_argument("--save", action="store_true", help="save frames with ≥1 face (boxes drawn) to data/frames/")
    ap.add_argument("--preview", action="store_true", help="show frames in a window, 2x, with boxes; q quits")
    ap.add_argument("--warmup", type=float, default=0,
                    help="seconds shown first but not measured or saved (default 0)")
    ap.add_argument("--recognize", action="store_true",
                    help=f"name each face from data/gallery/ (SFace cosine >= {RECOGNIZE_AT}, else unknown)")
    ap.add_argument("--add-embeddings", action="store_true",
                    help=f"with --recognize: add confirmed new views (sim >= {ADD_MIN_SIM}, "
                         f"cosine < {ADD_DIVERSITY} to kept) to the gallery, up to {GALLERY_CAP}")
    args = ap.parse_args()

    ip = cam_ip()
    if args.source == "capture":
        url = f"http://{ip}/capture"
        frames_iter = capture_frames(url, args.interval, timeout=max(args.interval * 3, 3))
        how = f"every {args.interval:g} s" if args.interval > 0 else "back to back"
    else:
        url = f"http://{ip}:81/stream"
        frames_iter = stream_frames(url, timeout=10)
        how = "every frame"
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # OpenCV 5 engine notices
    det = cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320), args.score, 0.3, 50,
                                    cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU)
    rec, gallery = None, {}
    if args.recognize:
        gallery = load_gallery()
        if not gallery:
            ap.error(f"--recognize: no embeddings in {GALLERY.relative_to(ROOT)}/ (run pc/enroll.py first)")
        rec = cv2.FaceRecognizerSF.create(str(SFACE), "", cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU)
        print("gallery: " + ", ".join(f"{n} ({len(e)} embeddings)" for n, e in gallery.items())
              + f"; same identity if cosine >= {RECOGNIZE_AT} (documented: {COSINE_SAME}); confirmed after "
              f"{CONFIRM_NEED} of {CONFIRM_WINDOW} frames; add-embeddings={'on' if args.add_embeddings else 'off'}",
              flush=True)
    elif args.add_embeddings:
        ap.error("--add-embeddings needs --recognize")
    confirmer = Confirmer()
    if args.save:
        FRAMES.mkdir(parents=True, exist_ok=True)
    print(f"{url} {how}; warm-up {args.warmup:g} s + measured {args.seconds:g} s; "
          f"YuNet {MODEL.name}, score >= {args.score}; save={'on' if args.save else 'off'}, "
          f"preview={'on' if args.preview else 'off'}", flush=True)
    if args.preview:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    frames, with_face, scores, fetch_t, det_t, errors, saved = 0, 0, [], [], [], 0, 0
    recognized, sims = 0, []
    quit_ = False
    measure_from = time.monotonic() + args.warmup
    t_end = measure_from + args.seconds
    first_measured = last_measured = None
    while not quit_ and time.monotonic() < t_end:
        ts = datetime.now()
        try:
            img, ft = next(frames_iter)
        except StopIteration:
            break
        except Exception as e:
            if time.monotonic() >= measure_from:
                errors += 1
            print(f"{ts:%H:%M:%S.%f}"[:-3] + f"  {args.source} FAILED: {e}", flush=True)
            # A failed generator is finished; start a new connection after a short pause.
            frames_iter = (capture_frames(url, args.interval, timeout=max(args.interval * 3, 3))
                           if args.source == "capture" else stream_frames(url, timeout=10))
            quit_ = wait(min(time.monotonic() + 1, t_end), args.preview)
            continue
        if img is None:  # capture pacing
            quit_ = wait(min(time.monotonic() + ft, t_end), args.preview)
            continue

        now = time.monotonic()
        t = now - measure_from
        warm = t < 0
        tag = "(warm-up) " if warm else ""
        h, w = img.shape[:2]
        det.setInputSize((w, h))
        t0 = time.perf_counter()
        _, faces = det.detect(img)
        dt = time.perf_counter() - t0
        faces = [] if faces is None else faces
        best = max((float(f[14]) for f in faces), default=0.0)
        ids = [identify(rec, gallery, img, f, RECOGNIZE_AT) for f in faces] if args.recognize else []
        confirmed, unknown_confirmed, added = set(), False, []
        if args.recognize:
            confirmed, unknown_confirmed = confirmer.update(
                {n for n, _, _ in ids if n != "unknown"}, any(n == "unknown" for n, _, _ in ids))
            if args.add_embeddings:
                for n, sim, feat in ids:
                    if n in confirmed and (k := maybe_add(gallery, n, feat, sim)):
                        added.append(f"{n}#{k}")
        if not warm:
            frames += 1
            first_measured = first_measured or now
            last_measured = now
            fetch_t.append(ft)
            det_t.append(dt)
            if len(faces):
                with_face += 1
                scores += [float(f[14]) for f in faces]
            if any(name != "unknown" for name, _ in ids):
                recognized += 1
            sims += [sim for _, sim, _ in ids]
        boxes = "  ".join(f"[{float(f[14]):.3f} x={int(f[0])} y={int(f[1])} w={int(f[2])} h={int(f[3])}"
                          + (f" id={ids[i][0]} sim={ids[i][1]:.3f}" if ids else "") + "]"
                          for i, f in enumerate(faces))
        if args.recognize:
            boxes += (f"  confirmed={','.join(sorted(confirmed)) or '-'}"
                      + ("  unknown-confirmed" if unknown_confirmed else "")
                      + (f"  ADDED {' '.join(added)}" if added else ""))
        print(f"{ts:%H:%M:%S.%f}"[:-3] + f"  {tag}t={t:+7.2f}s  {w}x{h}  fetch {ft * 1000:6.1f} ms"
              f"  detect {dt * 1000:5.1f} ms  faces {len(faces)}  best {best:.3f}  {boxes}", flush=True)
        if args.preview:
            if warm:
                label, color = f"WARM-UP {-t:.0f}s  faces {len(faces)}", YELLOW
            else:
                label, color = f"t={t:.0f}s  faces {len(faces)}  with face {with_face}/{frames}", GREEN
            labels = [(f"{name}{'*' if name in confirmed else ''} {sim:.2f}", name != "unknown")
                      for name, sim, _ in ids] if ids else None
            quit_ = show(img, faces, label, color, labels)
        if args.save and not warm and len(faces):
            draw_faces(img, faces)
            cv2.imwrite(str(FRAMES / f"{ts:%Y%m%d-%H%M%S-%f}.jpg"), img)
            saved += 1

    if args.preview:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    if quit_:
        print("stopped with q", flush=True)

    def mean(xs, fmt):
        return format(statistics.mean(xs), fmt) if xs else "n/a"

    span = (last_measured - first_measured) if frames > 1 else 0
    fps = f"{(frames - 1) / span:.2f}" if span > 0 else "n/a"
    print(f"\nsummary ({args.source}): {frames} frames, {fps} fps, {errors} errors; "
          f"frames with ≥1 face {with_face}/{frames}"
          + (f" ({100 * with_face / frames:.0f}%)" if frames else "")
          + f"; mean score {mean(scores, '.3f')}"
          + f"; mean fetch {mean([x * 1000 for x in fetch_t], '.1f')} ms"
          + f"; mean detect {mean([x * 1000 for x in det_t], '.1f')} ms"
          + (f"; saved {saved} frames to {FRAMES.relative_to(ROOT)}/" if args.save else "")
          + (f"; recognized (≥1 known face) {recognized}/{with_face} frames with a face"
             + (f", similarity {min(sims):.3f}–{max(sims):.3f}" if sims else "") if args.recognize else ""),
          flush=True)
    return 0 if frames else 1


if __name__ == "__main__":
    sys.exit(main())
