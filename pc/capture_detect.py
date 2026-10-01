#!/usr/bin/env python3
"""Poll the ESP32-CAM once a second and run YuNet face detection on the PC (CPU).

Every --interval seconds (default 1) fetches http://<CAM_IP>/capture (CAM_IP
from the environment, default 192.168.20.71), detects faces with YuNet and
prints one line per frame: timestamp, fetch time, detection time, number of
faces, then score and box (x, y, w, h) of each face. Stops after --seconds and
prints a summary: detection rate, mean score, mean fetch and detection times.

--save writes frames that contain at least one face, with the boxes drawn, to
data/frames/. Frames without a face are never written, and nothing is written
to disk without --save.

--preview shows each frame in a window, scaled up 2x, with the face boxes,
scores and the running count of frames with a face. Press q to stop early.
--warmup N first runs N seconds that are shown (labelled WARM-UP) but not
measured, printed as "(warm-up)", or saved; the measured --seconds follow
immediately in the same window.
"""
import argparse
import os
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
WINDOW = "ESP32-CAM faces (q to quit)"
GREEN, YELLOW = (0, 255, 0), (0, 215, 255)


def draw_faces(img, faces):
    for f in faces:
        x, y, bw, bh = (int(v) for v in f[:4])
        cv2.rectangle(img, (x, y), (x + bw, y + bh), GREEN, 2)
        cv2.putText(img, f"{float(f[14]):.2f}", (x, max(y - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)


def show(img, faces, label, color):
    """Scale 2x, draw boxes at the new scale and the status line; return True on q."""
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    draw_faces(big, [list(f[:4] * 2) + list(f[4:]) for f in faces])
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


def fetch(url, timeout):
    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = r.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode {len(data)} bytes as an image")
    return img, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=30, help="how long to run (default 30)")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between captures (default 1)")
    ap.add_argument("--score", type=float, default=0.9,
                    help="YuNet score threshold (default 0.9, as in the opencv_zoo demo)")
    ap.add_argument("--save", action="store_true", help="save frames with ≥1 face (boxes drawn) to data/frames/")
    ap.add_argument("--preview", action="store_true", help="show frames in a window, 2x, with boxes; q quits")
    ap.add_argument("--warmup", type=float, default=0,
                    help="seconds shown first but not measured or saved (default 0)")
    args = ap.parse_args()

    ip = os.environ.get("CAM_IP", "192.168.20.71")
    url = f"http://{ip}/capture"
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # OpenCV 5 engine notices
    det = cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320), args.score, 0.3, 50,
                                    cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU)
    if args.save:
        FRAMES.mkdir(parents=True, exist_ok=True)
    print(f"{url} every {args.interval:g} s; warm-up {args.warmup:g} s + measured {args.seconds:g} s; "
          f"YuNet {MODEL.name}, score >= {args.score}; save={'on' if args.save else 'off'}, "
          f"preview={'on' if args.preview else 'off'}", flush=True)
    if args.preview:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    frames, with_face, scores, fetch_t, det_t, errors, saved = 0, 0, [], [], [], 0, 0
    quit_ = False
    t_start = time.monotonic()
    measure_from = t_start + args.warmup
    t_end = measure_from + args.seconds
    k = 0
    while not quit_ and time.monotonic() < t_end:
        ts = datetime.now()
        warm = time.monotonic() < measure_from
        tag = "(warm-up) " if warm else ""
        try:
            img, ft = fetch(url, timeout=max(args.interval * 3, 3))
        except Exception as e:
            if not warm:
                errors += 1
            print(f"{ts:%H:%M:%S.%f}"[:-3] + f"  {tag}fetch FAILED: {e}", flush=True)
        else:
            h, w = img.shape[:2]
            det.setInputSize((w, h))
            t0 = time.perf_counter()
            _, faces = det.detect(img)
            dt = time.perf_counter() - t0
            faces = [] if faces is None else faces
            if not warm:
                frames += 1
                fetch_t.append(ft)
                det_t.append(dt)
                if len(faces):
                    with_face += 1
                    scores += [float(f[14]) for f in faces]
            boxes = "  ".join(f"[{float(f[14]):.3f} x={int(f[0])} y={int(f[1])} w={int(f[2])} h={int(f[3])}]"
                              for f in faces)
            print(f"{ts:%H:%M:%S.%f}"[:-3] + f"  {tag}{w}x{h}  fetch {ft * 1000:6.1f} ms"
                  f"  detect {dt * 1000:5.1f} ms  faces {len(faces)}  {boxes}", flush=True)
            if args.preview:
                if warm:
                    label = f"WARM-UP {measure_from - time.monotonic():.0f}s  faces {len(faces)}"
                    color = YELLOW
                else:
                    label = f"faces {len(faces)}  frames with face {with_face}/{frames}"
                    color = GREEN
                quit_ = show(img, faces, label, color)
            if args.save and not warm and len(faces):
                draw_faces(img, faces)
                cv2.imwrite(str(FRAMES / f"{ts:%Y%m%d-%H%M%S-%f}.jpg"), img)
                saved += 1
        k += 1
        quit_ = quit_ or wait(min(t_start + k * args.interval, t_end), args.preview)

    if args.preview:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    if quit_:
        print("stopped with q", flush=True)

    def mean(xs, fmt):
        return format(statistics.mean(xs), fmt) if xs else "n/a"

    print(f"\nsummary: {frames} frames, {errors} fetch errors; frames with ≥1 face {with_face}/{frames}"
          + (f" ({100 * with_face / frames:.0f}%)" if frames else "")
          + f"; mean score {mean(scores, '.3f')}"
          + f"; mean fetch {mean([t * 1000 for t in fetch_t], '.1f')} ms"
          + f"; mean detect {mean([t * 1000 for t in det_t], '.1f')} ms"
          + (f"; saved {saved} frames to {FRAMES.relative_to(ROOT)}/" if args.save else ""), flush=True)
    return 0 if frames else 1


if __name__ == "__main__":
    sys.exit(main())
