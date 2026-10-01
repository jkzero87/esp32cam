#!/usr/bin/env python3
"""Run YuNet face detection on the PC (CPU) on frames from the ESP32-CAM.

Two frame sources (CAM_IP from the environment, default 192.168.20.71):
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
    args = ap.parse_args()

    ip = os.environ.get("CAM_IP", "192.168.20.71")
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
    if args.save:
        FRAMES.mkdir(parents=True, exist_ok=True)
    print(f"{url} {how}; warm-up {args.warmup:g} s + measured {args.seconds:g} s; "
          f"YuNet {MODEL.name}, score >= {args.score}; save={'on' if args.save else 'off'}, "
          f"preview={'on' if args.preview else 'off'}", flush=True)
    if args.preview:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    frames, with_face, scores, fetch_t, det_t, errors, saved = 0, 0, [], [], [], 0, 0
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
        if not warm:
            frames += 1
            first_measured = first_measured or now
            last_measured = now
            fetch_t.append(ft)
            det_t.append(dt)
            if len(faces):
                with_face += 1
                scores += [float(f[14]) for f in faces]
        boxes = "  ".join(f"[{float(f[14]):.3f} x={int(f[0])} y={int(f[1])} w={int(f[2])} h={int(f[3])}]"
                          for f in faces)
        print(f"{ts:%H:%M:%S.%f}"[:-3] + f"  {tag}t={t:+7.2f}s  {w}x{h}  fetch {ft * 1000:6.1f} ms"
              f"  detect {dt * 1000:5.1f} ms  faces {len(faces)}  best {best:.3f}  {boxes}", flush=True)
        if args.preview:
            if warm:
                label, color = f"WARM-UP {-t:.0f}s  faces {len(faces)}", YELLOW
            else:
                label, color = f"t={t:.0f}s  faces {len(faces)}  with face {with_face}/{frames}", GREEN
            quit_ = show(img, faces, label, color)
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
          + (f"; saved {saved} frames to {FRAMES.relative_to(ROOT)}/" if args.save else ""), flush=True)
    return 0 if frames else 1


if __name__ == "__main__":
    sys.exit(main())
