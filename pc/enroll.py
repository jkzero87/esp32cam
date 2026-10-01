#!/usr/bin/env python3
"""Enroll one person for SFace recognition from the ESP32-CAM stream (CPU).

For --seconds (default 30) reads http://<CAM_IP>:81/stream (CAM_IP from the
environment, default 192.168.20.71), detects faces with YuNet, and for every
frame with exactly one face aligns and embeds it with SFace. The gallery
keeps a diverse set: an embedding is added only if its cosine similarity to
every embedding already kept for this person is below --diversity (default
0.9), up to --max (default 20).

Writes only the embeddings (float32, shape [k, 128]) to
data/gallery/NAME.npy (gitignored). No images are written. Frames with
several faces are skipped so nobody else gets enrolled by accident.
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_detect import GREEN, MODEL, ROOT, WINDOW, YELLOW, show, stream_frames  # noqa: E402

import cv2  # noqa: E402  (after capture_detect sets QT_QPA_PLATFORM)
import numpy as np  # noqa: E402

SFACE = ROOT / "models" / "face_recognition_sface_2021dec.onnx"
GALLERY = ROOT / "data" / "gallery"


def cosine(a, b):
    """Cosine similarity, as FaceRecognizerSF.match(..., FR_COSINE) computes it."""
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="person's name (letters, digits, _ and -)")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--diversity", type=float, default=0.9,
                    help="keep an embedding only if its cosine to every kept one is below this (default 0.9)")
    ap.add_argument("--max", type=int, default=20, help="maximum embeddings kept (default 20)")
    ap.add_argument("--score", type=float, default=0.9, help="YuNet score threshold (default 0.9)")
    ap.add_argument("--preview", action="store_true", help="show the stream in a window, 2x")
    ap.add_argument("--replace", action="store_true", help="overwrite an existing gallery for NAME")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        ap.error("--name may only contain letters, digits, '_' and '-'")
    out = GALLERY / f"{args.name}.npy"
    if out.exists() and not args.replace:
        ap.error(f"{out.relative_to(ROOT)} exists; pass --replace to overwrite it")

    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    det = cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320), args.score, 0.3, 50,
                                    cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU)
    rec = cv2.FaceRecognizerSF.create(str(SFACE), "", cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU)
    url = f"http://{os.environ.get('CAM_IP', '192.168.20.71')}:81/stream"
    print(f"enrolling '{args.name}' from {url} for {args.seconds:g} s; keep if cosine to all kept "
          f"< {args.diversity}, up to {args.max}", flush=True)
    if args.preview:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    kept, frames, one_face, multi = [], 0, 0, 0
    t_end = time.monotonic() + args.seconds
    quit_ = False
    for img, _ in stream_frames(url, timeout=10):
        t_left = t_end - time.monotonic()
        if t_left <= 0 or quit_:
            break
        frames += 1
        h, w = img.shape[:2]
        det.setInputSize((w, h))
        _, faces = det.detect(img)
        faces = [] if faces is None else faces
        note = ""
        if len(faces) > 1:
            multi += 1
            note = f"{len(faces)} faces: skipped"
        elif len(faces) == 1:
            one_face += 1
            feat = rec.feature(rec.alignCrop(img, faces[0]))
            nearest = max((cosine(feat, k) for k in kept), default=0.0)
            if len(kept) < args.max and nearest < args.diversity:
                kept.append(feat.copy())
                note = f"KEPT #{len(kept)} (nearest kept cosine {nearest:.3f})"
            else:
                note = f"similar to kept (cosine {nearest:.3f})" if nearest >= args.diversity else "gallery full"
        print(f"{time.strftime('%H:%M:%S')}  faces {len(faces)}  "
              + (f"score {float(faces[0][14]):.3f}  " if len(faces) == 1 else "") + note, flush=True)
        if args.preview:
            label = f"ENROLL {args.name}: kept {len(kept)}/{args.max}  {t_left:.0f}s left"
            quit_ = show(img, faces, label, GREEN if len(faces) == 1 else YELLOW)
    if args.preview:
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    if not kept:
        print(f"\nno single-face frames: nothing written ({frames} frames, {multi} with several faces)")
        return 1
    GALLERY.mkdir(parents=True, exist_ok=True)
    np.save(out, np.vstack(kept).astype(np.float32))
    print(f"\nkept {len(kept)} embeddings for '{args.name}' -> {out.relative_to(ROOT)} "
          f"({frames} frames, {one_face} with one face, {multi} with several faces skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
