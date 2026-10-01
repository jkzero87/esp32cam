#!/usr/bin/env python3
"""Impostor test: how often would strangers from LFW be named as an enrolled person?

Takes a fixed random sample (--seed 42, --n 2000) of LFW images
(data/lfw/lfw/<person>/*.jpg). For each image: detect faces at native size
and take the one nearest the centre (the labelled person); rescale the whole
image so that face is a random 50-100 px wide (seeded), like the faces in the
camera's 240x240 frames; detect again on the rescaled image (YuNet, score >=
0.7, as in the walk-by); then align, embed and compare exactly as
capture_detect.py --recognize does (identify(): best cosine over every
embedding). Two galleries for --name: all embeddings in data/gallery/NAME.npy,
and the single embedding closest to the gallery mean.

Prints false accepts at the documented 0.363 and the distribution of the best
similarity; with --walkby (per-frame similarities of the enrolled person's
own walk-by, data/walkby_sims.json) also the threshold trade-off table.
Caches the similarities in data/lfw/impostor_<name>_seed<seed>_n<n>.npz.
Writes no images.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_detect import COSINE_SAME, GALLERY, MODEL, ROOT, SFACE, identify  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

LFW = ROOT / "data" / "lfw" / "lfw"
THRESHOLDS = (COSINE_SAME, 0.40, 0.45, 0.50)


def central_face(det, img):
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None or not len(faces):
        return None
    cx, cy = w / 2, h / 2
    return min(faces, key=lambda f: (f[0] + f[2] / 2 - cx) ** 2 + (f[1] + f[3] / 2 - cy) ** 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="juan")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score", type=float, default=0.7, help="YuNet threshold (walk-by used 0.7)")
    ap.add_argument("--walkby", type=Path, default=ROOT / "data" / "walkby_sims.json")
    args = ap.parse_args()

    cache = ROOT / "data" / "lfw" / f"impostor_{args.name}_seed{args.seed}_n{args.n}.npz"
    raw = np.load(GALLERY / f"{args.name}.npy").astype(np.float32)
    full = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    mean = full.mean(axis=0)
    one_idx = int(np.argmax(full @ (mean / np.linalg.norm(mean))))
    galleries = {"all": {args.name: full}, "one": {args.name: full[one_idx:one_idx + 1]}}

    if cache.exists():
        d = np.load(cache)
        best_all, best_one, n_detected, n_lost = d["best_all"], d["best_one"], int(d["n_detected"]), int(d["n_lost"])
        print(f"using cached similarities {cache.relative_to(ROOT)}")
    else:
        paths = sorted(LFW.glob("*/*.jpg"))
        if not paths:
            sys.exit(f"no LFW images under {LFW.relative_to(ROOT)}")
        rng = random.Random(args.seed)
        sample = rng.sample(paths, args.n)
        widths = [rng.uniform(50, 100) for _ in sample]
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        det = cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320), args.score, 0.3, 50,
                                        cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU)
        rec = cv2.FaceRecognizerSF.create(str(SFACE), "", cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU)
        best_all, best_one, n_lost = [], [], 0
        for k, (path, target) in enumerate(zip(sample, widths), 1):
            img = cv2.imread(str(path))
            f0 = central_face(det, img) if img is not None else None
            if f0 is None:
                n_lost += 1
                continue
            scale = target / float(f0[2])
            small = cv2.resize(img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            f1 = central_face(det, small)
            if f1 is None:
                n_lost += 1
                continue
            best_all.append(identify(rec, galleries["all"], small, f1)[1])
            best_one.append(identify(rec, galleries["one"], small, f1)[1])
            if k % 500 == 0:
                print(f"{k}/{args.n} images, {len(best_all)} faces embedded", flush=True)
        best_all, best_one = np.array(best_all), np.array(best_one)
        n_detected = len(best_all)
        np.savez(cache, best_all=best_all, best_one=best_one, n_detected=n_detected, n_lost=n_lost)

    print(f"\nLFW sample: {args.n} images (seed {args.seed}); {n_detected} faces embedded, "
          f"{n_lost} not detected (at native size or after rescaling to 50-100 px)")
    print(f"gallery '{args.name}': all {len(full)} embeddings vs 1 (index {one_idx}, closest to the mean)\n")
    print(f"{'gallery':<16} {'false accepts @0.363':>21} {'max':>7} {'p99':>7} {'p95':>7} {'median':>7}")
    for label, b in (("20 embeddings", best_all), ("1 embedding", best_one)):
        fa = int((b >= COSINE_SAME).sum())
        print(f"{label:<16} {f'{fa} / {n_detected}':>21} {b.max():>7.3f} {np.percentile(b, 99):>7.3f} "
              f"{np.percentile(b, 95):>7.3f} {np.median(b):>7.3f}")

    if args.walkby.exists():
        w = json.loads(args.walkby.read_text())
        own = np.array([max(x["sim"] for x in fr["faces"]) for fr in w["frames"]])
        true145 = own[own >= COSINE_SAME]
        print(f"\nTrade-off (own walk-by: {len(own)} frames with a face, {len(true145)} recognized at "
              f"{COSINE_SAME}; strangers: {n_detected} LFW faces, 20-embedding gallery)")
        print(f"{'threshold':>9} {'own frames recognized':>22} {'of the 145':>11} {'false accepts':>14} {'per 2,000':>10}")
        for t in THRESHOLDS:
            ok = int((own >= t).sum())
            fa = int((best_all >= t).sum())
            print(f"{t:>9.3f} {f'{ok}/{len(own)} ({100 * ok / len(own):.1f}%)':>22} "
                  f"{100 * ok / len(true145):>10.1f}% {f'{fa}/{n_detected}':>14} {2000 * fa / n_detected:>10.1f}")


if __name__ == "__main__":
    main()
