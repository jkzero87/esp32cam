#!/usr/bin/env python3
"""Print what the greeter has stored: every person in cam.people (name,
created_at, consent_at, number of facts, embeddings in <gallery>/<name>.npy)
and each person's facts as text, newest first. Read-only (the session is
READ ONLY); connects as the role in .env (camuser).

Also lists gallery files with no cam.people row.

Usage:
  .venv/bin/python pc/status.py [--gallery-dir DIR]   # default data/gallery
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory import ROOT, Memory  # noqa: E402

GALLERY = ROOT / "data" / "gallery"


def embeddings(path):
    if not path.exists():
        return "no file"
    try:
        return str(np.load(path).shape[0])
    except Exception as e:
        return f"unreadable ({e.__class__.__name__})"


def fmt(ts):
    return ts.astimezone().strftime("%Y-%m-%d %H:%M:%S") if ts else "NULL"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gallery-dir", type=Path, default=GALLERY, help="gallery directory (default data/gallery)")
    gallery = ap.parse_args().gallery_dir

    mem = Memory(gallery)
    with mem._conn() as c:
        c.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        user = c.execute("SELECT current_user").fetchone()[0]
        people = c.execute(
            "SELECT p.id, p.name, p.created_at, p.consent_at, count(f.id) FROM cam.people p "
            "LEFT JOIN cam.facts f ON f.person_id = p.id GROUP BY p.id ORDER BY p.name").fetchall()
        facts = c.execute("SELECT person_id, fact, created_at FROM cam.facts "
                          "ORDER BY person_id, created_at DESC, id DESC").fetchall()

    print(f"as {user}; gallery {gallery}; {len(people)} people, {len(facts)} facts")
    for pid, name, created, consent, n_facts in people:
        print(f"\n{name}  (id {pid})")
        print(f"  created_at  {fmt(created)}")
        print(f"  consent_at  {fmt(consent)}")
        print(f"  facts       {n_facts}")
        print(f"  embeddings  {embeddings(gallery / f'{name}.npy')}")
        for _, fact, at in (f for f in facts if f[0] == pid):
            print(f"    - {fact}  [{fmt(at)}]")

    orphans = sorted(p.stem for p in gallery.glob("*.npy") if p.stem not in {r[1] for r in people})
    if orphans:
        print(f"\ngallery files with no cam.people row: {', '.join(orphans)}")


if __name__ == "__main__":
    main()
