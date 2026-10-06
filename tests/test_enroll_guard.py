#!/usr/bin/env python3
"""Offline test of the enrollment guard in greeter.unknown_exchange: a name that is
already in cam.people (any case, extra spaces) is refused and never reused, even with
an empty throwaway --gallery-dir (the 2026-10-06 live test re-stamped a real row's
consent_at that way). Uses throwaway people 'test_unit_guard*' in the local Postgres
and temporary galleries; all removed. No model, no network (tests/offline.py)."""
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline import guard  # noqa: E402

guard("tests/test_enroll_guard.py")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pc"))
import greeter  # noqa: E402
from memory import Memory  # noqa: E402

FAIL = []


def check(desc, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {desc}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAIL.append(desc)


def feature(seed):
    v = np.random.default_rng(seed).standard_normal(128).astype(np.float32)
    return v / np.linalg.norm(v)


def run_exchange(st, mem, answers, metrics_file):
    """unknown_exchange with scripted answers; each answer also 'shows' one face to the collector."""
    script = iter(answers)

    def fake_read_line(prompt, timeout, fresh=False):
        with st.lock:
            if st.collector is not None:
                st.collector.append(feature(len(st.collector)))
        return next(script, None)

    greeter.read_line = fake_read_line
    greeter.METRICS = metrics_file
    return greeter.unknown_exchange(st, mem, SimpleNamespace(metrics=True))


tmp = Path(tempfile.mkdtemp(prefix="test_enroll_guard_"))
real_gallery, live_gallery = tmp / "gallery", tmp / "gallery_live_test"
real_gallery.mkdir()
metrics_file = tmp / "metrics.jsonl"
mem = Memory(live_gallery)
existing, new = "test_unit_guard", "test_unit_guard_new"
old_consent = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

with mem._conn() as c:
    c.execute("DELETE FROM cam.people WHERE name LIKE 'test_unit_guard%'")
    pid = c.execute("INSERT INTO cam.people (name, consent_at) VALUES (%s, %s) RETURNING id",
                    (existing, old_consent)).fetchone()[0]
    c.execute("INSERT INTO cam.facts (person_id, fact, source_conversation_at) VALUES (%s, %s, %s)",
              (pid, "Le gusta el ajedrez", old_consent))
np.save(real_gallery / f"{existing}.npy", np.vstack([feature(99)]))
gallery_bytes = (real_gallery / f"{existing}.npy").read_bytes()


def snapshot():
    with mem._conn() as c:
        row = c.execute("SELECT id, name, created_at, consent_at FROM cam.people WHERE id = %s", (pid,)).fetchone()
        facts = c.execute("SELECT id, fact, created_at FROM cam.facts WHERE person_id = %s ORDER BY id",
                          (pid,)).fetchall()
    return row, facts


before = snapshot()
try:
    print("== name_taken / enroll")
    check("name_taken exact", mem.name_taken(existing), True)
    check("name_taken ignores case and spaces", mem.name_taken(f"  {existing.upper()} "), True)
    check("enroll of a taken name returns None", mem.enroll(existing), None)
    check("existing row untouched by enroll", snapshot(), before)

    print("== unknown_exchange: only existing names offered")
    st = greeter.State(live_gallery)
    got = run_exchange(st, mem, ["sí", existing, f" {existing.upper()} ", existing], metrics_file)
    check("returns False (nothing enrolled)", got, False)
    check("existing row: consent_at, created_at and facts unchanged", snapshot(), before)
    check("real gallery file unchanged", (real_gallery / f"{existing}.npy").read_bytes(), gallery_bytes)
    check("nothing written to the throwaway gallery", sorted(p.name for p in live_gallery.glob("*")), [])
    with mem._conn() as c:
        n = c.execute("SELECT count(*) FROM cam.people WHERE name LIKE 'test_unit_guard%'").fetchone()[0]
    check("no new row", n, 1)

    print("== unknown_exchange: existing name, then a new one")
    st = greeter.State(live_gallery)
    got = run_exchange(st, mem, ["sí", existing, new], metrics_file)
    check("returns True (enrolled under the new name)", got, True)
    check("existing row still unchanged", snapshot(), before)
    check("gallery file for the new name only", sorted(p.name for p in live_gallery.glob("*")), [f"{new}.npy"])
    with mem._conn() as c:
        row = c.execute("SELECT consent_at IS NOT NULL FROM cam.people WHERE name = %s", (new,)).fetchone()
    check("new row has consent_at", row, (True,))
    last = metrics_file.read_text().splitlines()[-1]
    check("metrics 'enrolled' event carries the name", ('"event": "enrolled"' in last, f'"name": "{new}"' in last),
          (True, True))
finally:
    with mem._conn() as c:
        c.execute("DELETE FROM cam.people WHERE name LIKE 'test_unit_guard%'")
    for p in sorted(tmp.rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    tmp.rmdir()

print("ALL OK" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
