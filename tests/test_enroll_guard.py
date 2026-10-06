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

    print("== camera frames around an enrollment (2026-10-06: a 2nd 'no te conozco' right after enrolling)")
    st = greeter.State(tmp / "gallery_frames")
    confirmer = greeter.Confirmer()
    face = feature(7)
    unknown_ids = [("unknown", 0.0, face)]
    evs = []
    st.unknown_busy = True                       # an unknown exchange is in progress
    old_gallery = st.gallery
    for k in range(3):                           # the face is unknown while the exchange runs
        evs += greeter.frame_events(st, confirmer, old_gallery, unknown_ids, 100.0 + k, 5)[1]
    check("no event while the exchange is busy", evs, [])
    # The exchange enrolls the face: gallery reloaded, last_seen set, then the worker frees unknown_busy.
    (tmp / "gallery_frames").mkdir()
    np.save(tmp / "gallery_frames" / "nuevo.npy", np.vstack([face]))
    st.gallery = greeter.load_gallery(tmp / "gallery_frames")
    st.last_seen["nuevo"] = 103.0
    st.unknown_busy = False
    # A frame identified with the old gallery lands after the reload ...
    evs += greeter.frame_events(st, confirmer, old_gallery, unknown_ids, 103.1, 5)[1]
    # ... then frames identified with the new one.
    ident = greeter.identify(None, st.gallery, None, None, greeter.RECOGNIZE_AT, feat=face)
    check("the enrolled embedding is recognized as the new name", ident[0], "nuevo")
    confirmed = set()
    for k in range(3):
        c, e = greeter.frame_events(st, confirmer, st.gallery, [ident], 103.2 + k * 0.1, 5)
        confirmed |= c
        evs += e
    check("no second unknown exchange after the enrollment", [e[0] for e in evs if e[0] == "unknown"], [])
    check("the new name is confirmed", "nuevo" in confirmed, True)
    check("not re-greeted right after enrolling", [e for e in evs if e[0] == "person"], [])
    # Still in view for a while: last_seen keeps refreshing, so no greeting (the 16:04 rerun).
    for k in range(20):
        evs += greeter.frame_events(st, confirmer, st.gallery, [ident], 104.0 + k * 0.3, 5)[1]
    check("no greeting while the person stays in view", [e for e in evs if e[0] == "person"], [])
    # Leaves (frames with no face) for more than 5 s, then comes back.
    t = 110.0
    for k in range(25):
        t += 0.3
        evs += greeter.frame_events(st, confirmer, st.gallery, [], t, 5)[1]
    for k in range(3):
        t += 0.3
        evs += greeter.frame_events(st, confirmer, st.gallery, [ident], t, 5)[1]
    check("greeted by name after leaving > 5 s and coming back",
          [e[:2] for e in evs if e[0] == "person"], [("person", "nuevo")])
    check("still no unknown exchange", [e[0] for e in evs if e[0] == "unknown"], [])

    print("== 'olvídame' with the person still in view (2026-10-06: forgotten person greeted and re-created)")
    gone = "test_unit_guard_gone"
    gdir = tmp / "gallery_forget"
    gdir.mkdir()
    fmem = Memory(gdir)
    face2 = feature(11)
    np.save(gdir / f"{gone}.npy", np.vstack([face2]))
    check("enroll the person to forget", fmem.enroll(gone) is not None, True)
    st = greeter.State(gdir)
    confirmer = greeter.Confirmer()
    seen = [(gone, 0.99, face2)]
    evs = []
    for k in range(3):                           # in view and confirmed
        evs += greeter.frame_events(st, confirmer, st.gallery, seen, 200.0 + k * 0.3, 5)[1]
    check("greeted once when first confirmed", [e[:2] for e in evs if e[0] == "person"], [("person", gone)])
    pre_forget_gallery = st.gallery
    llm_calls = []
    greeter.llm = lambda url, messages, **kw: (llm_calls.append(1) or ("Hola", 0.1, 2, 10.0))
    script = iter(["olvídame", "sí"])
    greeter.read_line = lambda prompt, timeout, fresh=False: next(script, None)
    greeter.person_conversation(st, fmem, SimpleNamespace(metrics=False), "http://unused", gone, 0.0)
    check("forget removed the row", fmem.existing_id(gone), None)
    check("forget removed the gallery file", (gdir / f"{gone}.npy").exists(), False)
    # A frame identified before the forget lands after it, then frames that still see the face.
    evs = greeter.frame_events(st, confirmer, pre_forget_gallery, seen, 201.0, 5)[1]
    for k in range(3):
        ids = [greeter.identify(None, st.gallery, None, None, greeter.RECOGNIZE_AT, feat=face2)] \
            if st.gallery else [("unknown", 0.0, face2)]
        evs += greeter.frame_events(st, confirmer, st.gallery, ids, 201.1 + k * 0.3, 5)[1]
    check("stale frames do not greet the forgotten person", [e for e in evs if e[0] == "person"], [])
    # Even if a greeting for that name was already queued, it neither greets nor creates a row.
    calls_before = len(llm_calls)
    greeter.person_conversation(st, fmem, SimpleNamespace(metrics=False), "http://unused", gone, 0.0)
    check("a queued greeting for a forgotten name does not call the model", len(llm_calls), calls_before)
    check("no row re-created for the forgotten name", fmem.existing_id(gone), None)
    check("add_facts for a name with no row stores nothing and creates nothing",
          (fmem.add_facts(gone, ["Le gusta el té"], datetime.now(timezone.utc)), fmem.existing_id(gone)), (0, None))
finally:
    with mem._conn() as c:
        c.execute("DELETE FROM cam.people WHERE name LIKE 'test_unit_guard%'")
    for p in sorted(tmp.rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    tmp.rmdir()

print("ALL OK" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
