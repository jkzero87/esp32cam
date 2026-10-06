#!/usr/bin/env python3
"""Offline tests: a conversation ends the way it does between people. A farewell ends
it, the person leaving the camera for --leave-seconds ends it, and either way the
facts are extracted and stored; a goodbye word inside a normal sentence does not end
it; the terminal read never stalls the camera loop. Uses a throwaway person
'test_unit_conv' in the local Postgres and a temporary gallery; all removed. The model
is a stub. No network (tests/offline.py)."""
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline import guard  # noqa: E402

guard("tests/test_conversation_end.py")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pc"))
import greeter  # noqa: E402
from memory import Memory, wants_goodbye  # noqa: E402

REAL_READ_LINE = greeter.read_line  # the select-based read, used as is in the leaving test

FAIL = []


def check(desc, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {desc}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAIL.append(desc)


print("== wants_goodbye")
for text in ["chao", "Chao!", "Adiós", "adios, gracias", "bueno, me voy", "Nos vemos", "hasta luego Prueba",
             "gracias, hasta mañana", "Ya me voy, chao", "me tengo que ir", "HASTA LUEGO"]:
    check(f"farewell: {text!r}", wants_goodbye(text, "prueba"), True)
for text in ["me voy a montar bici el sábado", "nos vemos el sábado en el parque", "adiós a las vacaciones, mañana trabajo",
             "hasta luego me dijo mi jefe que viniera", "me gusta el ciclismo", "chao pescao", ""]:
    check(f"not a farewell: {text!r}", wants_goodbye(text, "prueba"), False)

FACT = "Le gusta el ciclismo de montaña"
SAID = "Me gusta mucho el ciclismo de montaña"
llm_calls = []


def fake_llm(url, messages, stream_to_terminal=True, **kw):
    llm_calls.append(messages[-1]["content"])
    if not stream_to_terminal:                   # the fact extraction
        return '{"facts": ["%s"]}' % FACT, 0.1, 10, 10.0
    return "¡Qué bien!", 0.1, 3, 10.0


greeter.llm = fake_llm
tmp = Path(tempfile.mkdtemp(prefix="test_conversation_end_"))
gdir = tmp / "gallery"
gdir.mkdir()
greeter.METRICS = tmp / "metrics.jsonl"
mem = Memory(gdir)
who = "test_unit_conv"
args = SimpleNamespace(metrics=True, log_greeting=False)


def fresh_person():
    with mem._conn() as c:
        c.execute("DELETE FROM cam.people WHERE name LIKE 'test_unit_conv%'")
    np.save(gdir / f"{who}.npy", np.ones((1, 128), np.float32))
    mem.enroll(who)
    return greeter.State(gdir)


def facts_events():
    return [json.loads(line) for line in greeter.METRICS.read_text().splitlines()
            if json.loads(line)["event"] == "facts"] if greeter.METRICS.exists() else []


try:
    print("== a farewell ends the conversation; the fact is stored")
    st = fresh_person()
    script = iter([SAID, "bueno, chao"])
    greeter.read_line = lambda prompt, timeout, fresh=False, stop=None: next(script, None)
    greeter.person_conversation(st, mem, args, "http://unused", who, 0.0)
    ev = facts_events()[-1]
    check("ended by the goodbye", ev["end"], "goodbye")
    check("fact stored", (ev["stored"], mem.recent_facts(who)), (1, [FACT]))
    check("status: Listo: recordé 1 cosa(s)", st.status, "Listo: recordé 1 cosa(s)")
    check("no longer talking", st.talking, None)

    print("== a goodbye word inside a normal sentence does not end it")
    st = fresh_person()
    llm_calls.clear()
    script = iter(["me voy a montar bici el sábado", SAID, "adiós"])
    greeter.read_line = lambda prompt, timeout, fresh=False, stop=None: next(script, None)
    greeter.person_conversation(st, mem, args, "http://unused", who, 0.0)
    ev = facts_events()[-1]
    check("the sentence got a reply and the talk went on", "me voy a montar bici el sábado" in llm_calls, True)
    check("ended by the later 'adiós'", (ev["end"], ev["user_lines"]), ("goodbye", 3))

    print("== leaving the camera for --leave-seconds ends it; the fact is stored; reads never stall frames")
    st = fresh_person()
    r, w = os.pipe()
    saved_stdin = sys.stdin
    sys.stdin = os.fdopen(r, "r")
    greeter.read_line = REAL_READ_LINE
    try:
        st = greeter.State(gdir)
        conf = greeter.Confirmer()
        th = threading.Thread(target=greeter.person_conversation, daemon=True,
                              args=(st, mem, args, "http://unused", who, 0.0))
        llm_calls.clear()
        th.start()
        here = [(who, 0.9, None)]
        frame_times = []

        def frames_until(cond, seconds):
            end = time.monotonic() + seconds
            while time.monotonic() < end and not cond():
                t0 = time.perf_counter()
                greeter.frame_events(st, conf, st.gallery, here, time.monotonic(), 5, 8)
                frame_times.append(time.perf_counter() - t0)
                time.sleep(0.02)
            return cond()

        check("conversation open (camera loop running)", frames_until(lambda: st.talking == who, 3), True)
        os.write(w, (SAID + "\n").encode())
        check("the reply was read while frames kept flowing", frames_until(lambda: SAID in llm_calls, 3), True)
        check("still talking while in view", (st.talking, st.talk_end.is_set()), (who, False))
        greeter.frame_events(st, conf, st.gallery, [], time.monotonic() + 7.5, 5, 8)
        check("7.5 s out of view: not over yet", st.talk_end.is_set(), False)
        t0 = time.perf_counter()
        greeter.frame_events(st, conf, st.gallery, [], time.monotonic() + 8.5, 5, 8)
        frame_times.append(time.perf_counter() - t0)
        th.join(timeout=3)
        t_join = time.perf_counter() - t0
        check("8.5 s out of view ends the conversation", (st.talk_end.is_set(), th.is_alive()), (True, False))
        check("conversation thread let go of the read within 1 s", t_join < 1.0, True)
        check("no frame waited on the terminal (every frame_events < 50 ms)", max(frame_times) < 0.05, True)
        ev = facts_events()[-1]
        check("ended by leaving; fact stored", (ev["end"], ev["stored"]), ("left", 1))
        check("status: Listo: recordé 1 cosa(s)", st.status, "Listo: recordé 1 cosa(s)")
    finally:
        os.close(w)
        sys.stdin = saved_stdin
finally:
    with mem._conn() as c:
        c.execute("DELETE FROM cam.people WHERE name LIKE 'test_unit_conv%'")
    for p in sorted(tmp.rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    tmp.rmdir()

print("ALL OK" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
