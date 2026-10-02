#!/usr/bin/env python3
"""Offline test of greeter.CtrlC (Ctrl+C during the memory save), with real
SIGINTs sent to this process. No model, no network (tests/offline.py)."""
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline import guard  # noqa: E402

guard("tests/test_ctrlc.py")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pc"))
from greeter import CtrlC  # noqa: E402

FAIL = []


def check(desc, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {desc}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAIL.append(desc)


def sigint():
    """Send SIGINT to ourselves; return what the handler raised (or None)."""
    try:
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.05)  # Python runs the handler between bytecodes, in the main thread
        return None
    except KeyboardInterrupt:
        return "KeyboardInterrupt"


exits = []
handler = CtrlC(exit_fn=exits.append)
signal.signal(signal.SIGINT, handler)

print("== Ctrl+C before saving")
check("1st Ctrl+C stops the loop (KeyboardInterrupt)", sigint(), "KeyboardInterrupt")
check("no forced exit", exits, [])

print("== Ctrl+C while saving")
handler.saving = True
check("2nd Ctrl+C is ignored (no exception)", sigint(), None)
check("still no forced exit", exits, [])
check("3rd Ctrl+C force-quits with code 130", (sigint(), exits), (None, [130]))

print("== a fresh handler while saving, one press only")
h2 = CtrlC(exit_fn=exits.append)
h2.saving = True
exits.clear()
h2()
check("a single press during the save never exits", exits, [])

print("== join() of a saving worker survives an ignored Ctrl+C (the 17:33 failure)")
import threading  # noqa: E402
saved = []
worker = threading.Thread(target=lambda: (time.sleep(1.0), saved.append(True)), daemon=True)
h3 = CtrlC(exit_fn=exits.append)
h3.saving = True
signal.signal(signal.SIGINT, h3)
exits.clear()
worker.start()
threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGINT)).start()
try:
    worker.join(timeout=5)
    interrupted = False
except KeyboardInterrupt:
    interrupted = True
check("join not interrupted, save finished, no exit", (interrupted, saved, exits), (False, [True], []))

signal.signal(signal.SIGINT, signal.default_int_handler)
print("ALL OK" if not FAIL else f"{len(FAIL)} FAILURES")
sys.exit(1 if FAIL else 0)
