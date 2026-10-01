#!/usr/bin/env python3
"""Offline tests for pc/memory.py: fact parsing and filters, consent/name
parsing, and add/read/forget against the cam schema in de_postgres (uses a
throwaway person 'test_unit_mem' and a temporary gallery; both are removed)."""
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pc"))
from memory import Memory, is_clear_yes, name_slug, parse_facts, wants_forget  # noqa: E402

FAIL = []


def check(desc, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {desc}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAIL.append(desc)


print("== parse_facts")
check("plain JSON object", parse_facts('{"facts": ["Le gusta el ciclismo", "Trabaja en Medellín"]}'),
      (["Le gusta el ciclismo", "Trabaja en Medellín"], []))
check("fenced JSON + whitespace + dedup",
      parse_facts('```json\n{"facts": ["  Toca la guitarra ", "toca la guitarra"]}\n```')[0], ["Toca la guitarra"])
check("bare list", parse_facts('["Prefiere el café sin azúcar"]')[0], ["Prefiere el café sin azúcar"])
check("JSON inside prose", parse_facts('Claro: {"facts": ["Juega ajedrez"]} listo')[0], ["Juega ajedrez"])
check("garbage -> nothing", parse_facts("no hay nada que extraer"), ([], []))
check("empty list", parse_facts('{"facts": []}'), ([], []))
check("non-strings ignored", parse_facts('{"facts": [1, null, "Corre maratones"]}')[0], ["Corre maratones"])
kept, dropped = parse_facts('{"facts": ["Tiene diabetes", "Gana 3000 dólares al mes", "Su contraseña es gato123",'
                            ' "Su DNI es 12345678", "Mi hermana vive en Cali", "Le gusta la salsa"]}')
check("health/money/secrets/IDs/third parties dropped, durable fact kept", kept, ["Le gusta la salsa"])
check("drop reasons", sorted({r for _, r in dropped}), ["health", "money", "secrets/IDs", "third parties"])
kept, dropped = parse_facts('{"facts": ["A", "B", "C", "D"]}')
check("max 3 kept, 4th dropped", (kept, [r for _, r in dropped]), (["A", "B", "C"], ["over the limit of 3"]))

print("== consent, names, forget phrase")
for a, want in [("sí", True), ("Si", True), ("claro", True), ("sí, por favor", True), ("vale", True),
                ("sí, recuérdame", True), ("sí, claro", True), ("hola", False),
                ("no", False), ("sí, pero no", False), ("quizás", False), ("", False), ("no sé", False)]:
    check(f"is_clear_yes({a!r})", is_clear_yes(a), want)
check("name: 'Me llamo Prueba'", name_slug("Me llamo Prueba"), "prueba")
check("name: 'José'", name_slug("José"), "jose")
check("name: empty", name_slug("  "), "")
check("forget: 'olvídame'", wants_forget("por favor olvídame"), True)
check("forget: 'BÓRRAME ya'", wants_forget("BÓRRAME ya"), True)
check("forget: unrelated", wants_forget("no me olvides el café"), False)

print("== database round trip (cam schema)")
tmp = Path(tempfile.mkdtemp())
mem = Memory(tmp)
name = "test_unit_mem"
mem.forget(name)  # clean slate if a previous run died
np.save(tmp / f"{name}.npy", np.zeros((2, 128), np.float32))
pid = mem.person_id(name, consent=True)
check("person created", isinstance(pid, int), True)
check("person_id is stable", mem.person_id(name), pid)
when = datetime.now(timezone.utc)
check("add 2 facts", mem.add_facts(name, ["Le gusta el té", "Vive en Bogotá"], when), 2)
check("add 0 facts", mem.add_facts(name, [], when), 0)
check("recent facts, newest first (same time -> by id)", mem.recent_facts(name), ["Vive en Bogotá", "Le gusta el té"])
removed = mem.forget(name)
check("forget removes row, facts and gallery file", removed, {"people": 1, "facts": 2, "gallery_file": True})
check("gallery file gone", (tmp / f"{name}.npy").exists(), False)
check("no facts left", mem.recent_facts(name), [])
check("forget again is a no-op", mem.forget(name), {"people": 0, "facts": 0, "gallery_file": False})
tmp.rmdir()

print("ALL OK" if not FAIL else f"{len(FAIL)} FAILURES")
sys.exit(1 if FAIL else 0)
