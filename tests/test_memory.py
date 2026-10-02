#!/usr/bin/env python3
"""Offline tests for pc/memory.py: fact parsing and filters, consent/name
parsing, and add/read/consent/forget against the cam schema in de_postgres
(uses throwaway people 'test_unit_mem*' and a temporary gallery; all removed)."""
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline import guard  # noqa: E402

guard("tests/test_memory.py")  # no model, no network (Postgres via libpq still works)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pc"))
from memory import (Memory, ground_facts, is_clear_yes, name_slug, parse_facts, third_person,  # noqa: E402
                    ungrounded, wants_forget)

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

print("== grounding: names and content words must come from what was said")
said = ["estoy viendo videos de la AYN odin 3"]
check("17:45 live case: hallucinated title dropped",
      ground_facts(["Vea videos de Assassin's Creed Odyssey"], said),
      ([], [("Vea videos de Assassin's Creed Odyssey", "no está en lo que dijo: Assassin, Creed, Odyssey")]))
check("the real product name is kept (case-insensitive)", ungrounded("Ve videos de la AYN Odin 3", said), [])
check("a wrong model number is not inflection", ungrounded("Ve videos de la AYN Odin 2", said), ["2"])
check("accents and inflection: Esperando el partido de la selección",
      ungrounded("Esperando el partido de la selección", ["estoy esperando el partido de la seleccion colombia"]), [])
check("verb ending trabajo -> trabaja", ungrounded("Trabaja como diseñadora gráfica",
                                                  ["Trabajo como diseñadora gráfica."]), [])
check("plural suculentas ~ suculenta", ungrounded("Colecciona suculentas", ["Colecciono una suculenta"]), [])
check("framing verb allowed, invented noun not", ungrounded("Tiene un labrador como mascota",
                                                            ["Me gustan los perros, tengo un labrador."]), ["mascota"])
check("invented place dropped", ungrounded("Vive en Medellín", ["Me mudé a Manizales hace dos meses."]), ["Medellín"])
check("short words and stopwords ignored", ungrounded("Le gusta el té", ["Prefiero el té verde"]), [])

print("== third person: first-person facts are rewritten or dropped")
check("run 2 case 22: 'Voy al Valle de Cocora…' dropped",
      third_person(["Voy al Valle de Cocora los fines de semana"]),
      ([], [("Voy al Valle de Cocora los fines de semana", "en primera persona")]))
check("'Me gusta el café cargado' rewritten", third_person(["Me gusta el café cargado"]),
      (["Le gusta el café cargado"], []))
check("'Mi serie favorita es…' rewritten", third_person(["Mi serie favorita es de ciencia ficción"])[0],
      ["Su serie favorita es de ciencia ficción"])
check("first-person verb dropped", third_person(["Colecciono plantas suculentas"])[0], [])
check("pronoun inside dropped", third_person(["Juega fútbol con mis amigos"])[0], [])
check("third person kept as is", third_person(["Juega Silksong en la Steam Deck", "Es carpintero",
                                               "Se mudó a Manizales", "Viajó a Japón"])[0],
      ["Juega Silksong en la Steam Deck", "Es carpintero", "Se mudó a Manizales", "Viajó a Japón"])

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
check("record_consent on a missing person -> None, no row created",
      (mem.record_consent("test_unit_nobody"), mem.forget("test_unit_nobody")["people"]), (None, 0))
mem.forget(name + "_c")
pid_c = mem.person_id(name + "_c")  # enrolled by hand: consent_at NULL
consent_at = mem.record_consent(name + "_c")
check("record_consent sets consent_at to a recent time",
      consent_at is not None and abs((datetime.now(timezone.utc) - consent_at).total_seconds()) < 60, True)
check("record_consent keeps the same person", mem.person_id(name + "_c"), pid_c)
mem.forget(name + "_c")
removed = mem.forget(name)
check("forget removes row, facts and gallery file", removed, {"people": 1, "facts": 2, "gallery_file": True})
check("gallery file gone", (tmp / f"{name}.npy").exists(), False)
check("no facts left", mem.recent_facts(name), [])
check("forget again is a no-op", mem.forget(name), {"people": 0, "facts": 0, "gallery_file": False})
tmp.rmdir()

print("ALL OK" if not FAIL else f"{len(FAIL)} FAILURES")
sys.exit(1 if FAIL else 0)
