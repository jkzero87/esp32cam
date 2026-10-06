#!/usr/bin/env python3
"""Does the greeting invent what it "sees"? The model never gets the image, so any place,
room, object, clothing, appearance or activity it names is invented unless it is in a
remembered fact. Runs greetings on the running model (default the 4B on :8093), built as
greeter.person_conversation builds them: N without facts and M with one fact (5 facts
cycled), and flags words from a scene lexicon that are not in the fact. Every greeting is
printed so the flags can be checked by eye.
  .venv/bin/python tools/check_greeting_scene.py [--no-facts 20] [--with-fact 10] [--temperature 0.7]
       [--prompt-from GIT_REV]   # take SYSTEM/FACTS_HINT from pc/greeter.py at that revision"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pc"))
import greeter  # noqa: E402
from memory import fact_words_used, norm  # noqa: E402

FACTS = ["Le gusta el ciclismo de montaña", "Esperando el partido de la selección",
         "Trabaja como profesor de matemáticas", "Tiene un perro llamado Toby",
         "Está aprendiendo a tocar la guitarra"]
# Places/rooms, furniture/objects, clothing/appearance, "what you are doing" and seeing words.
SCENE = re.compile(r"\b(sala|salon|cuarto|habitacion|cocina|oficina|escritorio|mesa|silla|sofa|casa|hogar|"
                   r"puerta|ventana|jardin|parque|calle|lugar|rincon|espacio|ambiente|luz|"
                   r"cafe|cafecito|taza|libro|libros|computador|computadora|ordenador|pantalla|teclado|"
                   r"telefono|celular|movil|planta|plantas|cuadro|decoracion|"
                   r"ropa|camisa|camiseta|chaqueta|abrigo|gorra|sombrero|gafas|lentes|zapatos|"
                   r"pelo|cabello|peinado|sonrisa|cara|ojos|aspecto|guapo|guapa|elegante|cansado|cansada|"
                   r"te ves|se te ve|luces|te veo|estas haciendo|vienes de|llegando de)\b")


def prompt_constants(rev):
    if not rev:
        return greeter.SYSTEM, greeter.FACTS_HINT
    src = subprocess.run(["git", "show", f"{rev}:pc/greeter.py"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    ns = {}
    for const in ("SYSTEM", "FACTS_HINT"):
        m = re.search(rf"^{const} = \(.*?\)\n", src, re.S | re.M)
        exec(m.group(0), ns)
    return ns["SYSTEM"], ns["FACTS_HINT"]


ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--no-facts", type=int, default=20)
ap.add_argument("--with-fact", type=int, default=10)
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--name", default="Prueba")
ap.add_argument("--prompt-from", default="", help="git revision whose SYSTEM/FACTS_HINT to use (default: current)")
ap.add_argument("--out", type=Path)
a = ap.parse_args()
url = os.environ.get("LLM_URL", "http://127.0.0.1:8093/v1")
system_tpl, hint_tpl = prompt_constants(a.prompt_from)

rows = []
plan = [None] * a.no_facts + [FACTS[i % len(FACTS)] for i in range(a.with_fact)]
for i, fact in enumerate(plan, 1):
    system = system_tpl.format(name=a.name)
    if fact:
        system += hint_tpl.format(name=a.name, facts=f"- {fact}")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": f"({a.name} acaba de llegar.)"}]
    text, first, _, _ = greeter.llm(url, messages, stream_to_terminal=False, temperature=a.temperature)
    allowed = set(norm(fact or "").split())
    scene = sorted({m.group(0) for m in SCENE.finditer(norm(text))} - allowed)
    used = fact_words_used(text, [fact]) if fact else []
    rows.append(dict(i=i, fact=fact, text=text.strip(), scene_words=scene, fact_words_used=used,
                     first_word_s=round(first, 3) if first is not None else None))
    print(f"{i:2d} {'SCENE ' + str(scene) if scene else 'ok   '} {'| fact ' + str(used) if fact else ''} | {text.strip()}")
nf = [r for r in rows if not r["fact"]]
wf = [r for r in rows if r["fact"]]
print(f"\nprompt from: {a.prompt_from or 'current'}; temperature {a.temperature:g}")
print(f"flagged invented-scene greetings: {sum(bool(r['scene_words']) for r in rows)}/{len(rows)} "
      f"(no facts {sum(bool(r['scene_words']) for r in nf)}/{len(nf)}, with fact {sum(bool(r['scene_words']) for r in wf)}/{len(wf)})")
print(f"with-fact greetings that use the fact: {sum(bool(r['fact_words_used']) for r in wf)}/{len(wf)}")
firsts = sorted(r["first_word_s"] for r in rows if r["first_word_s"] is not None)
if firsts:
    print(f"first word (s, from the request; streamed): median {statistics.median(firsts):.3f}, "
          f"min {firsts[0]:.3f}, max {firsts[-1]:.3f}, n {len(firsts)}")
if a.out:
    a.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
