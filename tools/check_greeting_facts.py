#!/usr/bin/env python3
"""Does the greeting use a remembered fact? N greetings from the running model (default
the 4B on :8093) with one fixed fact in the prompt, built exactly as greeter.person_conversation
builds it. Prints each greeting and the fact words it uses.
  .venv/bin/python tools/check_greeting_facts.py [--n 5] [--temperature 0]"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pc"))
import greeter  # noqa: E402
from memory import fact_words_used  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--name", default="Prueba")
ap.add_argument("--fact", default="Le gusta el ciclismo de montaña")
a = ap.parse_args()
url = os.environ.get("LLM_URL", "http://127.0.0.1:8093/v1")

system = greeter.SYSTEM.format(name=a.name) + greeter.FACTS_HINT.format(name=a.name, facts=f"- {a.fact}")
hits = 0
for i in range(a.n):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": f"({a.name} acaba de llegar.)"}]
    text, _, _, _ = greeter.llm(url, messages, stream_to_terminal=False, temperature=a.temperature)
    used = fact_words_used(text, [a.fact])
    hits += bool(used)
    print(f"{i + 1}: {'USES' if used else 'no  '} {used} | {text.strip()}")
print(f"{hits}/{a.n} greetings use the fact (temperature {a.temperature:g}; fact: {a.fact!r})")
