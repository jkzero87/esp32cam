#!/usr/bin/env bash
# Live test of the unknown-person flow and "olvídame", on a throwaway gallery.
#   tools/live_test_unknown.sh
# Runs the greeter on data/gallery_live_test (gitignored under data/), so the
# real gallery in data/gallery is never read or written. In front of the camera:
#   1. "sí" -> give the name "prueba". When told, step out of view ~10 s and come back.
#   2. It greets you by name: tell it something durable ("me gusta el ciclismo de montaña"),
#      then end the conversation as with a person: say goodbye ("chao", "adiós", "nos vemos")
#      or just walk away (8 s out of view ends it). The window shows "Guardando..." and
#      then "Listo: recordé N cosa(s)".
#   3. Step out ~10 s and come back: the greeting should mention what you said.
#   4. Say "olvídame" -> "sí". The window shows "Olvidado".
# Quit the greeter with Ctrl+C; pc/status.py then shows what is left in the DB and in that
# gallery (after "olvídame" the test person should be gone from both). --log-greeting puts
# each greeting's text and the fact words it used in data/greeter_metrics.jsonl.
# Needs the camera (CAM_IP in .env) and the 4B model on 127.0.0.1:8093.
set -u
cd "$(dirname "$0")/.." || exit 1
GALLERY=data/gallery_live_test
mkdir -p "$GALLERY"

.venv/bin/python pc/greeter.py --preview --metrics --log-greeting --absent-minutes 0 --gallery-dir "$GALLERY"
echo
.venv/bin/python pc/status.py --gallery-dir "$GALLERY"
