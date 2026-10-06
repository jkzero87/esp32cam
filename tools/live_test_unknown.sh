#!/usr/bin/env bash
# Live test of the unknown-person flow and "olvídame", on a throwaway gallery.
#   tools/live_test_unknown.sh
# Runs the greeter on data/gallery_live_test (gitignored under data/), so the
# real gallery in data/gallery is never read or written. In front of the
# camera: "sí" -> give a name -> talk -> step out of view >5 s and back (with
# --absent-minutes 0 that greets again) -> "olvídame" -> "sí". Quit the greeter
# with Ctrl+C; pc/status.py then shows what is left in the DB and in that
# gallery (after "olvídame" the test person should be gone from both).
# Needs the camera (CAM_IP in .env) and the 4B model on 127.0.0.1:8093.
set -u
cd "$(dirname "$0")/.." || exit 1
GALLERY=data/gallery_live_test
mkdir -p "$GALLERY"

.venv/bin/python pc/greeter.py --preview --metrics --absent-minutes 0 --gallery-dir "$GALLERY"
echo
.venv/bin/python pc/status.py --gallery-dir "$GALLERY"
