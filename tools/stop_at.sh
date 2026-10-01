#!/usr/bin/env bash
# Stop a process at a fixed clock time (the PC is turned off ~19:00).
#   tools/stop_at.sh HH:MM PID PATTERN
# At HH:MM today, if PID's command line still matches the extended regex
# PATTERN, send SIGTERM, wait up to 60 s, then SIGKILL. The pattern check
# means a reused PID is never signalled. Exits early if the process is gone.
#
# Usage (detached, logs to ~/logs/stop_at.log):
#   setsid nohup tools/stop_at.sh 18:55 <PID> 'llama-server.*--port 8093' >> ~/logs/stop_at.log 2>&1 < /dev/null &
set -u
AT=${1:?HH:MM}
PID=${2:?PID}
PATTERN=${3:?PATTERN}

log() { echo "[$(date +%Y-%m-%dT%H:%M:%S)] stop_at: $*"; }
alive() { tr '\0' ' ' 2>/dev/null < "/proc/$PID/cmdline" | grep -qE -- "$PATTERN"; }

target=$(date -d "today $AT" +%s) || exit 1
[ "$target" -gt "$(date +%s)" ] || { log "$AT has already passed; nothing scheduled"; exit 1; }
alive || { log "PID $PID does not match '$PATTERN'; nothing to stop"; exit 1; }
log "will stop PID $PID ('$PATTERN') at $AT"

while left=$(( target - $(date +%s) )); [ "$left" -gt 0 ]; do
    alive || { log "PID $PID exited before $AT; nothing to stop"; exit 0; }
    sleep $(( left < 20 ? left : 20 ))
done

alive || { log "PID $PID exited just before $AT"; exit 0; }
log "$AT reached: SIGTERM to PID $PID"
kill -TERM "$PID" 2>/dev/null
for _ in $(seq 60); do alive || { log "PID $PID stopped"; exit 0; }; sleep 1; done
log "PID $PID still alive after 60 s; SIGKILL"
kill -KILL "$PID" 2>/dev/null
