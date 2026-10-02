"""Keep offline test runs offline: no model server, no network from Python.

guard() fails fast unless LLM_URL is unset or "stub", then makes every
Python-level connection (urllib, sockets) raise. Postgres is unaffected: psycopg
connects through libpq, not Python sockets, and the unit tests need it.
"""
import os
import socket
import sys
import urllib.request

STUB = "stub"


class NetworkBlocked(RuntimeError):
    pass


def _blocked(*args, **kwargs):
    raise NetworkBlocked("offline test run: network access is blocked (tests/offline.py)")


def guard(what):
    url = os.environ.get("LLM_URL")
    if url not in (None, "", STUB):
        sys.exit(f"{what}: refusing to run with LLM_URL={url!r}; offline runs must not reach a model. "
                 f"Unset LLM_URL or set LLM_URL={STUB}.")
    os.environ["LLM_URL"] = STUB
    urllib.request.urlopen = _blocked
    socket.socket.connect = _blocked
    socket.socket.connect_ex = _blocked
    socket.create_connection = _blocked
