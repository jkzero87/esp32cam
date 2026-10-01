"""People and facts for the greeter (phase 4), stored in Postgres schema cam.

Connection settings come from ~/esp32cam/.env (PGHOST, PGPORT, PGUSER,
PGPASSWORD, PGDATABASE; gitignored). Face embeddings stay in
<gallery>/<name>.npy and are linked by cam.people.name. No transcripts are
stored: only up to 3 short facts per conversation, extracted by the model from
what the person said, filtered for health, money, passwords/IDs and third
parties.
"""
import json
import re
import unicodedata
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
MAX_FACTS = 3
GREETING_FACTS = 5

EXTRACT_SYSTEM = (
    "Extraes datos para una memoria personal. Te doy frases que dijo {name}. "
    "Devuelve como máximo 3 hechos breves y duraderos sobre {name} (gustos, aficiones, planes, "
    "trabajo, costumbres), en español, cada uno en una frase corta en tercera persona. "
    "Usa solo lo que dijo {name}; no inventes. "
    "NO incluyas nada sobre salud, dinero, contraseñas, claves o números de identificación, "
    "ni nada sobre otras personas. "
    'Responde SOLO con JSON: {{"facts": ["...", "..."]}}. Si no hay hechos duraderos, {{"facts": []}}.\n'
    "Ejemplo. Frases: «Los sábados juego fútbol con mis amigos. Me duele la espalda. Estoy aprendiendo "
    'japonés.» Respuesta: {{"facts": ["Juega al fútbol los sábados", "Está aprendiendo japonés"]}}'
)

# Second line of defence: drop facts that touch excluded topics even if the model kept them.
EXCLUDED = {
    "health": r"salud|m[eé]dic|enferm|hospital|doctor|medicamento|pastilla|dolor|diabet|c[aá]ncer|"
              r"depresi|ansiedad|embaraz|terapia|operaci[oó]n|alerg",
    "money": r"dinero|sueldo|salario|euros?\b|d[oó]lares?|pesos\b|deuda|banco|tarjeta|cr[eé]dito|"
             r"pr[eé]stamo|hipoteca|inversi[oó]n|\$|€",
    "secrets/IDs": r"contrase[nñ]a|clave|\bpin\b|password|\bdni\b|pasaporte|c[eé]dula|"
                   r"n[uú]mero de (cuenta|tarjeta|identificaci[oó]n)|\d{6,}",
    "third parties": r"\bmis? (madre|padre|mam[aá]|pap[aá]|herman[oa]s?|hij[oa]s?|pareja|novi[oa]|"
                     r"espos[oa]|marido|mujer|amig[oa]s?|jefe|jefa|compa[nñ]er[oa]s?|vecin[oa]s?|"
                     r"abuel[oa]s?|t[ií][oa]s?|prim[oa]s?|suegr[oa])\b|\bsus? (madre|padre|herman|hij|pareja|"
                     r"novi|espos|amig|jefe|compa[nñ]er)",
}
EXCLUDED_RE = {k: re.compile(v, re.I) for k, v in EXCLUDED.items()}

YES = {"si", "sí", "claro", "vale", "dale", "ok", "okay", "por supuesto", "sí por favor", "si por favor",
       "sí claro", "si claro", "venga", "de acuerdo"}
FORGET_RE = re.compile(r"\b(olv[ií]dame|b[oó]rrame)\b", re.I)


def norm(text):
    """Lower-case, no accents, no punctuation, single spaces."""
    t = unicodedata.normalize("NFD", (text or "").lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


def is_clear_yes(answer):
    """Only an explicit yes counts as consent; anything else (or nothing) is no."""
    a = norm(answer)
    if "no" in a.split():
        return False
    return a in {norm(y) for y in YES} or (a.startswith("si ") and len(a.split()) <= 4)


def wants_forget(text):
    return bool(FORGET_RE.search(text or ""))


def name_slug(raw):
    """A spoken name -> gallery/DB name: lower-case ASCII letters, digits, _ and -."""
    s = norm(raw)
    for prefix in ("me llamo ", "soy ", "mi nombre es "):
        if s.startswith(prefix):
            s = s[len(prefix):]
    s = re.sub(r"[^a-z0-9_-]", "", s.split()[0]) if s.split() else ""
    return s[:40]


def excluded_topic(fact):
    return next((k for k, rx in EXCLUDED_RE.items() if rx.search(fact)), None)


def parse_facts(raw):
    """Model output -> (kept facts, dropped [(fact, reason)]). Accepts the JSON
    object, a bare JSON list, or JSON inside ```fences```; anything else -> nothing."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    data = None
    for candidate in (text, *(m.group(0) for m in re.finditer(r"\{.*\}|\[.*\]", text, re.S))):
        try:
            data = json.loads(candidate)
            break
        except (json.JSONDecodeError, TypeError):
            continue
    items = data.get("facts", []) if isinstance(data, dict) else data if isinstance(data, list) else []
    kept, dropped, seen = [], [], set()
    for item in items:
        if not isinstance(item, str):
            continue
        fact = " ".join(item.split()).strip(" -•")
        if not fact or norm(fact) in seen:
            continue
        if len(fact) > 200:
            dropped.append((fact, "too long"))
            continue
        if reason := excluded_topic(fact):
            dropped.append((fact, reason))
            continue
        seen.add(norm(fact))
        kept.append(fact)
    return kept[:MAX_FACTS], dropped + [(f, "over the limit of 3") for f in kept[MAX_FACTS:]]


def extraction_messages(name, user_lines):
    said = "\n".join(f"- {line}" for line in user_lines if line.strip())
    return [{"role": "system", "content": EXTRACT_SYSTEM.format(name=name)},
            {"role": "user", "content": f"Lo que dijo {name}:\n{said}"}]


def load_env(path=ROOT / ".env"):
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


class Memory:
    def __init__(self, gallery_dir, env=None):
        e = env or load_env()
        self.conninfo = dict(host=e["PGHOST"], port=e["PGPORT"], user=e["PGUSER"],
                             password=e["PGPASSWORD"], dbname=e["PGDATABASE"])
        self.gallery_dir = Path(gallery_dir)

    def _conn(self):
        return psycopg.connect(**self.conninfo, autocommit=True)

    def person_id(self, name, consent=False):
        """id of NAME, creating the row if needed (consent_at = now() if CONSENT)."""
        with self._conn() as c:
            row = c.execute("SELECT id FROM cam.people WHERE name = %s", (name,)).fetchone()
            if row:
                if consent:
                    c.execute("UPDATE cam.people SET consent_at = now() WHERE id = %s", (row[0],))
                return row[0]
            return c.execute("INSERT INTO cam.people (name, consent_at) VALUES (%s, CASE WHEN %s THEN now() END) "
                             "RETURNING id", (name, consent)).fetchone()[0]

    def recent_facts(self, name, n=GREETING_FACTS):
        with self._conn() as c:
            return [r[0] for r in c.execute(
                "SELECT f.fact FROM cam.facts f JOIN cam.people p ON p.id = f.person_id "
                "WHERE p.name = %s ORDER BY f.created_at DESC, f.id DESC LIMIT %s", (name, n))]

    def add_facts(self, name, facts, conversation_at):
        if not facts:
            return 0
        pid = self.person_id(name)
        with self._conn() as c:
            with c.cursor() as cur:
                cur.executemany("INSERT INTO cam.facts (person_id, fact, source_conversation_at) VALUES (%s, %s, %s)",
                                [(pid, f, conversation_at) for f in facts])
        return len(facts)

    def forget(self, name):
        """Delete NAME's row (facts cascade) and gallery file. Returns what was removed."""
        with self._conn() as c:
            n_facts = c.execute("SELECT count(*) FROM cam.facts f JOIN cam.people p ON p.id = f.person_id "
                                "WHERE p.name = %s", (name,)).fetchone()[0]
            n_people = c.execute("DELETE FROM cam.people WHERE name = %s", (name,)).rowcount
        path = self.gallery_dir / f"{name}.npy"
        had_file = path.exists()
        if had_file:
            path.unlink()
        return {"people": n_people, "facts": n_facts, "gallery_file": had_file}
