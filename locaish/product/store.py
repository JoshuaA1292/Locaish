"""Projects, location details, share links and leads: small JSON on disk.

A project is the unit of work a production company has: one job, one brief,
one shoot date, a handful of candidate rooms. A location's *details* are the
things no scanner records and every location package needs -- the address,
who holds the keys, where the trucks park, how many amps are on the wall --
plus the compass heading and coordinates that make the sun schedule true.

Each record is a JSON file. Writes go through a temp file and an atomic
rename so a crash mid-write never leaves a half-written project behind, and
a lock serialises writers inside one process (the studio is threaded).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The practical fields of a location package, in the order a scout fills
# them in. ``heading_deg``, ``latitude`` and ``longitude`` are the ones that
# change what the engine computes; the rest are carried into the package.
DETAIL_FIELDS = (
    "display_name", "address", "contact", "power", "parking", "access",
    "sound_notes", "restrictions", "fee", "notes",
    "heading_deg", "latitude", "longitude",
)

_NUMERIC = {"heading_deg", "latitude", "longitude"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def _read(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


class Store:
    """Everything the product layer persists, rooted beside the jobs."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.dir = self.root / "_product"
        self._lock = threading.RLock()

    # -- projects -----------------------------------------------------------
    @property
    def projects_dir(self) -> Path:
        return self.dir / "projects"

    def _project_path(self, pid: str) -> Path:
        safe = "".join(c for c in pid if c.isalnum() or c in "-_")
        if not safe:
            raise ValueError("bad project id")
        return self.projects_dir / f"{safe}.json"

    def create_project(self, title: str, *, client: str = "", brief: str = "",
                       shoot_date: str = "", locations: list[str] | None = None) -> dict:
        pid = uuid.uuid4().hex[:10]
        proj = {
            "id": pid,
            "title": title.strip() or "Untitled job",
            "client": client.strip(),
            "brief": brief,
            "shoot_date": _valid_date(shoot_date),
            "locations": list(dict.fromkeys(locations or [])),
            "notes": "",
            "created_at": _now(),
            "updated_at": _now(),
            # The last comparison run, keyed by what it was run on, so the
            # page can show the table without re-planning six rooms.
            "fit": None,
        }
        with self._lock:
            _atomic_write(self._project_path(pid), proj)
        return proj

    def project(self, pid: str) -> dict | None:
        try:
            return _read(self._project_path(pid), None)
        except ValueError:
            return None

    def projects(self) -> list[dict]:
        if not self.projects_dir.is_dir():
            return []
        out = []
        for p in self.projects_dir.glob("*.json"):
            d = _read(p, None)
            if d:
                out.append(d)
        out.sort(key=lambda d: d.get("updated_at") or "", reverse=True)
        return out

    def update_project(self, pid: str, changes: dict) -> dict | None:
        with self._lock:
            proj = self.project(pid)
            if proj is None:
                return None
            for key in ("title", "client", "brief", "notes"):
                if key in changes and changes[key] is not None:
                    proj[key] = str(changes[key])
            if "shoot_date" in changes:
                proj["shoot_date"] = _valid_date(str(changes.get("shoot_date") or ""))
            if "locations" in changes and isinstance(changes["locations"], list):
                proj["locations"] = list(dict.fromkeys(str(x) for x in changes["locations"]))
            if "add_location" in changes and changes["add_location"]:
                jid = str(changes["add_location"])
                if jid not in proj["locations"]:
                    proj["locations"].append(jid)
            if "remove_location" in changes and changes["remove_location"]:
                proj["locations"] = [x for x in proj["locations"] if x != str(changes["remove_location"])]
            if "fit" in changes:
                proj["fit"] = changes["fit"]
            proj["updated_at"] = _now()
            _atomic_write(self._project_path(pid), proj)
            return proj

    def delete_project(self, pid: str) -> bool:
        with self._lock:
            p = self._project_path(pid)
            if not p.exists():
                return False
            p.unlink()
            return True

    # -- location details ---------------------------------------------------
    @staticmethod
    def details_path(workdir: Path) -> Path:
        return Path(workdir) / "details.json"

    def details(self, workdir: Path) -> dict:
        d = _read(self.details_path(workdir), {})
        return d if isinstance(d, dict) else {}

    def update_details(self, workdir: Path, changes: dict) -> dict:
        with self._lock:
            d = self.details(workdir)
            for key in DETAIL_FIELDS:
                if key not in changes:
                    continue
                v = changes[key]
                if v is None or (isinstance(v, str) and not v.strip()):
                    d.pop(key, None)
                    continue
                if key in _NUMERIC:
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        continue
                    if key == "heading_deg":
                        v = v % 360.0
                    elif key == "latitude" and not -90 <= v <= 90:
                        continue
                    elif key == "longitude" and not -180 <= v <= 180:
                        continue
                    d[key] = v
                else:
                    d[key] = str(v).strip()[:4000]
            d["updated_at"] = _now()
            _atomic_write(self.details_path(workdir), d)
            return d

    # -- share links --------------------------------------------------------
    @property
    def shares_path(self) -> Path:
        return self.dir / "shares.json"

    def shares(self) -> dict[str, dict]:
        d = _read(self.shares_path, {})
        return d if isinstance(d, dict) else {}

    def create_share(self, kind: str, target: str, *, label: str = "") -> dict:
        """A read-only token for a package, a project or a viewer.

        Tokens are random, unguessable and revocable; they are the whole
        access model for people outside the company, which is the right
        model for a link that gets forwarded from a producer to a director
        to a client without anyone logging in.
        """
        if kind not in ("package", "project", "view"):
            raise ValueError("share kind must be package, project or view")
        with self._lock:
            shares = self.shares()
            # One live link per target: sharing twice should not mint a
            # second URL that someone later has to remember to revoke too.
            for tok, s in shares.items():
                if s.get("kind") == kind and s.get("target") == target and not s.get("revoked"):
                    return {"token": tok, **s}
            tok = secrets.token_urlsafe(12)
            rec = {"kind": kind, "target": target, "label": label, "created_at": _now(),
                   "opens": 0, "revoked": False}
            shares[tok] = rec
            _atomic_write(self.shares_path, shares)
            return {"token": tok, **rec}

    def resolve_share(self, token: str, *, count: bool = False) -> dict | None:
        with self._lock:
            shares = self.shares()
            rec = shares.get(token)
            if not rec or rec.get("revoked"):
                return None
            if count:
                rec["opens"] = int(rec.get("opens") or 0) + 1
                rec["last_opened_at"] = _now()
                _atomic_write(self.shares_path, shares)
            return {"token": token, **rec}

    def revoke_share(self, token: str) -> bool:
        with self._lock:
            shares = self.shares()
            if token not in shares:
                return False
            shares[token]["revoked"] = True
            _atomic_write(self.shares_path, shares)
            return True

    def shares_for(self, target: str) -> list[dict]:
        return [{"token": t, **s} for t, s in self.shares().items()
                if s.get("target") == target and not s.get("revoked")]

    # -- leads --------------------------------------------------------------
    def add_lead(self, lead: dict) -> dict:
        """A pilot request from the pricing page: appended, never lost."""
        rec = {
            "id": uuid.uuid4().hex[:8],
            "at": _now(),
            "name": str(lead.get("name") or "")[:200],
            "email": str(lead.get("email") or "")[:200],
            "company": str(lead.get("company") or "")[:200],
            "role": str(lead.get("role") or "")[:100],
            "plan": str(lead.get("plan") or "")[:40],
            "message": str(lead.get("message") or "")[:2000],
        }
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / "leads.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        return rec

    def leads(self) -> list[dict]:
        p = self.dir / "leads.jsonl"
        if not p.exists():
            return []
        out = []
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out


def _valid_date(s: str) -> str:
    """An ISO date or nothing: a shoot date that does not parse is no date."""
    s = (s or "").strip()
    if not s:
        return ""
    try:
        datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return ""
    return s


# ---------------------------------------------------------------------------
# the passcode gate
# ---------------------------------------------------------------------------


class Gate:
    """One passcode per hosted studio.

    A design partner gets their own deployment and one passcode for the
    team; share links bypass the gate by construction. The cookie carries an
    HMAC of the passcode rather than the passcode, and expires, so a leaked
    cookie is not a leaked password and does not live forever.
    """

    COOKIE = "locaish_auth"
    TTL_S = 30 * 24 * 3600

    def __init__(self, passcode: str | None):
        self.passcode = (passcode or "").strip() or None
        self._key = (self.passcode or "").encode()

    @property
    def enabled(self) -> bool:
        return self.passcode is not None

    def check(self, candidate: str) -> bool:
        return self.enabled and hmac.compare_digest(candidate.strip(), self.passcode)

    def token(self, now: float | None = None) -> str:
        exp = int((now or time.time()) + self.TTL_S)
        sig = hmac.new(self._key, str(exp).encode(), hashlib.sha256).hexdigest()[:32]
        return f"{exp}.{sig}"

    def valid(self, token: str | None, now: float | None = None) -> bool:
        if not self.enabled:
            return True
        if not token or "." not in token:
            return False
        exp_s, sig = token.split(".", 1)
        try:
            exp = int(exp_s)
        except ValueError:
            return False
        if exp < (now or time.time()):
            return False
        want = hmac.new(self._key, exp_s.encode(), hashlib.sha256).hexdigest()[:32]
        return hmac.compare_digest(want, sig)

    @classmethod
    def from_cookie_header(cls, header: str | None) -> str | None:
        if not header:
            return None
        for part in header.split(";"):
            k, _, v = part.strip().partition("=")
            if k == cls.COOKIE:
                return v
        return None
