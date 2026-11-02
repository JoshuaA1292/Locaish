"""Stage a finished scan for the demo, step by step, through the studio.

    python docs/demo/shoot.py JOB [--studio http://127.0.0.1:8765] [--skip labels,cast,...]

Steps, each printed with what it produced so the recordings can use the ids:
  labels     Muse Spark names what is in the room (the labels panel)
  cast       Maya 1.68 m and Theo 1.84 m with looks and wardrobe
  rehearsal  the dorm scene broken down, placed, rehearsed; every frame finished
  piece      the first chair/sofa/desk label lifted to a set piece, then to 3D
  sim        a camera move from the rehearsal, ready for the operator
  world      the room on 14 November at 16:40, sixty views finished and baked
Everything costs real Meta API and Modal time: about $2 and 25 minutes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

SCENE = """INT. DORM ROOM - LATE AFTERNOON

Theo is at the desk with his back to the door, typing, headphones on.
Maya comes in and stops just inside the door.

MAYA: You said you'd call.

THEO: I said I'd try.

Maya crosses to the window and looks out into the sun. Theo finally turns in the middle of the room toward the window.

THEO: Are you staying?

MAYA: That depends on you.
"""

CAST = [
    {"name": "Maya", "height_m": 1.68, "description": "19, short brown hair, freckles",
     "wardrobe": "dark green knitted jumper, black jeans, white trainers"},
    {"name": "Theo", "height_m": 1.84, "description": "20, dark curly hair, tall",
     "wardrobe": "grey t-shirt, white trousers, barefoot"},
]


def call(base: str, path: str, body=None, method: str | None = None, timeout: float = 900.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method or ("POST" if body is not None else "GET"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def follow(base: str, path: str, timeout: float = 1800.0) -> dict:
    """Read an event stream until its terminal event; print the stages."""
    req = urllib.request.Request(base + path, headers={"Accept": "text/event-stream"})
    last = {}
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            m = json.loads(line[5:])
            last = m
            if m.get("kind") in ("stage", "note", "error"):
                print(f"  {time.time() - t0:6.0f}s  {m.get('text', '')[:110]}", flush=True)
            if m.get("kind") in ("done", "error"):
                break
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--studio", default="http://127.0.0.1:8765")
    ap.add_argument("--skip", default="")
    ap.add_argument("--date", default="2026-11-14")
    ap.add_argument("--hour", type=float, default=16.67)
    ap.add_argument("--dressing", default="the room as a student bedroom on a November afternoon, bedside lamp on, duvet thrown back")
    a = ap.parse_args()
    base, job = a.studio.rstrip("/"), a.job
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    out: dict = {"job": job}
    status = call(base, f"/stage-status/{job}")
    print("stage status:", status)

    if "labels" not in skip:
        print("== labels: Muse Spark is naming the room")
        r = call(base, f"/labels/{job}/detect", {})
        labels = r.get("labels") or (call(base, f"/labels/{job}").get("labels") or [])
        print(f"   {len(labels)} labels:", ", ".join(sorted({str(l.get('label')) for l in labels})[:20]))
        out["labels"] = len(labels)

    if "cast" not in skip:
        print("== cast")
        have = {c.get("name") for c in (call(base, f"/cast/{job}").get("cast") or [])}
        for actor in CAST:
            if actor["name"] in have:
                continue
            r = call(base, f"/cast/{job}", actor)
            print("   added", r.get("name"), r.get("height_m"))
        out["cast"] = [c["name"] for c in CAST]

    if "rehearsal" not in skip:
        print("== rehearsal: Muse Spark breaks the scene down")
        r = call(base, f"/rehearse/{job}", {"scene": SCENE, "shoot_date": a.date, "shoot_start": "15:30", "setup_minutes": 25})
        rid = r["rehearsal_id"]
        follow(base, f"/rehearsal-events/{job}/{rid}")
        print("== finishing every frame with Muse Image")
        f = call(base, f"/rehearsal/{job}/{rid}/finish-all", {"force": False})
        follow(base, f["events"])
        out["rehearsal"] = rid
        print("   rehearsal", rid)

    if "piece" not in skip:
        print("== set piece: lift a labelled thing, then lift it to 3D")
        labels = call(base, f"/labels/{job}").get("labels") or []
        pick = next((l for kind in ("chair", "sofa", "armchair", "desk", "table", "bed")
                     for l in labels if kind in str(l.get("label", "")).lower()), None)
        if pick is None:
            print("   no furniture label to lift")
        else:
            r = call(base, f"/setpieces/{job}/lift", {"label_id": pick["id"]})
            piece = r["piece"]
            print("   lifted", piece["label"], piece["size"])
            if status.get("sam3d"):
                r = call(base, f"/setpieces/{job}/{piece['id']}/lift3d", {})
                print("   SAM 3D:", r.get("gaussians"), "gaussians" if r.get("gaussians") else r.get("error"))
            out["piece"] = piece["id"]

    if "sim" not in skip:
        print("== camera move from the rehearsal")
        rid = out.get("rehearsal")
        if not rid:
            runs = call(base, f"/rehearsals/{job}").get("rehearsals") or []
            rid = runs[-1]["rehearsal_id"] if runs else None
        r = call(base, f"/sims/{job}", {"title": "Handheld push-in on Theo at the desk"})
        sid = r["timeline"]["id"] if isinstance(r.get("timeline"), dict) and r["timeline"].get("id") else r.get("id")
        if rid and sid:
            try:
                call(base, f"/sim/{job}/{sid}/from-rehearsal", {"rehearsal_id": rid})
            except Exception as e:  # noqa: BLE001
                print("   from-rehearsal:", e)
        out["sim"] = sid
        print("   sim", sid)

    if "world" not in skip:
        print("== the world: sixty views finished by Muse Image, baked onto the scan")
        r = call(base, f"/world/{job}", {"date": a.date, "hour": a.hour, "dressing": a.dressing})
        follow(base, r["events"])
        out["world"] = r["run"]
        print("   world", r["run"])

    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
