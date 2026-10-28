"""The viewer's Simulate mode, as far as Python can see it.

The mode is JavaScript inside one self-contained HTML template, so what can be
asserted from here is the contract: that the template carries the mode, that
it handles every message the studio sends and sends every message the studio
expects, that the script still parses, that it reaches for nothing on the
network, and that a viewer built from a real ingest still embeds and opens.
What the mode *does* is checked by eye in a browser (see the ``?sim=1`` demo
flag), which no Python test can replace.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from locaish.viewer import build as viewer_build

TEMPLATE = Path(viewer_build.__file__).with_name("template.html")

# parent → viewer
INBOUND = [
    "locaish:sim",
    "locaish:sim-mode",
    "locaish:sim-play",
    "locaish:sim-time",
    "locaish:figures",
    "locaish:sim-rig",
]
# viewer → parent
OUTBOUND = [
    "locaish:sim-edit",
    "locaish:sim-time",
    "locaish:sim-mode",
    "locaish:sim-select",
]
# the timeline JSON the engine and the studio share; the viewer must spell
# every field the same way, because it edits the object in place
CONTRACT_FIELDS = [
    "duration_s", "aperture_f", "rail_height_m", "gear_key", "base_height_m", "arm_m",
    "look_at", "focal_mm", "roll_deg", "ease", "height_m", "facing_deg", "interp",
    "t_start", "t_end", "severity", "shot_size", "distance_m", "speed_mps", "near_m", "far_m",
]


def _template() -> str:
    return TEMPLATE.read_text(encoding="utf-8")


def _script(html: str) -> str:
    m = re.search(r"<script>(.*)</script>", html, re.S)
    assert m, "the viewer carries exactly one inline script"
    return m.group(1)


def test_template_carries_the_simulate_mode():
    t = _template()
    assert 'id="btn-sim"' in t and 'Simulate<span class="k">6</span>' in t
    assert 'id="simpanel"' in t and 'id="simbar"' in t
    assert 'k === "6"' in t, "the 6 key toggles the mode"
    for key in ('k === " "', 'k === "k"', 'k === ","', 'k === "."', 'k === "home"', 'k === "end"', '"delete"'):
        assert key in t, f"the simulator's {key} shortcut"
    assert "[?&#]sim=1" in t, "the ?sim=1 demo flag"
    assert "SIM_RIGS" in t
    for rig in ("tripod", "slider", "dolly", "jib", "handheld", "steadicam", "crane", "drone"):
        assert f'"{rig}"' in t


def test_template_handles_every_message_in_the_contract():
    js = _script(_template())
    for kind in INBOUND:
        assert f'm.type === "{kind}"' in js, f"viewer does not handle {kind}"
    for kind in OUTBOUND:
        assert f'type: "{kind}"' in js, f"viewer never sends {kind}"
    # edits are debounced, time is throttled: the numbers the studio relies on
    assert "setTimeout(simPostEditNow, 250)" in js
    assert "sim.lastTimePost < 100" in js


def test_template_spells_the_contract_fields():
    js = _script(_template())
    for field in CONTRACT_FIELDS:
        assert field in js, f"timeline/evaluation field {field!r} missing from the viewer"


def test_template_reaches_for_nothing_on_the_network():
    t = _template()
    assert "<link" not in t
    assert 'src="http' not in t and "src='http" not in t
    assert "import(" not in t and "importScripts" not in t
    assert "cdn." not in t.lower()


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_template_script_parses(tmp_path):
    js = _script(_template()).replace(viewer_build._PAYLOAD_TOKEN, "{}")
    path = tmp_path / "template.js"
    path.write_text(js, encoding="utf-8")
    proc = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_template_declares_each_name_once():
    js = _script(_template())
    names = re.findall(r"^(?:const|let|function)\s+([A-Za-z_$][\w$]*)", js, re.M)
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"top-level names declared twice: {dupes}"


def test_built_viewer_still_embeds_and_carries_the_simulator(clean, tmp_path):
    result, _fx = clean
    out = viewer_build.render_html(result.twin, tmp_path / "view.html", max_points=20_000)
    html = out.read_text(encoding="utf-8")
    assert viewer_build._PAYLOAD_TOKEN not in html and viewer_build._TITLE_TOKEN not in html
    assert html.count("<script>") == 1 and html.count("</script>") == 1
    assert 'id="btn-sim"' in html and 'id="simbar"' in html
    # the floor the simulator stands its actors on comes from the payload
    assert '"floor_z":' in html
    assert "[?&#]sim=1" in html
