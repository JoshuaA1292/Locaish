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


# --- the cast as meshes -------------------------------------------------------
#
# Actors are flat-shaded figure meshes, the same meshes the studio renders into
# its frames, so the twin and the frames agree on where a body is. The parent
# may send one per figure; without one the page builds its own.


def test_template_draws_figures_as_flat_shaded_meshes():
    js = _script(_template())
    # the page's own articulated figure: head, torso, limbs, posed by a walk swing
    assert "function simFigureMesh(h, swing)" in js
    for part in ("sphere(", "cyl(", "box("):
        assert part in js, f"the figure builder has no {part}"
    assert "function simWalkSwing" in js and "function simActorGait" in js
    assert "25 * Math.PI / 180" in js, "the legs swing to 25 degrees"
    # the shader: Lambert from a fixed light with 0.45 ambient, on the face normal
    assert "const VS_FIG" in js and "const FS_FIG" in js
    assert "0.45 + 0.55 * lam" in js
    assert "cross(dFdx(vPos), dFdy(vPos))" in js
    assert "progFig = program(VS_FIG, FS_FIG" in js
    # back faces culled, feet biased off the floor, a silhouette pass
    assert "gl.cullFace(gl.BACK)" in js and "gl.cullFace(gl.FRONT)" in js
    assert "gl.polygonOffset(-1.0, -2.0)" in js
    assert "FLOOR_T - ORIGIN[2] + 0.02" in js, "the page's figures stand 2 cm above the floor"
    assert "gl.drawElements(gl.TRIANGLES, f.mcount, gl.UNSIGNED_INT, 0)" in js


def test_template_accepts_a_mesh_in_the_figures_message():
    js = _script(_template())
    assert 'm.type === "locaish:figures"' in js
    # `{name, colour, mesh: {vertices, faces}}` -- twin metres, flat triples
    assert "f.mesh" in js and "mesh.vertices" in js and "mesh.faces" in js
    assert "ORIGIN[0]" in js and "meshSignedVolume(pos, faces) < 0" in js, "an inward-wound mesh is flipped, not culled away"
    # the point path survives for a figure without a mesh
    assert "radius_m" in js and "gl.drawArrays(gl.POINTS, 0, f.count)" in js
    # the label follows the highest vertex
    assert "lbl.p[2] = top + 0.12" in js


def test_template_keeps_a_near_actor_whole_through_the_lens():
    js = _script(_template())
    assert "sim.camView ? 0.05 : Math.max(0.01, cam.dist * 0.002)" in js


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_figure_mesh_is_closed_and_wound_outwards(tmp_path):
    """Run the figure builder itself under node: a few hundred triangles, every
    edge shared by exactly two faces, positive volume, standing on the floor."""
    js = _script(_template())
    start = js.index("function MeshBuilder()")
    end = js.index("// Area-weighted vertex normals")
    helpers = "\n".join(
        line for line in js.splitlines()
        if line.startswith(("function sub3(", "function len3(", "function norm3(", "function cross3("))
    )
    # cross3 spans lines in the template; pull it by brace matching
    c0 = js.index("function cross3(")
    c1 = js.index("}", js.index("];", c0)) + 1
    helpers = "\n".join(l for l in helpers.splitlines() if not l.startswith("function cross3(")) + "\n" + js[c0:c1]
    vol = js[js.index("function meshSignedVolume"):js.index("// A figure slot holds")]
    prog = helpers + "\n" + js[start:end] + "\n" + vol + r"""
function stats(swing) {
  const m = simFigureMesh(1.75, swing);
  const nv = m.vertices.length / 3, nf = m.faces.length / 3;
  const edges = new Map();
  for (let i = 0; i < m.faces.length; i += 3) {
    const t = [m.faces[i], m.faces[i + 1], m.faces[i + 2]];
    for (let k = 0; k < 3; k++) {
      const a = t[k], b = t[(k + 1) % 3];
      const key = a < b ? a + ":" + b : b + ":" + a;
      edges.set(key, (edges.get(key) || 0) + (a < b ? 1 : -1));
    }
  }
  let open = 0;
  for (const v of edges.values()) if (v !== 0) open++;
  let zmin = Infinity, zmax = -Infinity;
  for (let i = 2; i < m.vertices.length; i += 3) { zmin = Math.min(zmin, m.vertices[i]); zmax = Math.max(zmax, m.vertices[i]); }
  return { nv, nf, open, vol: meshSignedVolume(m.vertices, m.faces) / 6, zmin, zmax };
}
console.log(JSON.stringify({ standing: stats(0), striding: stats(0.4) }));
"""
    path = tmp_path / "figure.js"
    path.write_text(prog, encoding="utf-8")
    proc = subprocess.run(["node", str(path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    import json
    both = json.loads(proc.stdout.strip().splitlines()[-1])
    for pose, got in both.items():
        assert 300 <= got["nf"] <= 1200, (pose, got)
        assert got["open"] == 0, f"{pose}: {got['open']} edges are not shared by two opposite-wound faces"
        assert 0.03 < got["vol"] < 0.2, (pose, got)   # a person is a few tens of litres
        assert abs(got["zmin"]) < 0.02, (pose, got)   # a sole on the floor, standing or mid-stride
    assert 1.74 < both["standing"]["zmax"] <= 1.76, both["standing"]
    assert both["striding"]["zmax"] < both["standing"]["zmax"], "mid-stride the body drops"
