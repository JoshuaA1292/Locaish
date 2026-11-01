"""The studio page as one visual system.

The page is a product, not a dashboard: black and white, one hairline, one
warm tone that only ever marks a render, and a fixed order of sections that
leads with the room as it will be on the day. These tests hold the template
to that system so a restyle cannot quietly bring back a gradient or drop
an element the stage script reaches for by id.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from locaish import serve
from locaish.product import pages
from locaish.viewer import build as viewer_build

STAGE_JS = Path(serve.__file__).parent / "viewer" / "stage.js"
TEMPLATE = Path(viewer_build.__file__).with_name("template.html")


def _css(page: str) -> str:
    m = re.search(r"<style>(.*?)</style>", page, re.S)
    assert m, "the page carries one style block"
    return m.group(1)


def _markup(page: str) -> str:
    """Everything that is not the page's own script."""
    return re.sub(r"<script>.*?</script>", "", page, flags=re.S)


# ---------------------------------------------------------------------------
# the tokens
# ---------------------------------------------------------------------------


def test_page_defines_the_system_tokens():
    css = _css(serve.PAGE)
    root = re.search(r":root\s*\{(.*?)\}", css, re.S).group(1)
    for token, value in {
        "--bg": "#000000", "--surface": "#0a0a0a", "--line": "#1f1f1f", "--ink": "#ffffff",
        "--dim": "#9a9a9a", "--mute": "#5a5a5a", "--accent": "#ffffff", "--accent-ink": "#000000",
        "--gap": "64px", "--pad": "24px", "--r": "12px",
    }.items():
        assert re.search(rf"{re.escape(token)}:\s*{re.escape(value)}\s*;", root), f"{token} is {value}"
    assert "--render:" in root, "the one warm tone, for renders"
    assert re.search(r"--sans:\s*-apple-system, \"Helvetica Neue\", Inter, sans-serif", root)
    assert re.search(r"--mono:\s*ui-monospace", root)


def test_page_has_no_decoration():
    page = serve.PAGE
    for banned in ("linear-gradient", "radial-gradient", "box-shadow", "dashed", "backdrop-filter", "text-shadow"):
        assert banned not in page, f"the page uses {banned}"
    assert "linearGradient" not in page, "the mark is flat"
    css = _css(page)
    radii = [float(r) for r in re.findall(r"border-radius:\s*(\d+(?:\.\d+)?)px", css)]
    assert radii and max(radii) <= 12, "12 px is the largest radius"
    assert "999px" not in css


def test_headings_are_one_size_each():
    css = _css(serve.PAGE)
    assert re.search(r"\.sechead h2 \{[^}]*font-size: 22px;[^}]*font-weight: 600;", css)
    assert re.search(r"\.panelbox h4, \.ptitle \{[^}]*font-size: 13px;[^}]*letter-spacing: \.08em;[^}]*text-transform: uppercase;[^}]*color: var\(--dim\);", css)
    assert re.search(r"section \{ margin-top: var\(--gap\); \}", css)
    assert re.search(r"\.panelbox \{[^}]*padding: var\(--pad\);", css)


def test_primary_is_white_and_the_render_label_is_warm():
    css = _css(serve.PAGE)
    primary = re.search(r"\.primary, #st_run[^{]*\{([^}]*)\}", css).group(1)
    assert "background: var(--ink)" in primary and "color: var(--bg)" in primary
    secondary = re.search(r"\.lensbtn, #send[^{]*\{([^}]*)\}", css).group(1)
    assert "background: transparent" in secondary and "border: 1px solid var(--ink)" in secondary
    assert re.search(r"\.st_tag \{[^}]*color: var\(--render\)", css)
    assert re.search(r"\.st_stamp \{[^}]*color: var\(--render\)", css)
    assert re.search(r"\.st_strip img\.active \{[^}]*border-color: var\(--ink\)", css)
    assert re.search(r"\.st_framerow\.has_fin \{ grid-template-columns: minmax\(0, 1fr\) 300px; \}", css), "the finished frame leads, the measured one small beside it"


# ---------------------------------------------------------------------------
# the order of the page
# ---------------------------------------------------------------------------


def test_sections_come_in_the_product_order():
    markup = _markup(serve.PAGE)
    order = ['<div id="hero"', '<section id="world"', '<section id="stage"', '<section id="sim"', '<section id="contents"',
             '<details id="more"', '<section id="pkg"', '<section id="coverage"', '<section id="chat"']
    positions = [markup.index(x) for x in order]
    assert positions == sorted(positions), "hero, the world, the scene, the camera, the room, then the folded tools"
    # the scouting tools are folded away, the chat last inside them
    more = markup[markup.index('<details id="more"'):]
    assert more.index('id="pkg"') < more.index('id="coverage"') < more.index('id="chat"') < more.rindex("</details>")
    assert "Walk into it as it will be on the day" in markup and "The scene in this room" in markup
    assert "Simulate the camera" in markup and "What is in the room" in markup


def test_copy_is_one_plain_sentence_or_two_per_section():
    markup = _markup(serve.PAGE)
    for p in re.findall(r'<div class="sub"><h2>[^<]*</h2>\s*<p>(.*?)</p>', markup, re.S):
        text = re.sub(r"<[^>]+>", "", p)
        assert len(text) < 200, f"section copy runs long: {text[:60]}…"
        assert text.count(". ") <= 1, f"more than two sentences: {text[:60]}…"


# ---------------------------------------------------------------------------
# every id the scripts reach for exists
# ---------------------------------------------------------------------------


def _ids_used(js: str) -> set[str]:
    ids = set(re.findall(r"\$\('([A-Za-z0-9_\-]+)'\)", js))
    ids |= set(re.findall(r"getElementById\('([A-Za-z0-9_\-]+)'\)", js))
    return ids


def test_every_id_stage_js_reaches_for_exists():
    markup = _markup(serve.PAGE)
    present = set(re.findall(r'id="([A-Za-z0-9_\-]+)"', markup))
    js = STAGE_JS.read_text()
    # ids the script creates itself, inside strings it renders
    made = set(re.findall(r'id="([A-Za-z0-9_\-]+)"', js))
    missing = {i for i in _ids_used(js) if i not in present and i not in made}
    assert not missing, f"stage.js reaches for ids the page does not have: {sorted(missing)}"


def test_every_id_the_page_script_reaches_for_exists():
    page = serve.PAGE
    markup = _markup(page)
    present = set(re.findall(r'id="([A-Za-z0-9_\-]+)"', markup))
    js = re.search(r"<script>(.*?)</script>", page, re.S).group(1)
    made = set(re.findall(r'id="([A-Za-z0-9_\-]+)"', js))
    # ids built from a prefix and a key are checked by their prefix
    used = {i for i in _ids_used(js) if not i.startswith("d_")}
    missing = {i for i in used if i not in present and i not in made}
    assert not missing, f"the page script reaches for ids the page does not have: {sorted(missing)}"
    for k in ("display_name", "address", "heading_deg", "latitude", "longitude", "shoot_date", "notes"):
        assert f'id="d_{k}"' in markup


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_stage_js_parses():
    subprocess.run(["node", "--check", str(STAGE_JS)], check=True)


# ---------------------------------------------------------------------------
# the viewer chrome and the product pages share the system
# ---------------------------------------------------------------------------


def test_viewer_chrome_is_the_same_system():
    t = TEMPLATE.read_text()
    css = _css(t)
    for banned in ("linear-gradient", "box-shadow", "dashed"):
        assert banned not in css
    root = re.search(r":root\s*\{(.*?)\}", css, re.S).group(1)
    assert "--bg: #000000;" in root and "--ink: #ffffff;" in root and "--line: #1f1f1f;" in root
    assert "--render:" in root
    assert re.search(r"button\.tool \{[^}]*text-transform: uppercase;[^}]*color: var\(--ink-dim\);", css)
    assert re.search(r'button\.tool\[aria-pressed="true"\] \{ color: var\(--ink\);', css)
    assert re.search(r"#worldtag b \{ color: var\(--render\);", css), "the render label is the one warm thing"
    # the chrome's ids and the message contract are untouched
    for i in ("toolbar", "panel", "readout", "legend", "worldtag", "simpanel", "simbar", "letterbox", "btn-sim", "btn-walk"):
        assert f'id="{i}"' in t


def test_product_pages_are_the_same_system():
    for page in (pages.pricing_page(), pages.projects_page([], []), pages.login_page(), pages.capture_page()):
        css = _css(page)
        assert "--bg:#000000;" in css and "--ink:#ffffff;" in css and "--line:#1f1f1f;" in css
        for banned in ("linear-gradient", "box-shadow", "dashed", "999px"):
            assert banned not in page, f"{banned} in a product page"
        assert "linearGradient" not in page
        assert re.search(r"\.btn \{[^}]*background:var\(--ink\); color:var\(--bg\);", css)
    assert '<div class="price">$29' in pages.pricing_page()
