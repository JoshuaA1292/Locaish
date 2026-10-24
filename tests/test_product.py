"""The product layer: projects, packages, share links, the gate, the sun on
the floor. Pure parts run on the synthetic clean room; the HTTP parts run a
real studio server on a free port with that room restored into it."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from datetime import date
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from locaish.film import coverage as cov
from locaish.film import sweep as sweepmod
from locaish.product import fit, package, sunlight
from locaish.product.store import Gate, Store
from locaish.types import Georeference

LA = dict(latitude=34.05, longitude=-118.24)


# ---------------------------------------------------------------------------
# store and gate
# ---------------------------------------------------------------------------


def test_projects_round_trip_and_update(tmp_path):
    st = Store(tmp_path)
    p = st.create_project("Acme spot", client="Acme", shoot_date="2026-11-14", locations=["a", "b", "a"])
    assert p["locations"] == ["a", "b"], "duplicates collapse"
    assert st.project(p["id"])["title"] == "Acme spot"
    st.update_project(p["id"], {"add_location": "c", "remove_location": "a", "shoot_date": "not a date"})
    q = st.project(p["id"])
    assert q["locations"] == ["b", "c"]
    assert q["shoot_date"] == "", "a shoot date that does not parse is no date"
    assert [x["id"] for x in st.projects()] == [p["id"]]
    assert st.delete_project(p["id"]) and st.project(p["id"]) is None


def test_details_validate_the_numeric_fields(tmp_path):
    st = Store(tmp_path)
    d = st.update_details(tmp_path, {"address": " 1 Main St ", "heading_deg": "372", "latitude": "95", "power": ""})
    assert d["address"] == "1 Main St"
    assert d["heading_deg"] == 12.0, "headings wrap"
    assert "latitude" not in d, "an impossible latitude is dropped, not stored"
    assert "power" not in d


def test_share_tokens_are_reused_resolved_and_revoked(tmp_path):
    st = Store(tmp_path)
    a = st.create_share("package", "job1")
    b = st.create_share("package", "job1")
    assert a["token"] == b["token"], "one live link per target"
    assert st.resolve_share(a["token"], count=True)["opens"] == 1
    assert st.revoke_share(a["token"])
    assert st.resolve_share(a["token"]) is None
    c = st.create_share("package", "job1")
    assert c["token"] != a["token"], "a revoked link is never resurrected"
    with pytest.raises(ValueError):
        st.create_share("everything", "x")


def test_gate_tokens_expire_and_do_not_leak_the_passcode():
    g = Gate("hunter2")
    tok = g.token(now=1_000_000)
    assert "hunter2" not in tok
    assert g.valid(tok, now=1_000_000 + 10)
    assert not g.valid(tok, now=1_000_000 + Gate.TTL_S + 1)
    assert not g.valid(tok[:-1] + "0", now=1_000_000)
    assert g.check("hunter2") and not g.check("hunter3")
    assert Gate(None).valid(None), "no passcode means no gate"
    assert Gate.from_cookie_header("a=1; locaish_auth=xyz; b=2") == "xyz"


# ---------------------------------------------------------------------------
# the sun on the floor
# ---------------------------------------------------------------------------


@pytest.fixture
def georeferenced(clean):
    result, _fx = clean
    twin = result.twin
    old = twin.georeference
    # +X points west (heading 270): a window on the +X wall faces west.
    twin.georeference = Georeference(heading_deg=270.0, heading_source="user", **LA)
    try:
        yield twin
    finally:
        twin.georeference = old


def test_sun_direction_is_consistent_with_the_ephemeris(georeferenced):
    s = sunlight.sun_at(georeferenced, "2026-06-21", 12.0)
    assert s["elevation_deg"] > 70, "midsummer solar noon in LA is nearly overhead"
    v = s["toward_sun_twin"]
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert v[2] > 0.9
    night = sunlight.sun_patches(georeferenced, "2026-06-21", 0.0)
    assert night["patches"] == [] and "horizon" in night["note"]


def test_window_corners_span_the_opening(georeferenced):
    op = next(o for o in georeferenced.structure.openings if o.kind != "door")
    c = sunlight.window_corners(op)
    assert c.shape == (4, 3)
    assert np.isclose(c[:, 2].max() - c[:, 2].min(), op.height, atol=1e-6)
    assert np.isclose(np.linalg.norm(c[1] - c[0]), op.width, atol=1e-6)


def test_patches_land_inside_the_room_on_the_floor(georeferenced):
    twin = georeferenced
    found = False
    for h in (8.0, 10.0, 12.0, 14.0, 16.0, 17.0):
        p = sunlight.sun_patches(twin, "2026-09-09", h)
        for patch in p["patches"]:
            found = True
            op = twin.structure.openings[patch["window"]]
            n = np.asarray(op.normal)[:2]
            centre = np.mean(patch["floor_xy"], axis=0)
            # The patch lies on the room side of the window, further in the
            # lower the sun.
            assert np.dot(centre - np.asarray(op.center)[:2], n) > 0
            assert patch["throw_m"] > 0
    assert found, "a west window in LA gets afternoon sun on some hour"


def test_floor_plan_without_a_plan_draws_room_and_overlay(georeferenced):
    patches = sunlight.sun_patches(georeferenced, "2026-09-09", 16.0)
    svg = cov.floor_plan_svg(georeferenced, None,
                             overlay=lambda X, Y: sunlight.patches_svg(georeferenced, patches, X, Y),
                             caption="16:00")
    assert svg.startswith("<svg") and "clipPath" in svg and "16:00" in svg
    assert svg.count("#ffcf5a") == len(patches["patches"])
    assert "N (user)" in svg


# ---------------------------------------------------------------------------
# fit and package
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def local_source(clean):
    result, _fx = clean
    return cov.LocalSetups(sweepmod.sweep(result.twin))


def test_fit_scores_held_relaxed_and_missed(clean, local_source):
    result, _fx = clean
    shots = fit.shots_from_brief("1. Wide: MAYA at the counter\n2. Close-up MAYA\n3. Medium JON")
    assert len(shots) == 3
    row = fit.fit_location(result.twin, local_source, shots, shoot_date="2026-11-14")
    assert row["total"] == 3
    assert row["held"] + row["relaxed"] + row["missed"] == 3
    assert 0.0 <= row["score"] <= 1.0
    assert row["score"] == pytest.approx((row["held"] + 0.5 * row["relaxed"]) / 3, abs=1e-3)
    assert "sun" not in row, "no georeference, no sun claim"
    assert row["workable_floor_m2"] > 0
    assert "holds" in fit.summary_line(row)
    assert fit.summary_line({"total": 0}) == "no shot list to test against"


def test_rank_orders_by_score_then_floor():
    rows = [{"score": 0.5, "held": 1, "workable_floor_m2": 5},
            {"score": 0.8, "held": 2, "workable_floor_m2": 1},
            {"score": 0.5, "held": 1, "workable_floor_m2": 9}]
    out = fit.rank(rows)
    assert [r["rank"] for r in out] == [1, 2, 3]
    assert out[0]["score"] == 0.8 and out[1]["workable_floor_m2"] == 9


def test_trust_statement_bands_follow_scale_confidence(clean):
    result, _fx = clean
    twin = result.twin
    t = package.trust_statement(twin)
    assert t["band"] in ("measured", "metric", "approximate", "proportional", "unknown")
    assert t["line"]
    old = dict(twin.qa.metrics)
    try:
        twin.qa.metrics["scale_confidence"] = 0.5999999999999996
        assert package.trust_statement(twin)["band"] == "metric", "banded on the rounded value"
        twin.qa.metrics["scale_confidence"] = 0.2
        assert package.trust_statement(twin)["band"] == "proportional"
    finally:
        twin.qa.metrics.clear()
        twin.qa.metrics.update(old)


def test_package_builds_with_sun_plans_and_renders(georeferenced, tmp_path):
    pkg = package.build(georeferenced, job_id="j1", name="Test room", shoot_date="2026-11-14",
                        details={"address": "1 Main St", "power": "20 A"})
    assert pkg["shoot_date"] == "2026-11-14" and not pkg["shoot_date_is_today"]
    assert pkg["sun"] and pkg["sun"]["windows"]
    assert [f["label"] for f in pkg["floor_plans"]] == ["09:00", "12:00", "15:00"]
    assert pkg["details"] == {"address": "1 Main St", "power": "20 A"}
    html = package.render_html(pkg)
    assert "1 Main St" in html and "Light on 2026-11-14" in html and "<svg" in html
    assert "<script" not in html, "the package is a document: it carries no scripts"
    out = package.write(pkg, tmp_path)
    assert out.exists() and (tmp_path / "package.json").exists()


def test_package_without_georeference_says_so(clean):
    result, _fx = clean
    pkg = package.build(result.twin, job_id="j", name="n")
    assert pkg["sun"] is None
    assert len(pkg["floor_plans"]) == 1 and pkg["floor_plans"][0]["sun"] is None
    assert "no sun schedule" in package.render_html(pkg)


# ---------------------------------------------------------------------------
# the studio, over HTTP
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def studio_url(clean, tmp_path_factory):
    """A real server on a free port, with the clean room restored as a job."""
    import os

    from locaish import serve as servemod
    from locaish.viewer.build import render_html

    result, _fx = clean
    root = tmp_path_factory.mktemp("studio")
    job = root / "cleanroom"
    job.mkdir()
    twin = result.twin
    twin.save(job / "test-clean.twin")
    render_html(twin, job / "view.html", max_points=20_000)
    os.environ["LOCAISH_PASSCODE"] = "pilot"
    try:
        studio = servemod.Studio(root, max_points=20_000)
    finally:
        os.environ.pop("LOCAISH_PASSCODE", None)
    handler = type("H", (servemod._Handler,), {"studio": studio})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", studio
    finally:
        httpd.shutdown()


def _call(url, *, method="GET", body=None, cookie=None, accept="application/json"):
    data = None
    headers = {"Accept": accept}
    if body is not None:
        data = json.dumps(body).encode() if isinstance(body, dict) else body
        headers["Content-Type"] = "application/json" if isinstance(body, dict) else "application/x-www-form-urlencoded"
    if cookie:
        headers["Cookie"] = cookie
    req = urllib.request.Request(url, data=data, method=method, headers=headers)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(req, timeout=60) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_gate_blocks_private_routes_and_lets_public_ones_through(studio_url):
    url, _ = studio_url
    assert _call(f"{url}/jobs")[0] == 401
    status, headers, _ = _call(f"{url}/", accept="text/html")
    assert status == 302 and headers["Location"].startswith("/login")
    assert _call(f"{url}/pricing")[0] == 200
    assert _call(f"{url}/capture")[0] == 200
    status, headers, _ = _call(f"{url}/login", method="POST", body=b"passcode=wrong&next=/")
    assert status == 200 and "Set-Cookie" not in headers
    status, headers, _ = _call(f"{url}/login", method="POST", body=b"passcode=pilot&next=/projects")
    assert status == 302 and headers["Location"] == "/projects"
    assert headers["Set-Cookie"].startswith("locaish_auth=")


@pytest.fixture(scope="module")
def cookie(studio_url):
    url, _ = studio_url
    _, headers, _ = _call(f"{url}/login", method="POST", body=b"passcode=pilot&next=/")
    return headers["Set-Cookie"].split(";")[0]


def test_details_override_the_heading_and_unlock_the_sun(studio_url, cookie):
    url, studio = studio_url
    status, _, body = _call(f"{url}/sun/cleanroom?date=2026-11-14", cookie=cookie)
    assert status == 400 and b"georeference" in body
    status, _, body = _call(f"{url}/details/cleanroom", method="POST", cookie=cookie,
                            body={"display_name": "Clean room", "heading_deg": 270, **LA, "power": "20 A"})
    assert status == 200 and json.loads(body)["display_name"] == "Clean room"
    status, _, body = _call(f"{url}/sun/cleanroom?date=2026-11-14", cookie=cookie)
    sun = json.loads(body)
    assert status == 200 and sun["windows"] and "heading_caveat" not in sun
    status, headers, body = _call(f"{url}/floorplan/cleanroom?date=2026-11-14&hour=15", cookie=cookie)
    assert status == 200 and b"<svg" in body
    status, _, body = _call(f"{url}/package/cleanroom.json?date=2026-11-14", cookie=cookie)
    pkg = json.loads(body)
    assert pkg["name"] == "Clean room" and pkg["heading"]["source"] == "user"
    assert pkg["details"]["power"] == "20 A"
    status, _, body = _call(f"{url}/package/cleanroom?date=2026-11-14", cookie=cookie)
    assert status == 200 and b"Clean room" in body and b"20 A" in body


def test_project_compare_and_share_link(studio_url, cookie):
    url, _ = studio_url
    brief = "1. Wide: MAYA at the counter\n2. Close-up MAYA\n3. Medium JON"
    status, _, body = _call(f"{url}/projects", method="POST", cookie=cookie,
                            body={"title": "Acme spot", "shoot_date": "2026-11-14", "brief": brief,
                                  "locations": ["cleanroom", "nope"]})
    proj = json.loads(body)
    assert status == 200 and proj["locations"] == ["cleanroom"], "unknown jobs are not added"
    pid = proj["id"]
    status, _, body = _call(f"{url}/project/{pid}/compare", method="POST", cookie=cookie)
    fitd = json.loads(body)
    assert status == 200 and len(fitd["rows"]) == 1 and len(fitd["shots"]) == 3
    row = fitd["rows"][0]
    assert row["rank"] == 1 and row["display_name"] == "Clean room" and row["sun"]["date"] == "2026-11-14"
    assert json.loads(_call(f"{url}/project/{pid}.json", cookie=cookie)[2])["fit"]["rows"][0]["job_id"] == "cleanroom"
    status, _, body = _call(f"{url}/project/{pid}", cookie=cookie, accept="text/html")
    assert status == 200 and b"Acme spot" in body

    status, _, body = _call(f"{url}/share", method="POST", cookie=cookie, body={"kind": "project", "target": pid})
    tok = json.loads(body)["token"]
    # The director has no cookie.
    status, _, body = _call(f"{url}/s/{tok}", accept="text/html")
    assert status == 200 and b"Acme spot" in body and b"Clean room" in body
    assert _call(f"{url}/s/{tok}/package/cleanroom?date=2026-11-14")[0] == 200
    assert _call(f"{url}/s/{tok}/view/cleanroom")[0] == 200
    assert _call(f"{url}/s/{tok}/package/other")[0] == 404, "a token reaches only its own rooms"
    assert _call(f"{url}/s/{tok}/floorplan/cleanroom?hour=12")[0] == 200
    assert _call(f"{url}/s/garbage", accept="text/html")[0] == 404
    assert _call(f"{url}/share/{tok}", method="DELETE", cookie=cookie)[0] == 200
    assert _call(f"{url}/s/{tok}", accept="text/html")[0] == 404


def test_lead_is_recorded_without_a_passcode(studio_url):
    url, studio = studio_url
    assert _call(f"{url}/lead", method="POST", body={"name": "x"})[0] == 400
    status, _, body = _call(f"{url}/lead", method="POST", body={"name": "Dana", "email": "d@x.com", "plan": "pilot"})
    assert status == 200
    assert studio.store.leads()[-1]["email"] == "d@x.com"
