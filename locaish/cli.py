"""Command line entry point.

Ten verbs, and the shape of a Phase 1 session is meant to be obvious from
them: `demo` proves the pipeline works with no scan at all, `ingest` turns your
export -- or a video of the room -- into a twin, `inspect` tells you whether to
trust it, `view` lets you look at it, `measure` checks it against a tape
measure, `export` gets the geometry back out, `fixtures` lists the synthetic
rooms the accuracy claims are tested against, and `studio` puts a drop target in
front of `ingest` for when the input is a video and the audience is a person
rather than a shell.

The last two are Phase 2 and are the point of the whole exercise: `scout`
surveys a twin the way a location manager would survey the room, and `move`
walks a camera through it and reports where the shot breaks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path

import numpy as np


def _load_dotenv(path: Path = Path(".env")) -> None:
    """KEY=VALUE lines from a .env in the working directory, never overriding
    what the shell already set. Keys and hosts live here rather than in a
    shell profile, so the studio starts the same way from any terminal."""
    try:
        text = path.read_text()
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if getattr(args, "traceback", False):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="locaish",
        description="Scan any room, get a filming-ready metric digital twin.",
    )
    p.add_argument("--traceback", action="store_true", help="show the full traceback on error")
    sub = p.add_subparsers(dest="command")

    ing = sub.add_parser(
        "ingest",
        help="turn a scanner export -- or a video of the room -- into a .twin",
    )
    ing.add_argument(
        "source",
        type=Path,
        help="PLY / OBJ / GLB / GLTF / STL export, or a video (.mov/.mp4/...) "
        "of a sweep around the room",
    )
    ing.add_argument("-o", "--out", type=Path, help="output .twin path")
    ing.add_argument("--name", help="twin name (defaults to the source filename)")
    ing.add_argument("--lat", type=float, help="latitude of the room, decimal degrees")
    ing.add_argument("--lon", type=float, help="longitude of the room, decimal degrees")
    ing.add_argument(
        "--heading",
        type=float,
        help="true-north bearing of the twin's +X axis in degrees; without it "
        "every solar result downstream is an assumption, not a measurement",
    )
    ing.add_argument("--elevation", type=float, default=0.0, help="floor height above sea level, m")
    ing.add_argument("--unit", help="force the source unit (m/cm/mm/in/ft) instead of inferring it")
    ing.add_argument("--voxel", type=float, default=0.05, help="occupancy voxel size in m")
    ing.add_argument("--max-points", type=int, default=6_000_000)
    ing.add_argument("--force-mesh", action="store_true", help="re-derive the mesh even if the import had one")
    ing.add_argument("--no-mesh", action="store_true", help="skip mesh reconstruction entirely")
    ing.add_argument("--no-openings", action="store_true", help="skip window and door detection")
    ing.add_argument(
        "--no-fill",
        action="store_true",
        help="leave holes in the reconstructed surface instead of completing it "
        "against the volume the camera swept",
    )
    ing.add_argument(
        "--fill-radius",
        type=float,
        metavar="M",
        help="how wide a hole the completion may bridge, in metres of radius "
        "(default 0.45, so gaps up to about 0.9 m). Larger closes more and "
        "smooths furniture away with it",
    )
    ing.add_argument(
        "--no-fill-planes",
        action="store_true",
        help="do not resample points onto detected wall planes where the "
        "camera proved unbroken wall; leaves textureless walls as the holes "
        "the stereo matcher returned",
    )
    vid = ing.add_argument_group(
        "video",
        "only used when the source is footage rather than a scan file. A video "
        "gives the room's shape from parallax but never its size, so the scale "
        "is solved separately from the camera path and any doorway in shot, and "
        "reported with the error bar it deserves.",
    )
    vid.add_argument(
        "--fps",
        type=float,
        metavar="N",
        help="frames per second pulled from the sweep for matching (default 8). "
        "Classical feature matching chains correspondences frame to frame, so "
        "this is about keeping the chain unbroken, not about coverage",
    )
    vid.add_argument(
        "--scale-factor",
        type=float,
        help="metres per reconstruction unit, if you would rather supply the "
        "scale than have it inferred",
    )
    vid.add_argument("--start", type=float, help="ignore footage before this timestamp, seconds")
    vid.add_argument("--end", type=float, help="ignore footage after this timestamp, seconds")
    vid.add_argument(
        "--refresh",
        action="store_true",
        help="re-run the reconstruction instead of reusing the cached one",
    )
    vid.add_argument(
        "--recon-dir",
        type=Path,
        help="where to keep the chosen frames, raw cloud and manifest "
        "(default: alongside the video)",
    )
    ing.add_argument("--view", action="store_true", help="render and open the viewer when done")
    ing.add_argument("--quiet", action="store_true")
    ing.set_defaults(func=cmd_ingest)

    ins = sub.add_parser("inspect", help="print a twin's summary and QA report")
    ins.add_argument("twin", type=Path)
    ins.add_argument("--json", action="store_true", help="machine-readable output")
    ins.add_argument("--metrics", action="store_true", help="include the full metrics table")
    ins.set_defaults(func=cmd_inspect)

    vw = sub.add_parser("view", help="render a self-contained HTML viewer")
    vw.add_argument("twin", type=Path)
    vw.add_argument("-o", "--out", type=Path)
    vw.add_argument("--max-points", type=int, default=6_000_000)
    vw.add_argument("--no-open", action="store_true", help="write the file but do not open a browser")
    vw.set_defaults(func=cmd_view)

    ms = sub.add_parser(
        "measure",
        help="distance between two twin-space points, with an honest uncertainty",
    )
    ms.add_argument("twin", type=Path)
    ms.add_argument("--from", dest="a", required=True, metavar="X,Y,Z")
    ms.add_argument("--to", dest="b", required=True, metavar="X,Y,Z")
    ms.set_defaults(func=cmd_measure)

    ex = sub.add_parser("export", help="write the twin's geometry back out")
    ex.add_argument("twin", type=Path)
    ex.add_argument("-o", "--out", type=Path, required=True, help="output .ply / .obj / .glb")
    ex.add_argument("--points", action="store_true", help="export the point cloud rather than the mesh")
    ex.set_defaults(func=cmd_export)

    dm = sub.add_parser(
        "demo",
        help="run the whole pipeline on a synthetic room with known ground truth",
    )
    dm.add_argument("fixture", nargs="?", default="clean")
    dm.add_argument("-o", "--out", type=Path)
    dm.add_argument("--view", action="store_true")
    dm.set_defaults(func=cmd_demo)

    sc = sub.add_parser(
        "scout",
        help="survey a twin as a location: space, camera, grip and sound",
    )
    sc.add_argument("twin", type=Path)
    sc.add_argument("--json", action="store_true", help="machine-readable output")
    sc.add_argument("--cell", type=float, default=0.10, help="floor raster size in m")
    sc.set_defaults(func=cmd_scout)

    mv = sub.add_parser(
        "move",
        help="simulate a camera move through a twin and report where it breaks",
    )
    mv.add_argument("twin", type=Path)
    mv.add_argument("--from", dest="start", required=True, metavar="X,Y,Z",
                    help="where the camera starts")
    mv.add_argument("--to", dest="end", required=True, metavar="X,Y,Z",
                    help="where the camera ends")
    mv.add_argument("--subject", metavar="X,Y,Z",
                    help="where the actor stands (feet); omit for a move with no subject")
    mv.add_argument("--subject-to", metavar="X,Y,Z",
                    help="where the actor walks to, if they move")
    mv.add_argument("--height", type=float, default=1.75, help="actor's height in m")
    mv.add_argument("--lens", type=float, default=32.0, help="focal length in mm")
    mv.add_argument("--stop", type=float, default=2.8, help="aperture, f/T number")
    mv.add_argument("--sensor", default="super35", help="sensor format key")
    mv.add_argument("--gear", help="equipment key the camera rides on, e.g. super-peewee")
    mv.add_argument("--track", action="store_true",
                    help="check the floor is level enough to lay dolly track")
    mv.add_argument("--json", action="store_true")
    mv.set_defaults(func=cmd_move)

    st = sub.add_parser(
        "studio",
        help="the product: drop a room on a page, get a twin, ask the scout",
    )
    st.add_argument("--port", type=int, default=8765)
    st.add_argument("--host", help="bind address (default 127.0.0.1; 0.0.0.0 for hosting)")
    st.add_argument("--root", type=Path, default=Path("twins/studio"), help="where jobs land")
    st.add_argument("--max-points", type=int, default=6_000_000, dest="studio_max_points")
    st.add_argument("--no-open", action="store_true", help="do not open a browser")
    st.add_argument("--showcase", action="store_true",
                    help="read-only gallery of scanned locations; uploads are off")
    st.set_defaults(func=cmd_studio)

    sw = sub.add_parser(
        "sweep",
        help="sweep every camera setup a twin allows and load it into ClickHouse",
    )
    sw.add_argument("twin", type=Path)
    sw.add_argument("--no-load", action="store_true",
                    help="compute and summarise the sweep without touching ClickHouse")
    sw.add_argument("--json", action="store_true")
    sw.set_defaults(func=cmd_sweep)

    cv = sub.add_parser(
        "coverage",
        help="plan a scene's coverage against a twin: shot list, frames, camera plan",
    )
    cv.add_argument("twin", type=Path)
    cv.add_argument("--brief", type=Path, required=True,
                    help="a text file: the scene, or one shot per line")
    cv.add_argument("--title", default="")
    cv.add_argument("--out", type=Path, default=None,
                    help="plan directory (default: <twin dir>/plans/<id>)")
    cv.add_argument("--agent", action="store_true",
                    help="use the Gemini workflow (breakdown, placement loop, frame review)")
    cv.add_argument("--no-render", action="store_true", help="skip rendering the frames")
    cv.set_defaults(func=cmd_coverage)

    fx = sub.add_parser("fixtures", help="list the synthetic rooms used to validate accuracy")
    fx.set_defaults(func=cmd_fixtures)

    return p


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_ingest(args) -> int:
    from .scan.ingest import IngestOptions, ingest

    if not args.source.exists():
        raise FileNotFoundError(args.source)
    if (args.lat is None) != (args.lon is None):
        raise ValueError("--lat and --lon must be given together")

    opts = IngestOptions(
        name=args.name,
        max_points=args.max_points,
        voxel_xy=args.voxel,
        voxel_z=args.voxel,
        unit_hint=args.unit,
        latitude=args.lat,
        longitude=args.lon,
        heading_deg=args.heading,
        elevation_m=args.elevation,
        skip_mesh_reconstruction=args.no_mesh,
        force_mesh=args.force_mesh,
        skip_openings=args.no_openings,
        fill_holes=not args.no_fill,
        fill_radius_m=args.fill_radius,
        fill_planes=not args.no_fill_planes,
        progress=None if args.quiet else _progress,
        video_fps=args.fps,
        video_scale_factor=args.scale_factor,
        video_workdir=args.recon_dir,
        video_start_s=args.start,
        video_end_s=args.end,
        video_refresh=args.refresh,
    )
    result = ingest(args.source, opts)
    twin = result.twin

    out = args.out or Path("twins") / f"{twin.name}.twin"
    saved = twin.save(out)

    if not args.quiet:
        print()
        _print_summary(twin)
        print(_qa_text(twin))
        if result.warnings:
            print("notes:")
            for w in result.warnings:
                print(f"  - {w}")
        slow = sorted(result.timings.items(), key=lambda kv: -kv[1])[:4]
        timing = ", ".join(f"{k} {v:.1f}s" for k, v in slow if k != "total")
        print(f"\ningested in {result.timings.get('total', 0):.1f}s ({timing})")
    print(f"wrote {saved}")

    if args.view:
        _view(twin, None, open_browser=True)
    return 0 if twin.qa.verdict != "fail" else 2


def cmd_scout(args) -> int:
    from .film import report as reportmod
    from .types import Twin

    twin = Twin.load(args.twin)
    built = reportmod.build(twin, cell=args.cell)
    if args.json:
        print(json.dumps(built.to_dict(), indent=2, default=_jsonable))
    else:
        print(reportmod.render_text(built))
    return 0


def cmd_move(args) -> int:
    from .film import equipment as equipmod
    from .film import moves as movesmod
    from .film import space as spacemod
    from .types import Twin

    twin = Twin.load(args.twin)
    maps = spacemod.floor_maps(twin)
    occ = spacemod.occupancy(twin)

    subject_path = None
    if args.subject:
        first = _xyz(args.subject)
        subject_path = (
            np.stack([first, _xyz(args.subject_to)]) if args.subject_to else first[None, :]
        )

    gear = equipmod.get(args.gear) if args.gear else None
    result = movesmod.simulate(
        maps,
        occ,
        movesmod.straight(_xyz(args.start), _xyz(args.end)),
        name="move",
        subject_path=subject_path,
        subject_height_m=args.height,
        focal_mm=args.lens,
        aperture_f=args.stop,
        sensor=args.sensor,
        gear=gear,
        on_track=args.track,
    )

    if args.json:
        print(json.dumps(result.summary(), indent=2, default=_jsonable))
        return 0 if result.feasible else 2

    print(_move_text(result, args))
    return 0 if result.feasible else 2


def _move_text(result, args) -> str:
    from .film import optics as opticsmod

    s = result.summary()
    lines = [
        f"MOVE  {s['length_m']} m on a {args.lens:g} mm"
        + (f", riding a {args.gear}" if args.gear else ""),
        f"  verdict: {'WORKS' if s['feasible'] else 'DOES NOT WORK'}",
    ]
    if s["shot_range"]:
        lines.append(f"  framing: {s['shot_range']}   subject {s['distance_m'][0]}"
                     f"-{s['distance_m'][1]} m away")
    if s["track_level_range_m"] is not None:
        lines.append(f"  floor under the run varies by "
                     f"{s['track_level_range_m'] * 1000:.0f} mm")
    if not s["feasible"]:
        lines.append(f"  breaks {s['fails_at_m']} m into the move:")
        for p in s["problems"]:
            lines.append(f"    - {p}")
    for note in s["notes"]:
        lines.append(f"  note: {note}")

    lines.append("")
    lines.append("  beat      dist   framing              in frame  sightline  clearance")
    step = max(1, len(result.beats) // 8)
    for b in result.beats[::step]:
        lines.append(
            f"  {b.t * result.length_m:5.2f} m  {b.distance_m:5.2f}  {b.shot:<20s} "
            f"{'yes' if b.subject_in_frame else 'NO ':<9s} "
            f"{'clear' if b.clear_sightline else 'BLOCKED':<10s} {b.clearance_m:.2f} m"
        )
    return "\n".join(lines)


def cmd_studio(args) -> int:
    from .serve import serve

    serve(
        args.root,
        port=args.port,
        host=args.host,
        max_points=args.studio_max_points,
        open_browser=not args.no_open,
        showcase=args.showcase,
    )
    return 0


def cmd_coverage(args) -> int:
    from . import warehouse
    from .film import coverage as cov
    from .film import sweep as sweepmod
    from .types import Twin

    twin = Twin.load(args.twin)
    brief = args.brief.read_text()
    if warehouse.configured() and warehouse.location_counts().get(twin.name):
        source = cov.ClickHouseSetups()
        _progress(f"setups from ClickHouse ({warehouse.connection_env()['CLICKHOUSE_HOST']})")
    else:
        _progress("setups from a local sweep (ClickHouse not configured or not loaded)")
        source = cov.LocalSetups(sweepmod.sweep(twin, progress=_progress))
    out = args.out or (args.twin.parent / "plans" / cov.new_plan_id())
    if args.agent:
        from .agent.coverage import plan_coverage

        def on_event(e):
            if e["kind"] == "call":
                _progress(f"{e.get('agent', '')} -> {e['tool']}({json.dumps(e.get('args', {}))[:120]})")
            elif e["kind"] == "result":
                _progress(f"  <- {e.get('summary', '')}")
            elif e["kind"] in ("stage", "agent", "note"):
                _progress(str(e.get("text", "")))

        plan = plan_coverage(twin, workdir=out, source=source, brief=brief, title=args.title,
                             on_event=on_event)
    else:
        shots = cov.parse_shot_lines(brief)
        plan = cov.plan(twin, shots, source, title=args.title, brief=brief, workdir=out,
                        render=not args.no_render, progress=_progress)
    if warehouse.configured():
        try:
            warehouse.load_plan(plan)
        except Exception as exc:  # noqa: BLE001
            _progress(f"could not record the plan in ClickHouse: {exc}")
    print(cov.render_text(plan))
    print(f"plan written to {out}")
    return 0


def cmd_sweep(args) -> int:
    from . import warehouse
    from .film import sweep as sweepmod
    from .types import Twin

    twin = Twin.load(args.twin)
    sw = sweepmod.sweep(twin, progress=None if args.json else _progress)
    summary = sw.summary()

    if not args.no_load:
        if warehouse.configured():
            summary["loaded_rows"] = warehouse.load_sweep(
                sw, progress=None if args.json else _progress
            )
            summary["clickhouse"] = warehouse.connection_env()["CLICKHOUSE_HOST"]
        else:
            summary["clickhouse"] = None
            if not args.json:
                print("CLICKHOUSE_HOST is not set; computed the sweep but loaded nothing")

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"{summary['rows']:,} setups over {summary['subject_marks']} subject marks "
              f"and {summary['camera_cells']} camera positions")
        if summary.get("loaded_rows"):
            print(f"loaded into ClickHouse at {summary['clickhouse']}")
    return 0


def cmd_inspect(args) -> int:
    from .types import Twin

    twin = Twin.load(args.twin)
    if args.json:
        payload = {
            "summary": twin.summary(),
            "provenance": twin.provenance,
            "qa": twin.qa.to_dict(),
            "structure": {
                "floor_z": twin.structure.floor_z,
                "ceiling_z": twin.structure.ceiling_z,
                "floor_area_m2": twin.structure.floor_area,
                "planes": len(twin.structure.planes),
                "openings": [o.to_dict() for o in twin.structure.openings],
            },
        }
        print(json.dumps(payload, indent=2, default=_jsonable))
        return 0

    _print_summary(twin)
    _print_openings(twin)
    print(_qa_text(twin, metrics=args.metrics))
    return 0 if twin.qa.verdict != "fail" else 2


def cmd_view(args) -> int:
    from .types import Twin

    twin = Twin.load(args.twin)
    _view(twin, args.out, open_browser=not args.no_open, max_points=args.max_points)
    return 0


def cmd_measure(args) -> int:
    from .scan.qa import verify_measurement
    from .types import Twin

    twin = Twin.load(args.twin)
    a, b = _xyz(args.a), _xyz(args.b)
    r = verify_measurement(twin, a, b)
    d, u = r["distance_m"], r["uncertainty_m"]
    print(f"{d:.3f} m  +/- {u*1000:.0f} mm")
    if not r.get("within_capture_bounds", True):
        print("  warning: at least one endpoint is outside the captured region, "
              "so this distance leans on geometry the scanner never stood near")
    return 0


def cmd_export(args) -> int:
    from .formats.ply import write_ply
    from .types import Twin

    twin = Twin.load(args.twin)
    suffix = args.out.suffix.lower()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".ply":
        write_ply(
            args.out,
            twin.points if args.points or twin.mesh is None else None,
            None if args.points else twin.mesh,
        )
    else:
        import trimesh

        if twin.mesh is None or args.points:
            raise ValueError(f"{suffix} export needs a mesh; use --points with .ply instead")
        tm = trimesh.Trimesh(
            vertices=twin.mesh.vertices,
            faces=twin.mesh.faces,
            vertex_colors=twin.mesh.vertex_colors,
            process=False,
        )
        tm.export(args.out)
    print(f"wrote {args.out}")
    return 0


def cmd_demo(args) -> int:
    """Run the real pipeline over a synthetic room and grade it against truth.

    This is the command that answers "does the thing work" without anyone
    needing to hold a phone, and it is graded rather than merely run -- it
    prints the recovered dimensions beside the exact ones and the error in
    millimetres, which is the only honest way to display a claim of 1:1.
    """
    import tempfile

    from . import fixtures
    from .formats.ply import write_ply
    from .scan.ingest import IngestOptions, ingest

    fx = fixtures.build(args.fixture)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / f"{args.fixture}.ply"
        write_ply(src, fx.points, None)
        result = ingest(
            src,
            IngestOptions(name=f"demo-{args.fixture}", progress=_progress, seed=0),
        )
    twin = result.twin

    print()
    _print_summary(twin)

    from .scan.dimensions import measure_room

    dims = measure_room(twin.points, twin.structure)
    truth = fx.truth
    got_long, got_short = dims.plan
    want_long, want_short = sorted([truth.width, truth.depth], reverse=True)
    print("\nrecovered vs ground truth")
    print(f"  {'quantity':<16}{'truth':>10}{'measured':>12}{'error':>12}")
    rows = [
        ("long side", want_long, got_long),
        ("short side", want_short, got_short),
        ("ceiling height", truth.height, twin.structure.ceiling_height),
        ("floor area", truth.floor_area, twin.structure.floor_area),
    ]
    for label, t, g in rows:
        if g is None:
            print(f"  {label:<16}{t:>10.3f}{'not found':>12}{'':>12}")
            continue
        err = (g - t) * 1000.0
        print(f"  {label:<16}{t:>10.3f}{g:>12.3f}{err:>10.0f} mm")
    print(f"  {'openings':<16}{len(truth.openings):>10}{len(twin.structure.openings):>12}")

    print(_qa_text(twin))

    out = args.out or Path("twins") / f"demo-{args.fixture}.twin"
    print(f"wrote {twin.save(out)}")
    if args.view:
        _view(twin, None, open_browser=True)
    return 0


def cmd_fixtures(args) -> int:
    from . import fixtures

    print("synthetic rooms with exact ground truth:\n")
    for name, entry in fixtures.catalogue().items():
        print(f"  {name:<14}{entry['breaks']}")
    print("\nrun one end to end with:  locaish demo <name>")
    return 0


# ---------------------------------------------------------------------------
# shared output helpers
# ---------------------------------------------------------------------------


def _progress(step: str) -> None:
    print(f"  {step} ...", flush=True)


def _print_summary(twin) -> None:
    from .scan.dimensions import measure_room

    s = twin.summary()
    d = measure_room(twin.points, twin.structure)
    print(f"{twin.name}")
    est = "" if d.x.method == d.y.method == "planes" else "  (estimated from extents)"
    print(f"  {d.x.length:.2f} x {d.y.length:.2f} x {d.z.length:.2f} m{est}")
    ch = s["ceiling_height_m"]
    print(f"  ceiling      {'unknown' if ch is None else f'{ch:.2f} m'}")
    print(f"  floor area   {s['floor_area_m2']:.1f} m2")
    print(f"  points       {s['points']:,}")
    print(f"  faces        {s['faces']:,}")
    if twin.capture_bounds is not None:
        print(f"  captured     {twin.capture_bounds.area:.1f} m2 ({twin.capture_bounds.source})")
    if twin.georeference is not None:
        g = twin.georeference
        print(f"  located      {g.latitude:.5f}, {g.longitude:.5f}  "
              f"+X bears {g.heading_deg:.0f} deg ({g.heading_source})")


def _print_openings(twin) -> None:
    if not twin.structure.openings:
        return
    print("  openings")
    for o in twin.structure.openings:
        print(f"    {o.kind:<8}{o.width:.2f} x {o.height:.2f} m   "
              f"sill {o.sill_height:.2f} m   confidence {o.confidence:.2f}")


def _qa_text(twin, *, metrics: bool = False) -> str:
    from .scan.qa import format_report

    text = format_report(twin.qa, color=sys.stdout.isatty())
    if metrics:
        rows = sorted(twin.qa.metrics.items())
        text += "\n\nmetrics\n" + "\n".join(f"  {k:<32}{v:>12.4f}" for k, v in rows)
    return "\n" + text


def _view(twin, out: Path | None, *, open_browser: bool, max_points: int = 900_000) -> Path:
    from .viewer.build import render_html

    out = out or Path("twins") / f"{twin.name}.html"
    path = render_html(twin, out, max_points=max_points)
    size = path.stat().st_size / 1e6
    print(f"wrote {path} ({size:.1f} MB)")
    if open_browser:
        webbrowser.open(path.resolve().as_uri())
    return path


def _xyz(text: str) -> np.ndarray:
    parts = [p for p in text.replace(" ", "").split(",") if p]
    if len(parts) != 3:
        raise ValueError(f"expected X,Y,Z but got {text!r}")
    return np.array([float(p) for p in parts])


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"cannot serialise {type(obj)}")


if __name__ == "__main__":
    raise SystemExit(main())
