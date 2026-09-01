"""Dense stereo on a GCP GPU instance, from a laptop that has none.

COLMAP's PatchMatch stereo -- depth and normal per pixel, geometric consistency
enforced across views -- is the best classical densifier available and it is
CUDA-only, which on a Mac means it is no densifier at all. This module ships
the one stage that needs the GPU to a Google Cloud instance and brings the
fused cloud back: undistortion happens locally (it is cheap and CPU-bound),
the undistorted workspace goes up, `patch_match_stereo` and `stereo_fusion`
run there, and `fused.ply` comes home. On an L4 the stage that takes ten CPU
minutes through OpenMVS takes on the order of ninety seconds, with the
geometric-consistency pass that the CPU fallback does not have.

Everything classical about the pipeline stays classical: the remote machine
runs the same COLMAP commands a local CUDA build would, and nothing trained is
involved. What changes is only where the arithmetic happens -- which also puts
a genuine Google Cloud dependency into the compute path of every hosted twin,
not just into the agent wrapper around it.

Configuration is three environment variables, because the instance is
infrastructure and not an option a caller should be plumbing through the
pipeline:

    LOCAISH_GPU_INSTANCE   name of the GCE instance (required to enable)
    LOCAISH_GPU_ZONE       its zone, e.g. us-central1-a (required)
    LOCAISH_GPU_COLMAP     command that runs CUDA COLMAP there
                           (default "colmap"; a docker wrapper works, e.g.
                           "docker run --rm --gpus all -v /tmp:/tmp
                            colmap/colmap:latest colmap")

`docs/GCP_GPU_DENSE.md` holds the instance recipe. Every remote step is
best-effort from the pipeline's point of view: `reconstruct` falls back to the
local densifiers when this module raises, so a stopped instance degrades the
twin instead of killing it.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np

# Ceiling on any single gcloud invocation. The stereo solve dominates and is
# minutes on a GPU; an hour means the instance is wedged, not slow.
STEP_TIMEOUT_S = 3600


class RemoteDenseError(RuntimeError):
    """The GPU instance is unconfigured, unreachable, or failed the solve."""


def remote_config() -> dict | None:
    """The instance to use, or None when remote densification is not set up."""
    instance = os.environ.get("LOCAISH_GPU_INSTANCE")
    zone = os.environ.get("LOCAISH_GPU_ZONE")
    if not instance or not zone:
        return None
    if shutil.which("gcloud") is None:
        return None
    return {
        "instance": instance,
        "zone": zone,
        "colmap": os.environ.get("LOCAISH_GPU_COLMAP", "colmap"),
        "project": os.environ.get("LOCAISH_GPU_PROJECT"),
    }


def densify_remote(
    image_dir: str | Path,
    model_dir: str | Path,
    work_dir: str | Path,
    *,
    max_image_size: int = 1600,
    progress=None,
) -> np.ndarray:
    """PatchMatch stereo on the configured GCP instance. Returns (N, 6) xyz+rgb.

    The workspace layout mirrors `densify_patchmatch` exactly -- undistort
    locally into `work_dir/dense`, solve remotely, and the caller reads the
    same `fused.ply` either way. The remote working directory is keyed by the
    local workspace name so that two twins densifying concurrently on one
    instance cannot trample each other.
    """
    from .colmap import ColmapError, executable

    cfg = remote_config()
    if cfg is None:
        raise RemoteDenseError(
            "remote densification is not configured; set LOCAISH_GPU_INSTANCE "
            "and LOCAISH_GPU_ZONE (see docs/GCP_GPU_DENSE.md)"
        )

    work_dir = Path(work_dir).resolve()
    dense = work_dir / "dense"
    if dense.exists():
        shutil.rmtree(dense)
    dense.mkdir(parents=True)

    if progress:
        progress("colmap undistort (local)")
    proc = subprocess.run(
        [
            executable(), "image_undistorter",
            "--image_path", str(Path(image_dir).resolve()),
            "--input_path", str(Path(model_dir).resolve()),
            "--output_path", str(dense),
            "--output_type", "COLMAP",
            "--max_image_size", str(max_image_size),
        ],
        capture_output=True, text=True,
    )
    (work_dir / "undistort.log").write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        raise ColmapError(f"colmap undistort failed: {' / '.join(tail)[:400]}")

    remote_root = f"/tmp/locaish-dense/{work_dir.parent.name}-{work_dir.name}"
    archive = work_dir / "dense_up.tar.gz"
    if progress:
        progress("uploading workspace to GPU instance")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(dense, arcname="dense")

    def gcloud(args: list[str], step: str, timeout: int = STEP_TIMEOUT_S) -> str:
        cmd = ["gcloud", "compute"] + args + ["--zone", cfg["zone"]]
        if cfg["project"]:
            cmd += ["--project", cfg["project"]]
        got = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        (work_dir / f"remote_{step}.log").write_text(got.stdout + "\n" + got.stderr)
        if got.returncode != 0:
            tail = (got.stderr or got.stdout).strip().splitlines()[-3:]
            raise RemoteDenseError(f"gcloud {step} failed: {' / '.join(tail)[:400]}")
        return got.stdout

    def ssh(command: str, step: str) -> str:
        return gcloud(
            ["ssh", cfg["instance"], "--command", command], step
        )

    ssh(f"rm -rf {shlex.quote(remote_root)} && mkdir -p {shlex.quote(remote_root)}", "prepare")
    gcloud(
        ["scp", str(archive), f"{cfg['instance']}:{remote_root}/dense.tar.gz"],
        "upload",
    )
    archive.unlink()

    if progress:
        progress("colmap patchmatch stereo (remote GPU)")
    colmap = cfg["colmap"]
    ssh(
        f"cd {shlex.quote(remote_root)} && tar xzf dense.tar.gz && "
        f"{colmap} patch_match_stereo "
        f"--workspace_path {shlex.quote(remote_root)}/dense "
        "--workspace_format COLMAP "
        "--PatchMatchStereo.geom_consistency true",
        "patchmatch",
    )

    if progress:
        progress("colmap stereo fusion (remote GPU)")
    ssh(
        f"{colmap} stereo_fusion "
        f"--workspace_path {shlex.quote(remote_root)}/dense "
        "--workspace_format COLMAP "
        "--input_type geometric "
        f"--output_path {shlex.quote(remote_root)}/dense/fused.ply",
        "fusion",
    )

    if progress:
        progress("downloading fused cloud")
    fused = dense / "fused.ply"
    gcloud(
        ["scp", f"{cfg['instance']}:{remote_root}/dense/fused.ply", str(fused)],
        "download",
    )
    # The remote workspace is scratch; leaving it costs disk on a billed
    # machine and cleanup failing is not worth failing the twin over.
    try:
        ssh(f"rm -rf {shlex.quote(remote_root)}", "cleanup")
    except RemoteDenseError:
        pass

    if not fused.exists() or fused.stat().st_size == 0:
        raise RemoteDenseError("the remote solve returned no fused cloud")

    from ..formats.ply import read_ply

    scan = read_ply(fused)
    xyz = scan.points.xyz
    rgb = scan.points.rgb
    if rgb is None:
        rgb = np.full((len(xyz), 3), 200, dtype=np.uint8)
    return np.hstack([xyz, rgb.astype(np.float64)])
