"""The Modal app that serves SAM 3D Objects and SAM 3D Body to Locaish.

Deploy with `modal deploy locaish/meta/modal_app.py`; call it through
`locaish/meta/sam3d.py`; read `docs/SAM3D_MODAL.md` for the secrets, the
JSON contracts and the licence. This file cannot run on the Mac: `modal` is
not installed here and nothing in it imports `torch`. Everything under
`if modal is not None` is the app; everything above it is plain numpy that
the tests check (`tests/test_sam3d.py`): the bearer check, the image and
mask decoding, and the re-orientation that stands the body up.

Two models, two images, two GPUs, one app:

- SAM 3D Objects (github.com/facebookresearch/sam-3d-objects, pinned at
  `OBJECTS_COMMIT`): `doc/setup.md` asks for "NVIDIA GPU with at least 32 GB
  VRAM", which rules out the 24 GB L4 and A10G. It runs on an **L40S** (48 GB,
  $1.95/h on Modal's list price). Its environment is the conda file
  translated to a CUDA 12.1 devel container: torch 2.5.1+cu121, PyTorch3D
  built from source, Kaolin 0.17.0 from NVIDIA's wheel index, gsplat and
  MoGe from pinned commits, flash-attn 2.8.3, and the repo's hydra patch.
- SAM 3D Body (github.com/facebookresearch/sam-3d-body, pinned at
  `BODY_COMMIT`): an 840M-parameter DINOv3 backbone plus a ViTDet-H human
  detector and MoGe-2 for the field of view; about 8 GB resident. It runs on
  an **L4** (24 GB, $0.80/h). `INSTALL.md` leaves torch unpinned; the same
  2.5.1+cu121 as the objects image is used so the two stay comparable.

Each endpoint is a `@modal.fastapi_endpoint(method="POST")` on a class whose
`@modal.enter()` loads the model once per container. Modal gives every web
function its own hostname, fixed here with `label=`, so the client's URL
template is `https://<workspace>--locaish-sam3d-{endpoint}.modal.run`.
Requests carry `Authorization: Bearer <LOCAISH_SAM3D_TOKEN>` from the
`locaish-sam3d` secret; checkpoints are fetched at image build with
`HF_TOKEN` from the `huggingface` secret (both Hugging Face repos are gated:
accept the SAM License on each first).

Every inference step is annotated with the file in the upstream repo it
was copied from, because the next person to touch this will be debugging it
on Modal with a cold cache and no laptop GPU.
"""

from __future__ import annotations

import base64
import hmac
import io
import json
import os
import sys
import time
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# pins and sizes
# --------------------------------------------------------------------------

OBJECTS_REPO = "https://github.com/facebookresearch/sam-3d-objects.git"
OBJECTS_COMMIT = "f91db411c50efee93d8db7aeb323885650f6f722"   # main, 2 Jun 2026: "SA-3DAO Release - README Update (#190)"
OBJECTS_HF_REPO = "facebook/sam-3d-objects"                   # gated; files land under checkpoints/
OBJECTS_GPU = "L40S"                                          # setup.md: ">= 32 GB VRAM"; L40S has 48 GB

BODY_REPO = "https://github.com/facebookresearch/sam-3d-body.git"
BODY_COMMIT = "b5c765a0d89d789985e186d396315e7590887b94"      # main, 19 Feb 2026: "add arXiv"
BODY_HF_REPO = "facebook/sam-3d-body-dinov3"                  # gated; model.ckpt, model_config.yaml, assets/mhr_model.pt
BODY_GPU = "L4"                                               # 840M params + ViTDet-H + MoGe-2 fit in 24 GB

# MoGe pinned to the commit sam-3d-objects' requirements.txt pins; sam-3d-body's
# INSTALL.md says `pip install git+https://github.com/microsoft/MoGe.git`
# unpinned, and MoGe's main has since moved to MoGe-3 (needs Triton). The
# pinned commit still has `moge.model.v2`, which `tools/build_fov_estimator.py`
# imports.
MOGE_COMMIT = "a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"
DETECTRON2_COMMIT = "a1ce2f9"                                 # INSTALL.md step 4

OBJECTS_ROOT = "/opt/sam-3d-objects"
BODY_ROOT = "/opt/sam-3d-body"
BODY_CKPT_DIR = "/opt/checkpoints/sam-3d-body-dinov3"

try:  # the Mac that runs the tests has no modal; the module must still import
    import modal  # type: ignore
    MODAL_IMPORT_ERROR: str | None = None
except ImportError as _exc:  # pragma: no cover - depends on the environment
    modal = None  # type: ignore[assignment]
    MODAL_IMPORT_ERROR = (
        f"the `modal` package is not installed ({_exc}); this module only defines the app "
        "when it is. Install the `sam3d` extra (`pip install -e '.[sam3d]'`) on the machine "
        "that deploys, then `modal deploy locaish/meta/modal_app.py`."
    )


def available() -> bool:
    """Whether the Modal app is defined in this process."""
    return modal is not None


# --------------------------------------------------------------------------
# plain helpers: no modal, no torch; tested on the Mac
# --------------------------------------------------------------------------


def check_bearer(authorization: str | None, expected: str | None) -> bool:
    """`Authorization: Bearer <token>` against the deployment's token, in
    constant time. A missing expected token refuses everything: an app
    deployed without the `locaish-sam3d` secret is closed, not open."""
    if not expected or not authorization:
        return False
    scheme, _, token = authorization.strip().partition(" ")
    if scheme.lower() != "bearer" or not token:
        return False
    return hmac.compare_digest(token.strip().encode("utf-8"), expected.encode("utf-8"))


def decode_image_b64(b64: str) -> np.ndarray:
    """PNG/JPEG base64 -> (H, W, 3) uint8 RGB. Both models want RGB arrays:
    `notebook/inference.py::load_image` for objects (it then appends the
    mask as alpha), `SAM3DBodyEstimator.process_one_image` for body (which
    prints 'Please make sure the input image is in RGB format' for arrays)."""
    from PIL import Image

    im = Image.open(io.BytesIO(base64.b64decode(b64)))
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def decode_mask_b64(b64: str, shape_hw: tuple[int, int]) -> np.ndarray:
    """Mask PNG base64 -> (H, W) bool, resized (nearest) to the image when the
    sizes differ. Mirrors `notebook/inference.py::load_mask`: `mask > 0`, and
    the last channel if the PNG has several."""
    from PIL import Image

    im = Image.open(io.BytesIO(base64.b64decode(b64)))
    arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[..., -1]
    mask = arr > 0
    if mask.shape != tuple(shape_hw):
        h, w = shape_hw
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
    return mask


# Momentum Human Rig keypoints, the first 70 of 308: names from
# sam_3d_body/metadata/mhr70.py at BODY_COMMIT. Only the indices used to
# orient the figure are listed here.
MHR70_HEAD = (0, 1, 2, 3, 4)                 # nose, left-eye, right-eye, left-ear, right-ear
MHR70_FEET = (13, 14, 15, 16, 17, 18, 19, 20)  # ankles, toe tips, heels
MHR70_ANKLES = (13, 14)
MHR70_LEFT = (9, 5)                          # left-hip, left-shoulder
MHR70_RIGHT = (10, 6)                        # right-hip, right-shoulder

# The model's frame is the camera's, OpenCV style: +X right, +Y down, +Z
# into the scene (sam_3d_body/visualization/renderer.py rotates the mesh
# 180 degrees about X before handing it to pyrender, whose camera is Y-up).
# A person standing upright and facing the camera therefore has their head
# at -Y, their front toward -Z and their right side at -X. The fallback
# rotation below maps that to the contract: X = body's right, Y = facing,
# Z = up. Its determinant is +1 (a rotation, not a reflection).
CAMERA_TO_UPRIGHT_FALLBACK = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])


def upright_from_keypoints(
    vertices: np.ndarray, keypoints: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict[str, Any]]:
    """Stand the figure up from its own joints: feet lowest, head highest.

    Up is the vector from the feet (ankles, heels, toe tips) to the head
    (nose, eyes, ears). Left is hips and shoulders, left minus right, made
    orthogonal to up. Forward is left x up (so a person facing the camera
    faces -Z in the camera frame, as expected). The new axes are
    X = right (= -left), Y = forward, Z = up; the rows of R are those unit
    vectors in the old frame and R has determinant +1. Differences of
    keypoints are translation-invariant, so it does not matter whether
    `pred_keypoints_3d` and `pred_vertices` share an origin. The feet are
    dropped to z = 0 and the ankle midpoint moved to x = y = 0, so the
    figure stands on the origin like an actor on a mark.

    Assumption: the subject is a whole standing (or at least upright-torso)
    person. A figure lying down is stood up by this, which is what the cast
    module wants; a figure missing its feet in the keypoints falls back to
    the fixed camera-frame mapping, recorded in the returned notes.
    """
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    kp = None if keypoints is None else np.asarray(keypoints, dtype=np.float64).reshape(-1, 3)
    notes: dict[str, Any] = {}
    R = None
    if kp is not None and len(kp) > max(MHR70_FEET + MHR70_HEAD + MHR70_LEFT + MHR70_RIGHT):
        head = kp[list(MHR70_HEAD)].mean(axis=0)
        feet = kp[list(MHR70_FEET)].mean(axis=0)
        up = head - feet
        left = kp[list(MHR70_LEFT)].mean(axis=0) - kp[list(MHR70_RIGHT)].mean(axis=0)
        if np.linalg.norm(up) > 1e-6:
            up = up / np.linalg.norm(up)
            left = left - up * float(left @ up)
            if np.linalg.norm(left) > 1e-6:
                left = left / np.linalg.norm(left)
                forward = np.cross(left, up)
                R = np.stack([-left, forward, up])
                notes["method"] = "keypoints"
    if R is None:
        R = CAMERA_TO_UPRIGHT_FALLBACK
        notes["method"] = "fallback_camera_frame"
    v = v @ R.T
    kp = None if kp is None else kp @ R.T
    if kp is not None and notes["method"] == "keypoints":
        mark = kp[list(MHR70_ANKLES)].mean(axis=0)
        shift = np.array([mark[0], mark[1], v[:, 2].min()])
    else:
        c = (v.min(axis=0) + v.max(axis=0)) / 2.0
        shift = np.array([c[0], c[1], v[:, 2].min()])
    v = v - shift
    kp = None if kp is None else kp - shift
    notes["R"] = R.tolist()
    notes["shift"] = shift.tolist()
    notes["height_units"] = float(v[:, 2].max() - v[:, 2].min())
    if kp is not None and notes["method"] == "keypoints":
        # The construction guarantees this; it is recorded so a wrong joint
        # order after an upstream change shows up in the response, not in a
        # figure standing on its head in the twin.
        notes["head_above_feet"] = bool(kp[list(MHR70_HEAD)][:, 2].mean() > kp[list(MHR70_FEET)][:, 2].mean())
    return v, kp, R, notes


def _to_list(x: Any) -> Any:
    """Tensors and arrays in a result dict -> JSON-able lists."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


# --------------------------------------------------------------------------
# the app
# --------------------------------------------------------------------------

if modal is not None:

    try:
        # FastAPI resolves the `request: Request` annotation against this
        # module's globals when the container imports it; `modal` depends on
        # fastapi, so this only fails on a machine that cannot deploy anyway.
        from fastapi import Request
    except ImportError:  # pragma: no cover
        Request = Any  # type: ignore[misc,assignment]

    app = modal.App("locaish-sam3d")

    hf_secret = modal.Secret.from_name("huggingface")          # HF_TOKEN
    auth_secret = modal.Secret.from_name("locaish-sam3d")      # LOCAISH_SAM3D_TOKEN

    # -- SAM 3D Objects image --------------------------------------------
    # doc/setup.md at OBJECTS_COMMIT, translated from `mamba env create -f
    # environments/default.yml` (CUDA 12.1 toolchain from conda-forge) to the
    # CUDA 12.1 devel container. The three `pip install -e` steps are the
    # doc's, in its order, with its two index variables.

    def _download_objects_checkpoints() -> None:
        # doc/setup.md section 2:
        #   hf download --repo-type model --local-dir checkpoints/${TAG}-download --max-workers 1 facebook/sam-3d-objects
        #   mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
        # with TAG=hf, so demo.py's `checkpoints/hf/pipeline.yaml` resolves.
        import shutil
        from huggingface_hub import snapshot_download

        tmp = os.path.join(OBJECTS_ROOT, "checkpoints", "hf-download")
        snapshot_download(OBJECTS_HF_REPO, repo_type="model", local_dir=tmp, max_workers=1,
                          token=os.environ["HF_TOKEN"])
        dst = os.path.join(OBJECTS_ROOT, "checkpoints", "hf")
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.move(os.path.join(tmp, "checkpoints"), dst)
        shutil.rmtree(tmp, ignore_errors=True)

    def _warm_objects() -> None:
        # Build the pipeline once at image build so MoGe (the pointmap/depth
        # model `InferencePipelinePointMap.compute_pointmap` runs first) lands
        # in the Hugging Face cache inside the image, not at every cold start.
        os.chdir(OBJECTS_ROOT)
        sys.path.insert(0, os.path.join(OBJECTS_ROOT, "notebook"))
        from inference import Inference  # notebook/inference.py

        Inference(os.path.join(OBJECTS_ROOT, "checkpoints", "hf", "pipeline.yaml"), compile=False)

    objects_image = (
        modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
        .apt_install("git", "build-essential", "ninja-build", "libgl1", "libglib2.0-0", "libegl1",
                     "libgomp1", "libxrender1", "libxi6", "libxkbcommon0", "libsm6", "libxext6")
        .env({
            # notebook/inference.py line 5: os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
            # (it assumes a conda env; in the container both point at the toolkit).
            "CONDA_PREFIX": "/usr/local/cuda",
            "CUDA_HOME": "/usr/local/cuda",
            "LIDRA_SKIP_INIT": "true",                      # inference.py line 6
            # doc/setup.md: the two package indexes (cu121 torch, NGC) and the Kaolin wheel index
            "PIP_EXTRA_INDEX_URL": "https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121",
            "PIP_FIND_LINKS": "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html",
            # PyTorch3D, gsplat and flash-attn compile CUDA kernels at install:
            # one architecture (L40S = Ada, sm_89) keeps that under an hour, and
            # FORCE_CUDA avoids setup.md's "RuntimeError: Not compiled with GPU
            # support" when the build host has no visible device.
            "TORCH_CUDA_ARCH_LIST": "8.9",
            "FORCE_CUDA": "1",
            "MAX_JOBS": "8",
            "HF_HOME": "/root/.cache/huggingface",
        })
        .run_commands(
            f"git clone {OBJECTS_REPO} {OBJECTS_ROOT} && cd {OBJECTS_ROOT} && git checkout {OBJECTS_COMMIT}",
        )
        .workdir(OBJECTS_ROOT)
        # requirements.txt pins torchaudio==2.5.1+cu121, which fixes torch at
        # 2.5.1+cu121; installing torch first lets the source builds below see it.
        .run_commands("pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121")
        .run_commands("pip install -e '.[dev]'")                       # setup.md step: core + dev
        .run_commands("pip install -e '.[p3d]'", gpu=OBJECTS_GPU)      # setup.md: PyTorch3D separately (source build)
        .run_commands("pip install -e '.[inference]'", gpu=OBJECTS_GPU)  # setup.md: kaolin, gsplat, MoGe, flash-attn
        .run_commands("python ./patching/hydra")                       # setup.md: the hydra 1.3.2 patch
        .pip_install("fastapi[standard]", "huggingface_hub[cli]<1.0", "trimesh")
        .run_function(_download_objects_checkpoints, secrets=[hf_secret], timeout=3 * 3600)
        .run_function(_warm_objects, secrets=[hf_secret], gpu=OBJECTS_GPU, timeout=3600)
    )

    # -- SAM 3D Body image -----------------------------------------------
    # INSTALL.md at BODY_COMMIT, steps 1-5, plus the checkpoints section.

    BODY_PIP = (  # INSTALL.md step 3, verbatim list
        "pytorch-lightning", "pyrender", "opencv-python", "yacs", "scikit-image", "einops", "timm",
        "dill", "pandas", "rich", "hydra-core", "hydra-submitit-launcher", "hydra-colorlog",
        "pyrootutils", "webdataset", "chump", "networkx==3.2.1", "roma", "joblib", "seaborn", "wandb",
        "appdirs", "appnope", "ffmpeg", "cython", "jsonlines", "pytest", "xtcocotools", "loguru",
        "optree", "fvcore", "black", "pycocotools", "tensorboard", "huggingface_hub",
    )

    def _download_body_checkpoints() -> None:
        # README "Getting Started":
        #   hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3
        # giving model.ckpt, model_config.yaml and assets/mhr_model.pt.
        from huggingface_hub import snapshot_download

        snapshot_download(BODY_HF_REPO, repo_type="model", local_dir=BODY_CKPT_DIR, token=os.environ["HF_TOKEN"])

    def _build_body_estimator():
        # notebook/utils.py::setup_sam_3d_body, with the local checkpoint
        # instead of load_sam_3d_body_hf (the weights are already in the image).
        os.chdir(BODY_ROOT)
        if BODY_ROOT not in sys.path:
            sys.path.insert(0, BODY_ROOT)                    # `tools` and `sam_3d_body` are repo-root packages
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body   # sam_3d_body/__init__.py
        from tools.build_detector import HumanDetector               # ViTDet-H, weights from dl.fbaipublicfiles.com
        from tools.build_fov_estimator import FOVEstimator           # MoGe-2 "Ruicheng/moge-2-vitl-normal"

        model, model_cfg = load_sam_3d_body(                 # sam_3d_body/build_models.py
            os.path.join(BODY_CKPT_DIR, "model.ckpt"), device="cuda",
            mhr_path=os.path.join(BODY_CKPT_DIR, "assets", "mhr_model.pt"),
        )
        return SAM3DBodyEstimator(                           # sam_3d_body/sam_3d_body_estimator.py
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=HumanDetector(name="vitdet", device="cuda"),
            human_segmentor=None,                            # demo.py: SAM2 only when a segmentor path is given
            fov_estimator=FOVEstimator(name="moge2", device="cuda"),
        )

    def _warm_body() -> None:
        # Construct once at build: the ViTDet pickle and the MoGe-2 weights are
        # downloaded by their loaders and end up cached inside the image.
        _build_body_estimator()

    body_image = (
        modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
        .apt_install("git", "build-essential", "ninja-build", "libgl1", "libglib2.0-0", "libegl1",
                     "libgomp1", "libxrender1", "libxext6", "libsm6")
        .env({
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "8.9",                   # L4 is Ada too (sm_89)
            "FORCE_CUDA": "1",
            "PYOPENGL_PLATFORM": "egl",                      # pyrender without a display
            "HF_HOME": "/root/.cache/huggingface",
            "FVCORE_CACHE": "/root/.torch/fvcore_cache",     # where detectron2's checkpointer keeps the ViTDet pickle
        })
        .run_commands(f"git clone {BODY_REPO} {BODY_ROOT} && cd {BODY_ROOT} && git checkout {BODY_COMMIT}")
        .workdir(BODY_ROOT)
        # INSTALL.md step 2 says "install PyTorch following the official instructions";
        # pinned to the same build as the objects image.
        .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu121")
        .pip_install(*BODY_PIP)                                                       # step 3
        .run_commands(                                                                # step 4
            f"pip install 'git+https://github.com/facebookresearch/detectron2.git@{DETECTRON2_COMMIT}' "
            "--no-build-isolation --no-deps",
            gpu=BODY_GPU,
        )
        .pip_install(f"git+https://github.com/microsoft/MoGe.git@{MOGE_COMMIT}")     # step 5 (pinned, see MOGE_COMMIT)
        .pip_install("fastapi[standard]", "trimesh")
        .run_function(_download_body_checkpoints, secrets=[hf_secret], timeout=3600)
        .run_function(_warm_body, gpu=BODY_GPU, timeout=3600)
    )

    # -- shared endpoint plumbing ----------------------------------------

    def _require_bearer(request) -> None:
        from fastapi import HTTPException

        if not check_bearer(request.headers.get("authorization"), os.environ.get("LOCAISH_SAM3D_TOKEN")):
            raise HTTPException(status_code=401, detail="bad or missing bearer token")

    def _field(payload: dict, key: str) -> str:
        from fastapi import HTTPException

        v = payload.get(key)
        if not isinstance(v, str) or not v:
            raise HTTPException(status_code=422, detail=f"{key} (base64 string) is required")
        return v

    # -- SAM 3D Objects ----------------------------------------------------

    @app.cls(
        image=objects_image,
        gpu=OBJECTS_GPU,
        secrets=[auth_secret, hf_secret],
        timeout=20 * 60,
        scaledown_window=5 * 60,
        memory=32 * 1024,
    )
    class SAM3DObjects:
        @modal.enter()
        def load(self) -> None:
            # demo.py at OBJECTS_COMMIT:
            #   sys.path.append("notebook"); from inference import Inference, ...
            #   inference = Inference(f"checkpoints/{tag}/pipeline.yaml", compile=False)
            os.chdir(OBJECTS_ROOT)
            sys.path.insert(0, os.path.join(OBJECTS_ROOT, "notebook"))
            from inference import Inference

            t0 = time.time()
            self.inference = Inference(os.path.join(OBJECTS_ROOT, "checkpoints", "hf", "pipeline.yaml"), compile=False)
            self.load_s = time.time() - t0

        def _lift(self, image_png_b64: str, mask_png_b64: str, seed: int) -> dict[str, Any]:
            import tempfile

            t0 = time.time()
            image = decode_image_b64(image_png_b64)                       # (H, W, 3) uint8 RGB
            mask = decode_mask_b64(mask_png_b64, image.shape[:2])         # (H, W) bool
            if not mask.any():
                return {"error": "the mask is empty"}
            # demo.py: output = inference(image, mask, seed=42)
            # notebook/inference.py::Inference.__call__ embeds the mask as alpha
            # (merge_mask_to_rgba) and runs InferencePipelinePointMap.run with
            # with_mesh_postprocess=False, with_texture_baking=False,
            # use_vertex_color=True; decode_formats come from pipeline.yaml
            # (["gaussian", "mesh"] by default), so the result carries both
            # "gs" (the GaussianModel) and "glb" (a trimesh with vertex colours).
            output = self.inference(image, mask, seed=int(seed))
            gs = output.get("gs")
            if gs is None:
                return {"error": "the model returned no gaussians"}
            with tempfile.TemporaryDirectory() as d:
                # demo.py: output["gs"].save_ply("splat.ply"), the standard 3DGS
                # layout (x y z nx ny nz f_dc_0..2 opacity scale_0..2 rot_0..3), see
                # sam3d_objects/model/backbone/tdfy_dit/representations/gaussian/gaussian_model.py::save_ply
                p = os.path.join(d, "splat.ply")
                gs.save_ply(p)
                with open(p, "rb") as f:
                    ply = f.read()
            glb_b64 = None
            glb = output.get("glb")
            if glb is not None:
                try:
                    glb_b64 = base64.b64encode(glb.export(file_type="glb")).decode("ascii")
                except Exception as exc:  # the mesh is a bonus, the splat is the product
                    glb_b64 = None
                    glb_err = f"{type(exc).__name__}: {exc}"
                else:
                    glb_err = None
            else:
                glb_err = "no mesh in the decode formats"
            meta: dict[str, Any] = {
                "model": "sam-3d-objects",
                "commit": OBJECTS_COMMIT,
                "gpu": OBJECTS_GPU,
                "seed": int(seed),
                "n_gaussians": int(gs.get_xyz.shape[0]),
                "image_hw": [int(image.shape[0]), int(image.shape[1])],
                "mask_px": int(mask.sum()),
                "seconds": round(time.time() - t0, 2),
                "load_seconds": round(self.load_s, 2),
                "mesh_error": glb_err,
                # Object pose in the camera frame of the input, from the pipeline's
                # pose decoder (InferencePipelinePointMap.run -> ss_return_dict):
                # a quaternion, a translation and a uniform scale that take the
                # object-local gaussians into the pointmap's camera frame
                # (notebook/inference.py::make_scene applies exactly these).
                "pose": {k: _to_list(output[k]) for k in ("rotation", "translation", "scale") if k in output},
            }
            return {"gaussians_ply_b64": base64.b64encode(ply).decode("ascii"),
                    "mesh_glb_b64": glb_b64, "meta": meta}

        @modal.method()
        def lift(self, image_png_b64: str, mask_png_b64: str, seed: int = 42) -> dict[str, Any]:
            """The same job without HTTP, for `modal run` and for Modal-side callers."""
            return self._lift(image_png_b64, mask_png_b64, seed)

        @modal.fastapi_endpoint(method="POST", label="locaish-sam3d-object")
        def object(self, payload: dict, request: Request) -> dict[str, Any]:
            _require_bearer(request)
            return self._lift(_field(payload, "image_png_b64"), _field(payload, "mask_png_b64"),
                              int(payload.get("seed", 42)))

    # -- SAM 3D Body -------------------------------------------------------

    @app.cls(
        image=body_image,
        gpu=BODY_GPU,
        secrets=[auth_secret, hf_secret],
        timeout=10 * 60,
        scaledown_window=5 * 60,
        memory=16 * 1024,
    )
    class SAM3DBody:
        @modal.enter()
        def load(self) -> None:
            t0 = time.time()
            self.estimator = _build_body_estimator()
            # SAM3DBodyEstimator.__init__: self.faces = self.model.head_pose.faces.cpu().numpy()
            self.faces = np.asarray(self.estimator.faces, dtype=np.int64).reshape(-1, 3)
            self.load_s = time.time() - t0

        def _lift(self, image_png_b64: str) -> dict[str, Any]:
            t0 = time.time()
            image = decode_image_b64(image_png_b64)                       # RGB, as process_one_image expects for arrays
            # README quick start / demo.py:
            #   outputs = estimator.process_one_image(img_rgb, bbox_thr=0.8)
            # ViTDet finds the people (default_to_full_image=False, so no person
            # means an empty list), MoGe-2 estimates the focal length, and each
            # entry (sam_3d_body_estimator.py, the all_out dict) carries
            # pred_vertices (V, 3), pred_keypoints_3d (K, 3), pred_cam_t (3,),
            # focal_length, bbox (4,) and the MHR parameters.
            outputs = self.estimator.process_one_image(image, bbox_thr=0.8)
            if not outputs:
                return {"error": "no person detected"}
            areas = [float((o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1])) for o in outputs]
            person = outputs[int(np.argmax(areas))]
            verts = np.asarray(person["pred_vertices"], dtype=np.float64).reshape(-1, 3)
            kp = person.get("pred_keypoints_3d")
            kp = None if kp is None else np.asarray(kp, dtype=np.float64).reshape(-1, 3)
            v_up, kp_up, R, notes = upright_from_keypoints(verts, kp)
            meta: dict[str, Any] = {
                "model": "sam-3d-body",
                "commit": BODY_COMMIT,
                "hf_repo": BODY_HF_REPO,
                "gpu": BODY_GPU,
                "n_people": len(outputs),
                "bbox_xyxy": _to_list(person["bbox"]),
                "focal_length_px": _to_list(person["focal_length"]),
                "cam_t": _to_list(person["pred_cam_t"]),       # camera-frame translation of the body (model units)
                "orientation": notes,
                "keypoint_format": "mhr70+" if kp is not None else None,
                "units": "model units; the MHR family is trained in metres, so height_units is a stature estimate",
                "image_hw": [int(image.shape[0]), int(image.shape[1])],
                "seconds": round(time.time() - t0, 2),
                "load_seconds": round(self.load_s, 2),
            }
            return {
                "vertices": v_up.tolist(),
                "faces": self.faces.tolist(),
                "keypoints_3d": None if kp_up is None else kp_up.tolist(),
                "height_units": notes["height_units"],
                "meta": meta,
            }

        @modal.method()
        def lift(self, image_png_b64: str) -> dict[str, Any]:
            return self._lift(image_png_b64)

        @modal.fastapi_endpoint(method="POST", label="locaish-sam3d-body")
        def body(self, payload: dict, request: Request) -> dict[str, Any]:
            _require_bearer(request)
            return self._lift(_field(payload, "image_png_b64"))

    # -- local entrypoint ---------------------------------------------------

    @app.local_entrypoint()
    def main(image: str, mask: str = "", body: bool = False, out: str = "") -> None:
        """Post a test image through the deployed classes (no HTTP, no token):

            modal run locaish/meta/modal_app.py --image frame.png --mask mask.png --out sofa.ply
            modal run locaish/meta/modal_app.py --image headshot.jpg --body --out actor.json

        and print the two web URLs the client needs for LOCAISH_SAM3D_URL.
        """
        with open(image, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")
        t0 = time.time()
        if body:
            res = SAM3DBody().lift.remote(image_b64)
            if "error" in res:
                print("body:", res["error"])
                return
            print(f"body: {len(res['vertices'])} vertices, {len(res['faces'])} faces, "
                  f"height {res['height_units']:.3f} units, {time.time() - t0:.1f} s round trip")
            print(json.dumps(res["meta"], indent=2))
            if out:
                with open(out, "w") as f:
                    json.dump(res, f)
                print("wrote", out)
        else:
            if not mask:
                raise SystemExit("--mask is required for an object lift")
            with open(mask, "rb") as f:
                mask_b64 = base64.b64encode(f.read()).decode("ascii")
            res = SAM3DObjects().lift.remote(image_b64, mask_b64, 42)
            if "error" in res:
                print("object:", res["error"])
                return
            print(f"object: {res['meta']['n_gaussians']} gaussians, mesh={'yes' if res['mesh_glb_b64'] else 'no'}, "
                  f"{time.time() - t0:.1f} s round trip")
            print(json.dumps(res["meta"], indent=2))
            if out:
                with open(out, "wb") as f:
                    f.write(base64.b64decode(res["gaussians_ply_b64"]))
                print("wrote", out)
        try:
            print("object endpoint:", SAM3DObjects().object.get_web_url())
            print("body endpoint:  ", SAM3DBody().body.get_web_url())
        except Exception as exc:  # pragma: no cover - needs a deployment
            print("web URLs unavailable until `modal deploy`:", exc)
