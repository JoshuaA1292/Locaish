# SAM 3D on Modal: Objects and Body as HTTP endpoints

*Written 24 October 2026 against sam-3d-objects `f91db411` (main, 2 Jun 2026) and
sam-3d-body `b5c765a0` (main, 19 Feb 2026), Modal 1.6.2 (`@modal.fastapi_endpoint`,
`scaledown_window`, `Image.run_function`). Code: `locaish/meta/modal_app.py` (the app),
`locaish/meta/sam3d.py` (the client), `tests/test_sam3d.py` (no modal, no network).*

The two open-weight SAM 3D models need a CUDA GPU; the laptop has Metal. They run
on Modal and Locaish posts a frame to them. SAM 3D Objects turns one masked object
into a textured gaussian splat (plus a vertex-coloured GLB), which `fit_to_box`
places on the label's measured footprint and `to_field_npz` writes in the same
representation as the Brush layer, so `film/render.py` composites by concatenation.
SAM 3D Body turns one person into a Momentum Human Rig mesh, stood up on the
origin, which the cast module rescales to the typed height.

## Deploy

```sh
# on the machine that deploys (not the Mac that runs the tests, which has no modal)
pip install -e '.[sam3d]'            # modal>=1.0
modal setup                          # once; browser login
modal secret create huggingface HF_TOKEN=hf_...
modal secret create locaish-sam3d LOCAISH_SAM3D_TOKEN="$(openssl rand -hex 32)"
modal deploy locaish/meta/modal_app.py
```

Before the first deploy, **request access to all three gated Hugging Face repos**
with the account that owns `HF_TOKEN` and accept the SAM License on each:
[`facebook/sam-3d-objects`](https://huggingface.co/facebook/sam-3d-objects),
[`facebook/sam-3d-body-dinov3`](https://huggingface.co/facebook/sam-3d-body-dinov3)
(and `facebook/sam-3d-body-vith` if you switch backbones). Meta's gate asks for a
full legal name, date of birth and organisation and refuses sanctioned
jurisdictions. The image build downloads the checkpoints with that token and bakes
them into the image (not a Volume: one fewer moving part, and Modal caches image
layers), so a revoked token only bites on the next rebuild.

The build is heavy and happens once per change to the image definition:

| Image | What it builds | Expect |
|---|---|---|
| `objects_image` | CUDA 12.1 devel container; torch 2.5.1+cu121; `pip install -e '.[dev]'`, `'.[p3d]'` (PyTorch3D from source), `'.[inference]'` (Kaolin 0.17.0 wheel, gsplat from source, MoGe, flash-attn 2.8.3); the repo's hydra patch; ~10 GB of checkpoints; one warm-up instantiation so MoGe's weights are cached | 45-90 min the first time; the source builds dominate. `TORCH_CUDA_ARCH_LIST=8.9` keeps them to one architecture (L40S) |
| `body_image` | CUDA 12.1 devel container; torch 2.5.1+cu121; INSTALL.md's pip list; detectron2 `a1ce2f9`; MoGe pinned; ~3.5 GB checkpoint; one warm-up so ViTDet-H and MoGe-2 weights are cached | 15-30 min |

Then point the client at it:

```sh
export LOCAISH_SAM3D_URL='https://<workspace>--locaish-sam3d-{endpoint}.modal.run'
export LOCAISH_SAM3D_TOKEN='<the same hex>'
```

Modal gives every web function its own hostname; the app fixes them with
`label="locaish-sam3d-object"` and `label="locaish-sam3d-body"`, which is why the
URL is a template with an `{endpoint}` slot. A URL without the slot is treated as a
base and `/object`, `/body` appended (for a reverse proxy); `LOCAISH_SAM3D_OBJECT_URL`
and `LOCAISH_SAM3D_BODY_URL` override one endpoint each. `modal deploy` prints the
two URLs; so does the local entrypoint.

Smoke test without HTTP or the token (calls the classes directly):

```sh
modal run locaish/meta/modal_app.py --image frame.png --mask sofa_mask.png --out sofa.ply
modal run locaish/meta/modal_app.py --image headshot.jpg --body --out actor.json
```

## GPUs, time and cost

Neither repository publishes inference times or VRAM figures beyond one line:
`doc/setup.md` in sam-3d-objects requires an "NVIDIA GPU with at least 32 GB VRAM".
That rules out the L4 and A10G (24 GB) the plan assumed, so:

| Endpoint | GPU | Why | Modal list price |
|---|---|---|---|
| `object` | **L40S** (48 GB) | the cheapest Modal card that clears 32 GB (the A100-40 also does, at $2.10/h, and the A100-80 at $2.50/h) | $0.000542/s = $1.95/h |
| `body` | **L4** (24 GB) | 840M-parameter DINOv3-H+ backbone plus ViTDet-H plus MoGe-2 is ~8 GB resident | $0.000222/s = $0.80/h |

Timing figures below are **estimates, not measurements**; the first live run should
replace them (every response carries `meta.seconds` and `meta.load_seconds`).

- Cold start: image pull plus model load. Objects loads six checkpoints and MoGe,
  roughly 60-120 s; body roughly 30-60 s. `scaledown_window` is 5 min, so a
  session that lifts several objects pays this once.
- Per call, warm: objects is a two-stage latent diffusion (sparse structure, then
  structured latent) plus a gaussian and a mesh decode, in the TRELLIS family,
  so expect 20-60 s on an L40S; body is one forward pass plus a detector and a
  FOV estimate, 2-5 s on an L4.
- Cost per call at those numbers: objects about $0.02-0.04 warm (plus up to $0.07
  for a cold start), body under a cent. Modal's Starter plan carries $30/month of
  credit; a demo day of fifty lifts is a few dollars.
- Modal cuts a web request at **150 s** and answers `303` to a result URL that
  blocks until the job finishes; the client follows redirects, so a slow object
  lift still returns in one `lift_object` call (timeout 300 s by default).

## JSON contracts

Both endpoints are `POST` with `Content-Type: application/json` and
`Authorization: Bearer <LOCAISH_SAM3D_TOKEN>`. A bad or missing token is `401`; a
missing field is `422`; a model failure the server could name is `200` with
`{"error": "..."}` (the client raises `Sam3DError` on it); anything else is a
`5xx` with Modal's body. Request bodies may be up to 4 GiB; a 4K PNG frame is fine.

### `object`

Request:

```json
{"image_png_b64": "<PNG or JPEG of the whole frame>",
 "mask_png_b64":  "<PNG, same size; non-zero pixels are the object; RGBA uses alpha>",
 "seed": 42}
```

Response:

```json
{"gaussians_ply_b64": "<standard 3DGS binary PLY: x y z nx ny nz f_dc_0..2 opacity scale_0..2 rot_0..3>",
 "mesh_glb_b64": "<GLB with vertex colours, or null>",
 "meta": {"model": "sam-3d-objects", "commit": "f91db411...", "gpu": "L40S", "seed": 42,
          "n_gaussians": 48213, "image_hw": [1080, 1920], "mask_px": 142031,
          "seconds": 38.1, "load_seconds": 71.0, "mesh_error": null,
          "pose": {"rotation": [[w, x, y, z]], "translation": [[x, y, z]], "scale": [[s, s, s]]}}}
```

The PLY is what `GaussianModel.save_ply` writes (scales as log standard
deviations, opacity as a logit, colour as the degree-0 SH coefficient, quaternion
wxyz), which `video.splat.read_gaussian_ply` already decodes. The gaussians are in
the **object's local frame**: a unit-ish box, **Z up** (the notebook's
`_fix_gaussian_alignment` swaps Y and Z only to render with a Y-up camera), so
`fit_to_box` needs a uniform scale, a yaw and a translation and nothing else.
`meta.pose` is the pipeline's own estimate of where the object sits in the input
camera's frame (`notebook/inference.py::make_scene` applies exactly these three to
build a multi-object scene); Locaish does not use it, because the label pipeline
already has the position from the ray-marched hit and the size from the box
extent at that depth, but it is there for a cross-check.

### `body`

Request: `{"image_png_b64": "<PNG or JPEG>"}`

Response:

```json
{"vertices": [[x, y, z], ...],
 "faces": [[i, j, k], ...],
 "keypoints_3d": [[x, y, z], ...],
 "height_units": 1.74,
 "meta": {"model": "sam-3d-body", "commit": "b5c765a0...", "hf_repo": "facebook/sam-3d-body-dinov3",
          "gpu": "L4", "n_people": 1, "bbox_xyxy": [..], "focal_length_px": 1456.2, "cam_t": [x, y, z],
          "orientation": {"method": "keypoints", "R": [[..]], "shift": [..], "height_units": 1.74,
                          "head_above_feet": true},
          "keypoint_format": "mhr70+", "units": "...", "image_hw": [..], "seconds": 3.2, "load_seconds": 41.0}}
```

Vertices and keypoints are in the model's own units with the figure **already stood
up on the server**: +Z up, feet at z = 0, facing +Y, ankle midpoint at x = y = 0.
When several people are detected the largest box wins and `n_people` says so.
`height_units` is the vertical extent after re-orientation; the MHR family is
trained in metres, so it is a stature estimate, which is what the scale anchor in
`scan/scale.py` wants, but the cast module never relies on the units: it rescales
to the typed height.

**Orientation assumption.** The model's frame is the camera's (OpenCV: +X right,
+Y down, +Z into the scene; `visualization/renderer.py` rotates the mesh 180° about
X before pyrender, whose camera is Y-up). The server builds the upright frame from
the joints: up = head (nose, eyes, ears) minus feet (ankles, heels, toe tips); left
= hips and shoulders, left minus right, orthogonalised; forward = left × up; the
new axes are X = right, Y = forward, Z = up (determinant +1). The subject is
assumed to be a whole person with an upright torso in the body's own sense; a
person lying on the floor is stood up, which is what casting wants. If the
keypoints are missing or too few, a fixed camera-to-upright rotation is applied
that assumes the person was standing and facing the camera, and
`meta.orientation.method` says `fallback_camera_frame`. The client re-checks from
the vertex extents (tallest axis to Z, wider than deep) before writing the npz and
records what it changed in `meta.client_orientation`.

### Client-side files

`to_field_npz(lift, path, transform=M)` writes `xyz (N,3) f64`, `rgb (N,3) f32 0-1`,
`opacity (N,) f32`, `cov (N,3,3) f32` (Σ = R S Sᵀ Rᵀ, then A Σ Aᵀ for the linear
part A of `M`), plus `source` and `meta`. `fit_to_box(gaussians, size, centre, yaw)`
returns `M` with the smallest per-axis ratio as a uniform scale (nothing exceeds the
box), the yaw about Z, the box centre in x and y, and the object's base on the
bottom of the box (a sofa sits on the floor).

`to_body_npz(lift, path)` writes `vertices (V,3) f64`, `faces (F,3) i64`, `height_m`
(the lift's height in its units, passed through), `keypoints` when present,
`rgb` when the server ever supplies one, plus `source` and `meta`.

## What the upstream inference code actually looks like

For debugging on Modal with no laptop GPU.

**SAM 3D Objects** (`demo.py` and `notebook/inference.py` at `f91db411`):

```python
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask   # inference.py sets CUDA_HOME = $CONDA_PREFIX at import
inference = Inference("checkpoints/hf/pipeline.yaml", compile=False)
image = load_image(".../image.png")            # (H, W, 3|4) uint8
mask = load_single_mask(".../folder", index=14)  # (H, W) bool: PNG > 0, last channel if RGBA
output = inference(image, mask, seed=42)
output["gs"].save_ply("splat.ply")
```

`Inference.__call__` embeds the mask as the alpha channel (`merge_mask_to_rgba`) and
calls `InferencePipelinePointMap.run(image, None, seed, stage1_only=False,
with_mesh_postprocess=False, with_texture_baking=False, with_layout_postprocess=False,
use_vertex_color=True, pointmap=None)`. `run` first computes a pointmap with the
depth model (MoGe), samples the sparse structure (stage 1), decodes the pose
(`rotation`, `translation`, `scale`), samples the structured latent (stage 2),
decodes to the formats in `pipeline.yaml` (`["gaussian", "mesh"]` by default) and
returns one dict with `gs` (the `GaussianModel`), `glb` (a trimesh built by
`postprocessing_utils.to_glb` from the mesh and the gaussians' colours; `None` when
the mesh decoder is not in the formats), `gaussian`, `mesh`, `pointmap`,
`pointmap_colors` and the pose keys. The hydra-instantiated config is checked
against a whitelist (`sam3d_objects`, `torch`, `torchvision`, `moge`), so adding a
module to the pipeline config needs a whitelist entry. Checkpoints: the HF repo's
`checkpoints/` folder (`pipeline.yaml`, `ss_*`, `slat_*` .ckpt/.yaml) moved to
`checkpoints/hf/`, exactly as `doc/setup.md` does with `hf download` and `mv`.

**SAM 3D Body** (`notebook/utils.py`, `sam_3d_body/sam_3d_body_estimator.py`,
`sam_3d_body/build_models.py` at `b5c765a0`):

```python
model, model_cfg = load_sam_3d_body("checkpoints/.../model.ckpt", device="cuda", mhr_path=".../assets/mhr_model.pt")
estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg,
                               human_detector=HumanDetector(name="vitdet", device="cuda"),   # tools/build_detector.py
                               human_segmentor=None,
                               fov_estimator=FOVEstimator(name="moge2", device="cuda"))     # tools/build_fov_estimator.py
outputs = estimator.process_one_image(img_rgb, bbox_thr=0.8)      # list, one dict per detected person
faces = estimator.faces                                           # (F, 3) from model.head_pose.faces
```

Each dict carries `bbox (4,) xyxy`, `focal_length`, `pred_vertices (V, 3)`,
`pred_keypoints_3d (K, 3)`, `pred_keypoints_2d (K, 2)`, `pred_cam_t (3,)`,
`pred_pose_raw`, `global_rot`, `body_pose_params`, `hand_pose_params`,
`scale_params`, `shape_params`, `expr_params`, `pred_joint_coords`,
`pred_global_rots`, `mhr_model_params`, and in `full` inference `lhand_bbox`,
`rhand_bbox`. `load_sam_3d_body` finds `model_config.yaml` next to (or one level
above) the checkpoint. Keypoint names for the first 70 are in
`sam_3d_body/metadata/mhr70.py`: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8
elbows, 9-10 hips, 11-12 knees, 13-14 ankles, 15-17 left big toe / small toe /
heel, 18-20 right ditto, 21-41 right hand, 42-62 left hand, 63-69 olecranon,
cubital fossa, acromion, neck. `process_one_image(ndarray)` expects **RGB** (a
path is read as BGR and converted); with the detector and
`default_to_full_image=False` an image without a person returns `[]`. The ViTDet-H
pickle comes from `dl.fbaipublicfiles.com` through detectron2's checkpointer and
MoGe-2 from `Ruicheng/moge-2-vitl-normal`; the image build constructs the estimator
once so both are cached.

Known risks: `flash_attn==2.8.3` may have no prebuilt wheel for torch 2.5.1 /
Python 3.11 and fall back to a long source build; `requirements.txt` also pins
`bpy==4.3.0` (Python 3.11 only) and `sagemaker`, which is why the image is Python
3.11 and why the `.[dev]` step is slow; MoGe is pinned at `a8c37341` (the commit
sam-3d-objects pins) in both images because MoGe's main is now MoGe-3.

## Licence: the SAM License

Both repositories ship the same `LICENSE` ("SAM License, Last Updated: November 19,
2025"; the two files are byte-identical) and the Hugging Face checkpoints are under
it too. It is a permissive, Meta-drafted licence with no commercial-use condition,
no user threshold and no acceptable-use policy. The clauses that matter:

- **Grant.** "You are granted a non-exclusive, worldwide, non-transferable and
  royalty-free limited license under Meta's intellectual property or other rights
  owned by Meta embodied in the SAM Materials to use, reproduce, distribute, copy,
  create derivative works of, and make modifications to the SAM Materials."
  (§1.a) There is no field-of-use restriction: commercial use, including a paid
  product such as Locaish, is within the grant. A patent licence comes with it
  (§1.a.i) and terminates if you sue over patents in the materials.
- **Redistribution.** "If you distribute or make the SAM Materials, or any
  derivative works thereof, available to a third party, you may only do so under
  the terms of this Agreement and you shall provide a copy of this Agreement with
  any such SAM Materials." (§1.b.i) The Modal image contains the weights and code
  but is private to the workspace; shipping a container or fine-tune to customers
  would need the licence file alongside. *Outputs* (splats, meshes) are not "SAM
  Materials" under the definition ("models, software and algorithms, including
  machine-learning model code, trained model weights, inference-enabling code...")
  and carry no such obligation.
- **Attribution.** Only for publications: "If you submit for publication the
  results of research you perform on, using, or otherwise in connection with SAM
  Materials, you must acknowledge the use of SAM Materials in your publication."
  (§1.b.ii) No product attribution is required; the hackathon write-up should
  name them anyway.
- **Use restrictions.** Compliance with law including trade controls and privacy
  law (§1.b.iii); no reverse engineering (§1.b.iv); and "You agree not to use, or
  permit others to use, SAM Materials for any activities subject to the
  International Traffic in Arms Regulations (ITAR) or end uses prohibited by Trade
  Controls, including those related to military or warfare purposes, nuclear
  industries or applications, espionage, or the development or use of guns or
  illegal weapons." (§1.b.v) You must not be a sanctions target.
- **Ownership.** "with respect to any derivative works and modifications of the
  SAM Materials that are made by you, as between you and Meta, you are and will be
  the owner of such derivative works and modifications." (§5.a) Suing Meta or
  anyone over IP in the materials or their outputs terminates the licence, and you
  indemnify Meta against third-party claims from your use (§5.b).
- **Warranty and liability.** As-is, no warranty, including for "any output and
  results" (§3); no liability for lost profits or indirect damages (§4). Meta may
  terminate on breach, after which "you shall delete and cease use of the SAM
  Materials" (§6); it may amend the licence "provided that they are similar in
  spirit" and continued use is acceptance (§8). California law and courts (§7).

Privacy note for the body endpoint: §1.b.iii binds Locaish to "applicable privacy
and data protection laws", and a headshot is personal data. The endpoint keeps
nothing (no disk, no log of the image) and the Modal container is torn down after
`scaledown_window`; the product should say so where it asks for the headshot.
