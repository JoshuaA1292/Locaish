# Dense stereo on a GCP GPU

COLMAP's PatchMatch stereo — the best classical densifier, with geometric
consistency across views — is CUDA-only. On a Mac the pipeline falls back to
OpenMVS on the CPU (~10 minutes a room) or block matching (worse). Pointing the
pipeline at a GCP GPU instance runs the same classical stage in ~90 seconds on
an L4, and puts Google Cloud into the compute path of every twin rather than
just around it.

Nothing about the pipeline changes: undistortion still runs locally, the
undistorted workspace is shipped up with `gcloud compute scp`,
`patch_match_stereo` + `stereo_fusion` run remotely, and `fused.ply` comes
back to exactly where a local CUDA build would have written it. If the
instance is stopped or unreachable, the reconstruction warns and falls back to
the local CPU densifier — a dead GPU degrades the twin, never kills it.

## One-time setup

```sh
# An L4 is the sweet spot; a T4 works and is cheaper.
gcloud compute instances create locaish-dense \
    --zone=us-central1-a \
    --machine-type=g2-standard-4 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=common-cu124-debian-11 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE

# On the instance: COLMAP with CUDA. The prebuilt docker image is simplest.
gcloud compute ssh locaish-dense --zone=us-central1-a \
    --command='docker pull colmap/colmap:latest'
```

## Enabling it

```sh
export LOCAISH_GPU_INSTANCE=locaish-dense
export LOCAISH_GPU_ZONE=us-central1-a
# Only needed when colmap is not directly on the instance's PATH:
export LOCAISH_GPU_COLMAP='docker run --rm --gpus all -v /tmp:/tmp colmap/colmap:latest colmap'
# Optional, when the instance is not in the active gcloud project:
export LOCAISH_GPU_PROJECT=my-project
```

With those set, `locaish ingest sweep.mp4 ...` uses the remote GPU for the
dense stage automatically; the manifest records `"stereo":
"patchmatch-remote"`. Unset the variables (or stop the instance) to fall back
to the local path.

Remember the instance bills while running:

```sh
gcloud compute instances stop locaish-dense --zone=us-central1-a
```
