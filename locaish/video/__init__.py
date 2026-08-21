"""Video front-end: a hand-held sweep becomes a metric point cloud.

The rest of Locaish takes a point cloud and asks what room it is. This package
answers the question one step earlier -- what point cloud is this *video* --
and hands the result to the same ingest pipeline a Polycam export goes through,
so a twin built from video is inspected, QA'd and measured by exactly the same
code, with the same right to be distrusted.
"""

from __future__ import annotations

from .frames import VIDEO_EXTENSIONS, FrameSet, VideoError, VideoInfo, extract_frames, probe

__all__ = [
    "VIDEO_EXTENSIONS",
    "FrameSet",
    "VideoError",
    "VideoInfo",
    "extract_frames",
    "probe",
    "reconstruct_video",
]


def __getattr__(name: str):
    # Deferred so that importing locaish.video for `probe` alone does not drag
    # in torch, which costs seconds and half a gigabyte of RSS.
    if name == "reconstruct_video":
        from .reconstruct import reconstruct_video

        return reconstruct_video
    raise AttributeError(name)
