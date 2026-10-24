"""Meta's models in Locaish: the semantic and agentic layer over the measured twin.

The reconstruction stack (COLMAP, OpenMVS, MoGe-2, Brush) stays what it is;
nothing in this package measures. What lives here is every place a model
*reads, names, judges or plans*: Muse Spark naming what the frames show and
scoring a rendered frame as a DP would, Muse Wonder running the scout and the
rehearsal, Muse Image finishing a frame photoreal, SAM 3.1 masking the people
who walked through the scan, and the open-weight SAM 3D models lifting a sofa
or an actor into the twin. Each output is a layer labelled with its source,
and the one that touches a measurement -- the stature scale anchor -- enters
through the same inverse-variance fusion as the other anchors.

`client.MetaClient` is the only thing that talks to the network; everything
else takes a client so the tests can hand in `client.FakeMetaClient`.
"""

from .client import (  # noqa: F401
    DEFAULT_MODELS,
    Detection,
    FakeMetaClient,
    MetaClient,
    MetaError,
    configured,
    model_for,
)
