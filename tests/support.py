"""The one entry point the tests use to run a synthetic room through the real pipeline.

Ingesting a fixture runs the entire pipeline, which takes seconds, so the
result is cached for the whole session and shared across tests. The cache is
keyed on the fixture name only -- if a test needs different ingest options it
should call `ingest_fixture` itself rather than mutating the cached twin.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from locaish import fixtures
from locaish.scan.ingest import IngestOptions, ingest

_CACHE: dict[str, object] = {}


def ingest_fixture(name: str, **option_overrides):
    """Run the real pipeline on a named synthetic room, via a real PLY file.

    Going through the file rather than handing the pipeline an in-memory cloud
    is deliberate: the reader is part of what we are claiming works, and a
    round trip through PLY is the closest cheap analogue of a phone export.
    """
    key = name + repr(sorted(option_overrides.items()))
    if key in _CACHE:
        return _CACHE[key]

    from locaish.formats.ply import write_ply

    fx = fixtures.build(name)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / f"{name}.ply"
        write_ply(src, fx.points, None)
        opts = IngestOptions(name=f"test-{name}", seed=0, **option_overrides)
        result = ingest(src, opts)
    _CACHE[key] = (result, fx)
    return _CACHE[key]


