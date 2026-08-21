"""Self-contained HTML viewer for a `Twin`.

`render_html` writes one file that opens from `file://` with no network and no
build step; `twin_to_payload` returns the same data as a plain dict so a server
can hand it to a browser without touching the filesystem. The page itself lives
in `template.html` as hand-written WebGL2, which is where to go to change how
the twin looks -- this package only injects data into it.
"""

from __future__ import annotations

from .build import PAYLOAD_VERSION, PLANE_TINTS, render_html, twin_to_payload

__all__ = ["render_html", "twin_to_payload", "PAYLOAD_VERSION", "PLANE_TINTS"]
