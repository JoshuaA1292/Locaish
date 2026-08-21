"""Phase 2: the location scout's job, done against the twin.

Phase 1 answers "what shape is this room". This package answers the questions
someone would otherwise drive out to ask: where can the camera go, what lens
does that corner need, will the dolly turn, can the operator see the actor from
there, does the room ring.

Every answer traces to twin geometry or to a published equipment dimension, and
anything resting on a class-typical figure rather than a measured one says so.
"""

from __future__ import annotations

__all__ = ["equipment", "optics", "space", "moves", "acoustics", "report"]
