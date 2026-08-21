"""Scan-side analysis: turning a raw export into something that means metres.

The modules here sit between the file readers and the geometry kernel. They
answer the questions that are about the *capture* rather than about the shapes:
what unit the numbers are in, what the room's architecture is, and whether the
result is trustworthy enough to be called a 1:1 twin.
"""

from __future__ import annotations
