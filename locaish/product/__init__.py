"""The product layer: what a production company pays for.

The engine under ``locaish/scan``, ``locaish/film`` and ``locaish/agent``
turns a capture into a twin and a twin into measurements. This package turns
measurements into the three things a commercial production actually buys
during a bid:

* a **location package** per room -- the document a scout hands in, with the
  sun on the shoot date, the dimensions, what fits and a link into the twin
  (``package.py``);
* a **project** -- the job being bid, its brief, its shoot date and the
  candidate locations, ranked by how many of the scene's shots each room
  physically holds (``store.py``, ``fit.py``);
* a **share link** -- a read-only URL a director can open on a phone in
  another city without an account (``store.py``).

Everything here is files on disk beside the jobs: no database of its own,
because the product has to run on a scout's laptop as well as on Cloud Run.
"""
