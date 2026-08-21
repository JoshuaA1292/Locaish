"""Session-scoped ingests, so the pipeline runs once per fixture rather than
once per test. Tests that need different ingest options call `ingest_fixture`
directly instead of mutating a cached twin."""

from __future__ import annotations

import pytest

from support import ingest_fixture


@pytest.fixture(scope="session")
def clean():
    return ingest_fixture("clean")


@pytest.fixture(scope="session")
def tilted():
    return ingest_fixture("tilted")


@pytest.fixture(scope="session")
def centimetres():
    return ingest_fixture("centimetres")


@pytest.fixture(scope="session")
def inches():
    return ingest_fixture("inches")


@pytest.fixture(scope="session")
def tall():
    return ingest_fixture("tall")


@pytest.fixture(scope="session")
def noceiling():
    return ingest_fixture("noceiling")
