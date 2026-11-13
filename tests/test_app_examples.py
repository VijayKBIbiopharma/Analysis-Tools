"""The bundled examples.

A tool that cannot be opened without first finding a file is a tool most
people never open, so the sidebar offers these. That makes them part of the
application rather than documentation: an example that has been renamed, or
that lands the user on an error page, is a broken front door.
"""

from __future__ import annotations

import pytest

from analysis_tools.app.main import EXAMPLES, _examples_directory
from analysis_tools.io import read_delimited
from analysis_tools.roles import classify_column

NAMED = [(label, filename) for label, filename in EXAMPLES.items() if filename]


def test_the_picker_offers_a_way_out():
    """The first option must be the one that clears the selection, or a user
    who picks an example can never get back to their own upload."""
    first = list(EXAMPLES.values())[0]
    assert first is None


@pytest.mark.parametrize("label,filename", NAMED, ids=[f for _, f in NAMED])
def test_every_offered_example_exists(label, filename):
    assert (_examples_directory() / filename).is_file(), (
        "the sidebar offers %r but %s is not in examples/" % (label, filename))


@pytest.mark.parametrize("label,filename", NAMED, ids=[f for _, f in NAMED])
def test_every_offered_example_loads_and_classifies(label, filename):
    """Read it the way the sidebar does — from bytes, never a path."""
    data = (_examples_directory() / filename).read_bytes()
    table = read_delimited(data, source_name=filename)
    assert len(table.frame) > 0, "%s is empty" % filename
    assert len(table.frame.columns) >= 2, "%s has nothing to compare" % filename

    roles = [classify_column(c, table.frame[c]) for c in table.frame.columns]
    assert any(r.analysable for r in roles), (
        "%s classifies to nothing analysable, so the module it is offered for "
        "would show an error" % filename)


def test_every_example_file_is_offered():
    """The converse: a file sitting in examples/ that the picker never shows
    is one nobody will find."""
    on_disk = {p.name for p in _examples_directory().glob("*.csv")}
    offered = {f for f in EXAMPLES.values() if f}
    unoffered = on_disk - offered
    # catalogue.csv and programs.csv are loaded inside the vector page, not
    # as the main table, so they are not front-door examples.
    assert unoffered <= {"catalogue.csv", "programs.csv"}, (
        "these examples ship but are never offered: %s" % sorted(unoffered))


def test_no_example_carries_anything_that_looks_like_real_data():
    """Synthetic only. A shipped example is public the moment the repository
    is, and a file named for a real programme would be a disclosure."""
    import re
    suspicious = re.compile(
        r"\b(kbi|biopharma|confidential|proprietary|client|customer)\b", re.I)
    for path in _examples_directory().glob("*.csv"):
        text = path.read_text(errors="ignore")
        found = suspicious.findall(text)
        assert not found, "%s contains %s" % (path.name, set(found))
