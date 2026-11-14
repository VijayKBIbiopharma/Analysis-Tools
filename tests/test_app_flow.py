"""Driving the app itself: every example, every module.

The reported fault was "if you pick the wrong example for a tool you get
errors". Sweeping all of it headlessly showed there were no uncaught
exceptions — what a user meets is a red box with a 🚫 on it, which reads as a
broken tool rather than as a file that does not suit the question. Both parts
are fixed: picking an example moves you to the module it demonstrates, and
the refusals explain what the file *can* do.

These tests hold that. They drive the real Streamlit app rather than calling
functions underneath it, which is the only way to catch a guard clause that
refers to a variable not in scope — a NameError raised only when a user takes
that path, invisible to every unit test, and looking to them exactly like the
errors they reported.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from analysis_tools.app.fitness import REQUIREMENTS, suitable
from analysis_tools.app.main import EXAMPLES, MODULES

APP = "app.py"
EXAMPLE_LABELS = [label for label, (f, _) in EXAMPLES.items() if f]
BUILT = [name for name, _, built in MODULES if built]
COMBINATIONS = [(e, m) for e in EXAMPLE_LABELS for m in BUILT]


def state(app, key):
    """AppTest's session_state has no .get() — it reads "get" as a key."""
    try:
        return app.session_state[key]
    except KeyError:
        return None


def started():
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    return app


def with_example(label):
    app = started()
    app.sidebar.selectbox[0].select(label).run()
    return app


def in_module(app, module):
    radio = app.sidebar.radio[0]
    option = next(o for o in radio.options if o.startswith(module))
    return radio.set_value(option).run()


def test_the_app_starts_with_nothing_loaded():
    app = started()
    assert not app.exception
    assert app.sidebar.selectbox, "no example picker in the sidebar"
    assert app.sidebar.radio, "no module picker in the sidebar"


@pytest.mark.parametrize("label,module", COMBINATIONS,
                         ids=["%s/%s" % (e.split(" (")[0], m)
                              for e, m in COMBINATIONS])
def test_no_combination_raises(label, module):
    """42 combinations, none of which may throw.

    An uncaught exception in Streamlit renders a traceback in the page. That
    is the only thing here that is genuinely a bug; a module politely saying
    the file does not suit it is correct behaviour.
    """
    app = in_module(with_example(label), module)
    assert not app.exception, (
        "%s in %s raised: %s"
        % (label, module, app.exception[0].value if app.exception else ""))


@pytest.mark.parametrize("label", EXAMPLE_LABELS,
                         ids=[e.split(" (")[0] for e in EXAMPLE_LABELS])
def test_picking_an_example_lands_you_in_its_module(label):
    """The fix for the reported fault. Choosing the capability example while
    standing in Cell line development used to leave you there, reading a
    refusal."""
    expected = EXAMPLES[label][1]
    app = with_example(label)
    assert not app.exception
    assert app.sidebar.radio[0].value.startswith(expected), (
        "%s should land in %s, landed in %r"
        % (label, expected, app.sidebar.radio[0].value))


@pytest.mark.parametrize("label", EXAMPLE_LABELS,
                         ids=[e.split(" (")[0] for e in EXAMPLE_LABELS])
def test_each_example_actually_works_in_its_own_module(label):
    """Landing there is no use if the module then refuses the file."""
    filename, module = EXAMPLES[label]
    app = in_module(with_example(label), module)
    assert not app.exception
    roles = state(app, "roles")
    assert roles, "%s produced no column classification" % filename
    assert suitable(module, roles), (
        "%s is offered for %s but does not meet its requirement: %s"
        % (filename, module, REQUIREMENTS[module].needs))


def test_switching_module_after_picking_an_example_sticks():
    """Steering must happen on a change of example, not on every rerun, or
    the module picker becomes unusable: every click springs back."""
    app = with_example("Capability run (Cp, Cpk, stability)")
    app = in_module(app, "Relationships")
    assert app.sidebar.radio[0].value.startswith("Relationships")
    # a rerun with no new selection must not drag it back
    app.run()
    assert app.sidebar.radio[0].value.startswith("Relationships")


def test_the_sidebar_marks_what_a_file_cannot_drive():
    """Marked rather than hidden: a module that disappears makes the tool
    look like it does less than it does."""
    app = with_example("Capability run (Cp, Cpk, stability)")
    labels = app.sidebar.radio[0].options
    roles = state(app, "roles")
    for name in BUILT:
        label = next(o for o in labels if o.startswith(name))
        if not suitable(name, roles):
            assert "not for this file" in label, (
                "%s cannot take this file but is not marked" % name)


def test_choosing_no_example_clears_the_table():
    app = with_example("Capability run (Cp, Cpk, stability)")
    assert state(app, "table") is not None
    app.sidebar.selectbox[0].select("— none, I will upload my own —").run()
    assert state(app, "table") is None
