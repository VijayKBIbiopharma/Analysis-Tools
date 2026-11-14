"""Which modules a loaded file can actually drive.

Every page used to carry its own precondition check, written inline. That is
fine for the page — it knows what it needs — but nothing else did. The sidebar
offered all eight modules whatever you had loaded, so picking the capability
example and then opening Cell line development produced a red error, which
reads as a broken tool rather than as a mismatched file. Choosing wrongly was
the only way to find out you had chosen wrongly.

The conditions live here instead, once, and three things read them: the
sidebar marks what will not work, the pages explain *why* rather than only
*that*, and the tests check every shipped example against the module it is
offered for. One source of truth, so the sidebar cannot drift away from what
the pages actually do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from analysis_tools.roles import (ATTRIBUTE_2, ATTRIBUTE_3PLUS, IDENTIFIER,
                                  VARIABLES)

GROUPING = (ATTRIBUTE_2, ATTRIBUTE_3PLUS)


def _continuous(roles) -> list:
    return [r for r in roles if r.role == VARIABLES]


def _grouping(roles) -> list:
    return [r for r in roles if r.role in GROUPING]


def _analysable(roles) -> list:
    return [r for r in roles if r.analysable]


@dataclass
class Requirement:
    #: True when a file with these roles can drive the module.
    check: Callable
    #: What it needs, in the words a user would use.
    needs: str
    #: False for the modules that work before any file is loaded.
    needs_data: bool = True


REQUIREMENTS = {
    "Explore": Requirement(
        check=lambda roles: True,
        needs="any file at all"),

    "Compare groups": Requirement(
        check=lambda roles: len(_analysable(roles)) >= 2,
        needs="at least two usable columns — something measured, and "
              "something to split it by"),

    "Capability": Requirement(
        check=lambda roles: len(_continuous(roles)) >= 1,
        needs="a continuous measurement to compare against a specification"),

    "Measurement systems": Requirement(
        check=lambda roles: (len(_continuous(roles)) >= 1
                             and len(_grouping(roles)) + len(
                                 [r for r in roles if r.role == IDENTIFIER]) >= 2),
        needs="a measurement, plus two columns identifying what was measured "
              "and who measured it — parts and operators"),

    "Relationships": Requirement(
        check=lambda roles: (len(_continuous(roles)) >= 1
                             and len(_analysable(roles)) >= 2),
        needs="a continuous response and at least two usable inputs"),

    "Design of experiments": Requirement(
        check=lambda roles: True,
        needs="nothing to build a design; a completed run sheet to analyse one",
        needs_data=False),

    "Cell line development": Requirement(
        check=lambda roles: (
            (len(_continuous(roles)) >= 1 and len(_grouping(roles)) >= 1)
            or len(_continuous(roles)) >= 2),
        needs="either a continuous response with the construct parts beside "
              "it, or two or more continuous measurements"),

    "Learn": Requirement(
        check=lambda roles: True,
        needs="nothing",
        needs_data=False),
}


def suitable(module: str, roles: Optional[Sequence]) -> bool:
    """Can this module do anything with what is loaded?"""
    requirement = REQUIREMENTS.get(module)
    if requirement is None:
        return True
    if not requirement.needs_data:
        return True
    if not roles:
        return False
    return bool(requirement.check(roles))


def suitable_modules(roles: Optional[Sequence]) -> list:
    return [name for name in REQUIREMENTS if suitable(name, roles)]


def explain(module: str, roles: Optional[Sequence],
            source_name: str = "this file") -> str:
    """Why this module cannot run, and where to go instead.

    Deliberately not phrased as a failure. Nothing has gone wrong: the file
    simply does not contain what this particular question needs, and the
    useful thing to say is which questions it *does* answer.
    """
    requirement = REQUIREMENTS.get(module)
    needs = requirement.needs if requirement else "different columns"
    others = [n for n in suitable_modules(roles)
              if n != module and REQUIREMENTS[n].needs_data]

    message = ("**%s has not got what %s needs.** It wants %s.\n\n"
               % (source_name, module, needs))
    if others:
        message += ("This file *will* drive %s. See **Explore** for how each "
                    "column was read — a column classified as a label rather "
                    "than a measurement is the usual reason."
                    % _list(["**%s**" % n for n in others]))
    else:
        message += ("See **Explore** for how each column was read. If a "
                    "measurement was classified as a label, that is the "
                    "usual reason.")
    return message


def _list(items) -> str:
    items = list(items)
    if len(items) == 1:
        return items[0]
    return "%s and %s" % (", ".join(items[:-1]), items[-1])
