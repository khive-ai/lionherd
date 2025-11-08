"""
Capability-based knowledge system for agent specialization.

This module implements atomic knowledge units aligned with Lion ecosystem
capability-based security principles. Each atom represents an unforgeable
capability grant that provides agents with:

- Knowledge content (what agent KNOWS)
- Actionable affordances (what agent CAN DO)
- Compositional semantics (how atoms COMBINE)
- Graph relationships (how atoms RELATE)

Future: Cryptographic sealing with HMAC for unforgeable capability tokens.
"""

from lionherd.capabilities.base import (
    AtomApplicability,
    AtomCategory,
    AtomConstraints,
    AtomKnowledge,
    AtomNode,
    AtomRelationships,
    AtomRights,
)

__all__ = [
    "AtomApplicability",
    "AtomCategory",
    "AtomConstraints",
    "AtomKnowledge",
    "AtomNode",
    "AtomRelationships",
    "AtomRights",
]
