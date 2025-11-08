"""
Tests for capability-based atomic knowledge system.

Rationale:
----------
These tests validate the atomic knowledge schema against the design principles:

1. Affordance Structure: Atoms provide situation/action/outcome triads
2. Dual-Space Consistency: Embedding + graph relationships work together
3. Atomicity Validation: Heuristics detect incorrectly sized atoms
4. Partial Order: Relationships form valid capability chains
5. Prompt Generation: Atoms produce actionable prompts for agents

Design Philosophy Tests:
------------------------
Rather than exhaustive unit tests, these validate architectural invariants
that align with Lion ecosystem capability-based security principles.

Reference:
- Lion capability formal verification: /lionrust/docs/proofs/v2/layer0/L0-CAP-1/
- Affordance theory: Gibson's ecological psychology
"""

import pytest

from lionherd.capabilities import (
    AtomApplicability,
    AtomCategory,
    AtomKnowledge,
    AtomNode,
    AtomRelationships,
)


class TestAtomicKnowledgeSchema:
    """Test atomic knowledge schema correctness."""

    def test_atom_affordance_structure(self):
        """
        Validate atoms provide complete affordance structure (situation/action/outcome).

        Rationale:
        Affordance theory (Gibson) requires complete specification of when/how/what
        for agents to perceive and act on knowledge. Incomplete atoms cannot be
        independently actionable.
        """
        atom = AtomNode(
            ln_id="test-001",
            atom_ref="test/affordance_atom",
            category=AtomCategory.PATTERN,
            applicability=AtomApplicability(
                situation="Test situation requiring action",
                context=["test_context"],
                anti_patterns=["avoid_this"],
            ),
            knowledge=AtomKnowledge(
                what="Test concept definition",
                why="Test rationale",
                how={
                    "steps": ["step1", "step2"],
                    "tools": ["tool1"],
                    "patterns": ["pattern1"],
                },
                outcomes={
                    "achieves": ["goal1", "goal2"],
                    "enables": ["capability1"],
                    "metrics": {"metric1": "value1"},
                },
                trade_offs={
                    "benefits": ["benefit1"],
                    "costs": ["cost1"],
                    "when_to_skip": ["skip_condition"],
                },
            ),
        )

        # Validate affordance completeness
        assert atom.applicability.situation  # WHEN
        assert atom.knowledge.how["steps"]  # HOW (action)
        assert atom.knowledge.outcomes["achieves"]  # WHAT (outcome)

        # Validate actionability
        validation = atom.validate_atomicity()
        assert validation["self_contained"], "Atom must be self-contained"
        assert validation["actionable"], "Atom must have actionable procedure"

    def test_atom_partial_order_relationships(self):
        """
        Validate relationships form valid partial order (capability chains).

        Rationale:
        Lion capability system is mathematically a partial order (C₁ ≤ C₂ if R₁ ⊆ R₂).
        Atom relationships must respect this ordering for compositional reasoning.

        Invariants:
        - If A enables B, then A ≤ B (A is lower in capability order)
        - If A requires B, then B ≤ A (B is prerequisite)
        - Conflicts break the order (incomparable)
        """
        atom = AtomNode(
            ln_id="test-002",
            atom_ref="test/ordered_atom",
            category=AtomCategory.PATTERN,
            applicability=AtomApplicability(
                situation="Test ordering",
                context=[],
                anti_patterns=[],
            ),
            knowledge=AtomKnowledge(
                what="Test",
                why="Test",
                how={"steps": ["test"]},
                outcomes={"achieves": ["test"]},
                trade_offs={"benefits": [], "costs": [], "when_to_skip": []},
            ),
            relationships=AtomRelationships(
                enables=["higher_capability"],  # This atom enables higher one
                requires=["lower_prerequisite"],  # This requires lower one
                conflicts=["incompatible_atom"],  # Breaks order
            ),
        )

        # Validate partial order structure
        assert atom.relationships.enables, "Should enable higher capabilities"
        assert atom.relationships.requires, "Should require prerequisites"
        assert atom.relationships.conflicts, "Should have conflicts"

        # Validate no circular dependencies (would break partial order)
        assert (
            "test/ordered_atom" not in atom.relationships.enables
        ), "Cannot enable self"
        assert (
            "test/ordered_atom" not in atom.relationships.requires
        ), "Cannot require self"

    def test_atom_prompt_generation(self):
        """
        Validate atoms generate actionable prompts for agent consumption.

        Rationale:
        Atoms must be renderable as 100-500 token prompts that agents can directly
        use for reasoning and generation. This tests the prompt interface.
        """
        atom = AtomNode(
            ln_id="test-003",
            atom_ref="patterns/test_pattern",
            category=AtomCategory.PATTERN,
            applicability=AtomApplicability(
                situation="When building test systems",
                context=["testing", "automation"],
                anti_patterns=["Manual testing only"],
            ),
            knowledge=AtomKnowledge(
                what="Automated testing pattern",
                why="Ensures code correctness at scale",
                how={
                    "steps": ["Write tests", "Run CI", "Monitor coverage"],
                    "tools": ["pytest", "coverage.py"],
                    "patterns": ["TDD", "BDD"],
                },
                outcomes={
                    "achieves": ["Automated quality assurance"],
                    "enables": ["Continuous deployment"],
                    "metrics": {"coverage": ">80%", "runtime": "<5min"},
                },
                trade_offs={
                    "benefits": ["Prevents regressions", "Enables refactoring"],
                    "costs": ["Initial time investment", "Maintenance overhead"],
                    "when_to_skip": ["Prototypes", "Throwaway scripts"],
                },
            ),
            relationships=AtomRelationships(
                enables=["capabilities/can_deploy_continuously"],
                requires=["best_practices/code_organization"],
                complements=["tools/ci_cd_pipeline"],
            ),
        )

        prompt = atom.to_prompt()

        # Validate prompt structure
        assert atom.atom_ref in prompt, "Should include atom reference"
        assert atom.applicability.situation in prompt, "Should include situation"
        assert atom.knowledge.what in prompt, "Should include concept definition"
        assert "Implementation" in prompt, "Should have implementation section"
        assert "Trade-offs" in prompt, "Should have trade-offs section"

        # Validate token count (rough approximation)
        token_count = len(prompt.split())
        assert (
            100 <= token_count <= 500
        ), f"Prompt should be 100-500 tokens, got {token_count}"

    def test_atom_atomicity_validation(self):
        """
        Validate atomicity heuristics detect correctly sized atoms.

        Rationale:
        Atoms must be "atomic" in the sense of being:
        1. Self-contained (no external dependencies to understand)
        2. Actionable (provides procedure, not just description)
        3. Graph connected (has 2-6 relationships, not isolated)
        4. Semantically positioned (has embedding)
        5. Prompt-sized (generates reasonable token count)

        These heuristics prevent atoms that are too fine-grained or too coarse.
        """
        # Valid atom (should pass all heuristics)
        valid_atom = AtomNode(
            ln_id="test-004",
            atom_ref="test/valid_atom",
            category=AtomCategory.PATTERN,
            embedding=[0.1] * 768,  # 768-dim vector
            applicability=AtomApplicability(
                situation="Valid situation",
                context=["context1"],
                anti_patterns=[],
            ),
            knowledge=AtomKnowledge(
                what="Valid concept",
                why="Valid rationale",
                how={"steps": ["step1", "step2", "step3"]},  # Has procedure
                outcomes={"achieves": ["goal"]},
                trade_offs={"benefits": ["b1"], "costs": ["c1"], "when_to_skip": []},
            ),
            relationships=AtomRelationships(
                enables=["atom1", "atom2"],
                requires=["atom3"],
                complements=["atom4"],
                # Total: 4 relationships (within 2-6 range)
            ),
        )

        validation = valid_atom.validate_atomicity()

        assert validation["self_contained"], "Should be self-contained"
        assert validation["actionable"], "Should have actionable procedure"
        assert validation["graph_connected"], "Should have 2-6 relationships"
        assert validation["embedding_exists"], "Should have 768-dim embedding"
        assert validation["prompt_sized"], "Should generate reasonable prompt"

    def test_atom_dual_space_consistency(self):
        """
        Validate atoms exist consistently in both embedding and graph spaces.

        Rationale:
        Atoms must work in BOTH computational spaces:
        - Embedding space (continuous): Vector similarity for semantic search
        - Graph space (discrete): Typed relationships for compositional reasoning

        Graph edges should correlate with embedding proximity for consistency.
        """
        atom = AtomNode(
            ln_id="test-005",
            atom_ref="test/dual_space_atom",
            category=AtomCategory.PATTERN,
            embedding=[0.5] * 768,  # Semantic position
            applicability=AtomApplicability(
                situation="Test",
                context=[],
                anti_patterns=[],
            ),
            knowledge=AtomKnowledge(
                what="Test",
                why="Test",
                how={"steps": ["test"]},
                outcomes={"achieves": ["test"]},
                trade_offs={"benefits": [], "costs": [], "when_to_skip": []},
            ),
            relationships=AtomRelationships(
                enables=["related_atom_1"],  # Graph edges
                complements=["related_atom_2"],
            ),
        )

        # Validate dual-space presence
        assert len(atom.embedding) == 768, "Must have semantic position (embedding)"
        assert (
            len(atom.relationships.enables) + len(atom.relationships.complements) > 0
        ), "Must have graph relationships"

        # Note: Actual consistency check (graph proximity ≈ embedding similarity)
        # requires multiple atoms and distance calculations - deferred to integration tests


class TestAtomComposition:
    """Test atom composition and capability chains."""

    def test_capability_chain_transitivity(self):
        """
        Validate capability chains are transitive (if A enables B, B enables C, then A enables C).

        Rationale:
        Category theory foundation requires compositional reasoning to preserve
        properties. Capability chains must be transitive for graph traversal to work.
        """
        # This is a design invariant test - actual implementation would require
        # graph traversal logic (deferred to future integration)
        base_atom = AtomNode(
            ln_id="base",
            atom_ref="base/capability",
            category=AtomCategory.PATTERN,
            applicability=AtomApplicability(
                situation="Base capability", context=[], anti_patterns=[]
            ),
            knowledge=AtomKnowledge(
                what="Base",
                why="Base",
                how={"steps": ["base"]},
                outcomes={"achieves": ["base"]},
                trade_offs={"benefits": [], "costs": [], "when_to_skip": []},
            ),
            relationships=AtomRelationships(
                enables=["intermediate/capability"]  # Base enables intermediate
            ),
        )

        intermediate_atom = AtomNode(
            ln_id="intermediate",
            atom_ref="intermediate/capability",
            category=AtomCategory.PATTERN,
            applicability=AtomApplicability(
                situation="Intermediate", context=[], anti_patterns=[]
            ),
            knowledge=AtomKnowledge(
                what="Intermediate",
                why="Intermediate",
                how={"steps": ["intermediate"]},
                outcomes={"achieves": ["intermediate"]},
                trade_offs={"benefits": [], "costs": [], "when_to_skip": []},
            ),
            relationships=AtomRelationships(
                requires=["base/capability"],  # Intermediate requires base
                enables=["advanced/capability"],  # Intermediate enables advanced
            ),
        )

        # Validate chain structure
        assert "intermediate/capability" in base_atom.relationships.enables
        assert "base/capability" in intermediate_atom.relationships.requires
        assert "advanced/capability" in intermediate_atom.relationships.enables

        # Transitive chain: base → intermediate → advanced
        # Graph traversal would find: base enables advanced (transitively)


@pytest.mark.parametrize(
    "category,expected_type",
    [
        (AtomCategory.PATTERN, "pattern"),
        (AtomCategory.TOOL, "tool"),
        (AtomCategory.BEST_PRACTICE, "best_practice"),
        (AtomCategory.METRIC, "metric"),
    ],
)
def test_atom_categories(category, expected_type):
    """
    Validate atom categories are correctly defined.

    Rationale:
    Categories enable type-safe filtering and composition. Each category represents
    a different aspect of agent capability (patterns, tools, practices, metrics).
    """
    assert category.value == expected_type
    assert isinstance(category, AtomCategory)
