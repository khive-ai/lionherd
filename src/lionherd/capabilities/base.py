"""
Atomic Knowledge Unit System for Agent Capability Specialization.

This module defines the foundational schema for atomic knowledge units aligned with
Lion ecosystem capability-based security principles. Atoms represent the smallest
complete actionable affordances for AI agent specialization.

Design Philosophy:
------------------
Atoms as capability grants (conceptual alignment with Lion formal verification):
- Each atom grants knowledge + permissions to agent
- Compositional via category theory (symmetric monoidal category)
- Partial order via relationships (enables/requires graph)
- Future: HMAC sealing for unforgeable capability tokens

Atomicity Definition:
---------------------
An atom is atomic when it provides:
1. Situation - WHEN this knowledge applies (preconditions)
2. Action - HOW to apply it (procedure/pattern)
3. Outcome - WHAT it achieves (postconditions/effects)
4. Relationships - HOW it connects (graph edges)
5. Semantic position - WHERE in knowledge space (embedding)

Dual-Space Design:
------------------
Atoms exist in two computational spaces:
- Embedding space (continuous): Vector similarity for semantic search
- Graph space (discrete): Typed relationships for compositional reasoning

These must be consistent: graph edges correlate with embedding proximity.

References:
-----------
- Lion capability formal verification: /lionrust/docs/proofs/v2/layer0/L0-CAP-1/
- Category theory foundations: /lionrust/docs/proofs/v1/latex/ch1_content.tex
- Affordance theory: Gibson's ecological psychology
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class AtomCategory(str, Enum):
    """
    Atom categories define the type of knowledge represented.

    Categories align with different aspects of agent capability:
    - PATTERN: Architectural/design patterns (how to structure)
    - TOOL: Concrete tools and libraries (what to use)
    - BEST_PRACTICE: Proven practices and principles (what works)
    - METRIC: Measurement and evaluation criteria (how to measure)
    """

    PATTERN = "pattern"
    TOOL = "tool"
    BEST_PRACTICE = "best_practice"
    METRIC = "metric"


class AtomRights(BaseModel):
    """
    Rights define what an agent can DO with an atom (future: capability semantics).

    Aligned with Lion capability rights (read/write/execute/delegate):
    - read: Can retrieve atom knowledge from storage
    - apply: Can use atom in generation/reasoning (analogous to execute)
    - compose: Can combine atom with others (composition permission)
    - delegate: Can grant atom to other agents (delegation chain)

    Future: Enforce via capability manager with HMAC sealing.
    """

    read: bool = True
    apply: bool = True
    compose: bool = True
    delegate: bool = False  # Most restricted by default


class AtomConstraints(BaseModel):
    """
    Constraints define WHEN and HOW an atom can be used (future: capability constraints).

    Aligned with Lion capability constraints (valid_until, max_uses, nonce):
    - valid_until: Time-limited knowledge (expires after date)
    - max_uses: Usage counter (prevents over-reliance)
    - context_required: Prerequisites for applicability
    - conflicts: Mutually exclusive atoms (prevent contradiction)

    Future: Runtime enforcement via constraint checker.
    """

    valid_until: datetime | None = None
    max_uses: int | None = None
    context_required: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)


class AtomApplicability(BaseModel):
    """
    Applicability defines WHEN an atom applies (preconditions/situation).

    This is the "situation" component of the affordance structure:
    - situation: Natural language description of when to use
    - context: Required contextual conditions (e.g., "API", "multi-server")
    - anti_patterns: When NOT to use (negative conditions)

    Example:
        situation: "Need stateless authentication with horizontal scaling"
        context: ["API", "multi-server", "microservices"]
        anti_patterns: ["Single server", "Stateful session required"]
    """

    situation: str = Field(..., description="When this atom applies (preconditions)")
    context: list[str] = Field(
        default_factory=list, description="Required contextual conditions"
    )
    anti_patterns: list[str] = Field(
        default_factory=list, description="When NOT to use this atom"
    )


class AtomKnowledge(BaseModel):
    """
    Knowledge defines the core content of an atom (what/why/how/outcomes).

    This is the "action + outcome" component of the affordance structure:
    - what: Core concept definition (ontology)
    - why: Rationale/purpose (motivation)
    - how: Procedure/implementation (action)
    - outcomes: Effects/postconditions (what it achieves)
    - trade_offs: Benefits vs costs (decision support)

    Structure designed for agent actionability: every atom tells agent
    both WHAT to do and WHY it works.
    """

    what: str = Field(..., description="Core concept definition")
    why: str = Field(..., description="Rationale and purpose")
    how: dict[str, list[str]] = Field(
        ...,
        description="Implementation details: steps, patterns, tools",
    )
    outcomes: dict[str, any] = Field(
        ...,
        description="Effects: achieves, enables, metrics",
    )
    trade_offs: dict[str, list[str]] = Field(
        ...,
        description="Benefits, costs, when_to_skip",
    )


class AtomRelationships(BaseModel):
    """
    Relationships define HOW atoms connect in the knowledge graph.

    Aligned with Lion capability partial order (C₁ ≤ C₂ if R₁ ⊆ R₂):
    - enables: Forward capability chain (this atom enables those)
    - requires: Backward dependency (this needs those first)
    - conflicts: Mutually exclusive (cannot coexist)
    - complements: Synergistic (work well together)
    - specializes: More specific version of (inheritance)
    - generalizes: Abstraction of (inverse of specializes)

    Partial order semantics:
    - If A enables B, then A ≤ B in capability order
    - If A requires B, then B ≤ A in capability order
    - Conflicts break the order (no relationship)

    Graph queries use these for compositional reasoning and capability chains.
    """

    enables: list[str] = Field(
        default_factory=list,
        description="Atoms that this enables (forward chain)",
    )
    requires: list[str] = Field(
        default_factory=list,
        description="Atoms required before this (dependencies)",
    )
    conflicts: list[str] = Field(
        default_factory=list,
        description="Mutually exclusive atoms",
    )
    complements: list[str] = Field(
        default_factory=list,
        description="Atoms that work well with this",
    )
    specializes: list[str] = Field(
        default_factory=list,
        description="More general atoms this specializes",
    )
    generalizes: list[str] = Field(
        default_factory=list,
        description="More specific atoms this generalizes",
    )


class AtomNode(BaseModel):
    """
    AtomNode represents a complete atomic knowledge unit for agent specialization.

    Conceptual Alignment with Lion Capabilities:
    ---------------------------------------------
    Each atom is analogous to a Lion capability grant:
    - ln_id: Unforgeable identifier (like ObjectId in Lion)
    - atom_ref: Human-readable reference (URI-like)
    - rights: What agent can do with this atom (future: enforced)
    - constraints: Usage limits and preconditions (future: enforced)
    - content: Knowledge payload (what agent learns)
    - relationships: Graph edges (capability partial order)
    - embedding: Semantic position (continuous space)

    Future: HMAC sealing for unforgeable capability tokens (deferred until needed).

    Validation Heuristics (atom correctly sized):
    ---------------------------------------------
    1. Self-contained: Understandable without loading other atoms
    2. Actionable: Agent can apply to solve problems
    3. Graph connected: Has 2-6 meaningful relationships
    4. Embedding distinct: Distance 0.3-0.7 from nearest neighbors
    5. Prompt sized: Generates 100-500 tokens when rendered

    Dual-Space Retrieval:
    ---------------------
    - Semantic search: ORDER BY embedding <=> query_embedding
    - Graph traversal: Recursive CTE on relationships
    - Hybrid: Semantic search within graph neighborhood

    Example:
        atom = AtomNode(
            ln_id="550e8400-e29b-41d4-a716-446655440000",
            atom_ref="patterns/jwt_auth",
            category=AtomCategory.PATTERN,
            applicability=AtomApplicability(
                situation="Need stateless auth with horizontal scaling",
                context=["API", "multi-server"],
                anti_patterns=["Single server app"]
            ),
            knowledge=AtomKnowledge(
                what="JWT-based stateless authentication pattern",
                why="Enables horizontal scaling without session state",
                how={
                    "steps": ["Generate RS256 keypair", "Issue JWT on login", ...],
                    "tools": ["jsonwebtoken", "passport-jwt"],
                    "patterns": ["middleware pattern"]
                },
                outcomes={
                    "achieves": ["stateless auth", "horizontal scalability"],
                    "enables": ["load balancing", "zero-downtime deployment"],
                    "metrics": {"latency_overhead": "~5ms", "token_size": "~200 bytes"}
                },
                trade_offs={
                    "benefits": ["Scalable", "No session storage"],
                    "costs": ["Token size", "Key management complexity"],
                    "when_to_skip": ["Single server", "Need instant revocation"]
                }
            ),
            relationships=AtomRelationships(
                enables=["capabilities/can_scale_horizontally"],
                requires=["best_practices/key_management"],
                conflicts=["patterns/server_side_sessions"],
                complements=["patterns/refresh_tokens"]
            ),
            embedding=[0.1, 0.2, ...],  # 768-dim vector
        )
    """

    # Identity (unforgeable in Lion model, conceptual for now)
    ln_id: str = Field(..., description="LionAGI unique identifier (UUID)")
    atom_ref: str = Field(
        ...,
        description="Human-readable reference (e.g., 'patterns/jwt_auth')",
    )
    category: AtomCategory = Field(..., description="Atom category")
    version: str = Field(default="1.0.0", description="Semantic version")

    # Dual-space positioning
    embedding: list[float] = Field(
        default_factory=list,
        description="768-dim embedding vector for semantic search",
    )

    # Capability semantics (future: enforced)
    rights: AtomRights = Field(
        default_factory=AtomRights,
        description="What agent can do with this atom",
    )
    constraints: AtomConstraints = Field(
        default_factory=AtomConstraints,
        description="Usage limits and preconditions",
    )

    # Actionable content (affordance structure)
    applicability: AtomApplicability = Field(
        ..., description="When this atom applies (situation/context)"
    )
    knowledge: AtomKnowledge = Field(
        ..., description="Core knowledge content (what/why/how/outcomes)"
    )

    # Graph relationships (partial order)
    relationships: AtomRelationships = Field(
        default_factory=AtomRelationships,
        description="How this atom connects to others",
    )

    # Metadata
    applies_to: list[str] = Field(
        default_factory=list,
        description="Languages/domains this applies to",
    )
    tags: list[str] = Field(
        default_factory=list, description="Discovery and categorization tags"
    )
    created_at: datetime | None = None
    metadata: dict[str, any] = Field(default_factory=dict)

    def to_prompt(self, context: dict[str, any] | None = None) -> str:
        """
        Generate a contextualized prompt from this atom for agent consumption.

        Args:
            context: Optional context dict for template rendering

        Returns:
            Formatted prompt string (100-500 tokens typically)
        """
        return f"""## {self.atom_ref}

**When to use**: {self.applicability.situation}
**Purpose**: {self.knowledge.why}

**What it is**: {self.knowledge.what}

**Implementation**:
{chr(10).join(f"- {step}" for step in self.knowledge.how.get('steps', []))}

**Tools**: {', '.join(self.knowledge.how.get('tools', []))}
**Achieves**: {', '.join(self.knowledge.outcomes.get('achieves', []))}

**Trade-offs**:
- Benefits: {', '.join(self.knowledge.trade_offs.get('benefits', []))}
- Costs: {', '.join(self.knowledge.trade_offs.get('costs', []))}
- Skip when: {', '.join(self.knowledge.trade_offs.get('when_to_skip', []))}

**Related atoms**:
- Enables: {', '.join(self.relationships.enables)}
- Requires: {', '.join(self.relationships.requires)}
"""

    def validate_atomicity(self) -> dict[str, bool]:
        """
        Validate that this atom is correctly sized (atomic).

        Returns:
            Dict of validation results for atomicity heuristics
        """
        prompt = self.to_prompt()
        token_count = len(prompt.split())  # Rough approximation

        return {
            "self_contained": bool(
                self.knowledge.what and self.knowledge.why
            ),  # Has core content
            "actionable": bool(self.knowledge.how.get("steps")),  # Has procedure
            "graph_connected": 2
            <= len(
                self.relationships.enables
                + self.relationships.requires
                + self.relationships.complements
            )
            <= 6,
            "embedding_exists": len(self.embedding) == 768,  # Has semantic position
            "prompt_sized": 100
            <= token_count
            <= 500,  # Reasonable prompt size
        }


# Type aliases for clarity
AtomRef = str  # e.g., "patterns/jwt_auth"
AtomID = str  # e.g., UUID
