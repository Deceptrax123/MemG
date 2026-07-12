from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field
from typing import List


class AtomicFact(BaseModel):
    fact: str = Field(
        description="A single, unambiguous factual statement. Resolve all pronouns.")
    topic: str = Field(
        description="The overarching theme (e.g., 'Server Outage', 'Weekend Plans').")
    entities: List[str] = Field(
        description="Named entities or specific objects mentioned.")
    timestamp: Union[str, float] = Field(
        description="The absolute timestamp of this specific fact."
    )


class WindowExtraction(BaseModel):
    facts: List[AtomicFact] = Field(
        default_factory=list,
        description="List of all independent facts discussed in the conversation window."
    )


class MCQResponse(BaseModel):
    answer_choice: str = Field(
        ..., description="The exact letter of the correct option(e.g.,'A','B','C','D').")
    rationale: str = Field(...,
                           description='A one-sentence explanation based on the context.')


class FreeTextResponse(BaseModel):
    rationale: str = Field(
        description="A step-by-step logical deduction using ONLY the provided verified facts. If dates/times conflict, explain which one you trusted based on the context."
    )
    answer: str = Field(
        description="The final, concise answer to the user's question (1-5 words if possible, or a short sentence)."
    )


class TemporalConstraint(BaseModel):
    is_relative: bool = Field(
        description="True if the question asks for a sequence like 'before', 'after', or 'currently/latest'.")
    anchor_event: Optional[str] = Field(
        description="The specific event or condition acting as the temporal anchor, if any.")
    sequence_direction: Optional[str] = Field(
        description="One of: 'latest', 'oldest', 'before_anchor', 'after_anchor', or 'all'.")


class RetrievalPlan(BaseModel):
    dense_semantic_query: str = Field(
        description="The primary declarative search string.")
    decomposed_queries: List[str] = Field(
        description="For multi-hop queries, break the question down into 2-3 independent, single-hop sub-queries. Leave EMPTY for single-hop.",
        default_factory=list
    )
    difficulty_level: Literal["single_hop",
                              "multi_hop", "open_domain"] = Field(...)
    target_entities: List[str] = Field(
        description="Exact proper nouns (e.g., 'Calvin', 'Apple').")
    search_keywords: List[str] = Field(
        description="Crucial nouns, objects, or action words from the query (e.g., 'blog', 'car', 'camping'). Do not include generic words."
    )
    temporal_intent: Literal["current", "historical",
                             "trajectory", "all"] = Field(...)


class MemoryEntry(BaseModel):
    lossless_restatement: str = Field(
        description="Complete unambiguous restatement (must include all subjects, objects, time, location, etc.). No pronouns."
    )
    keywords: List[str] = Field(
        description="Core keywords (names, places, entities, topic words)."
    )
    timestamp: Optional[str] = Field(
        description="Absolute time in ISO 8601 format if explicit time mentioned, else null."
    )
    location: Optional[str] = Field(
        description="Specific location name if mentioned, else null."
    )
    persons: List[str] = Field(
        description="All person names mentioned."
    )
    entities: List[str] = Field(
        description="Companies, products, organizations, etc."
    )
    topic: str = Field(
        description="The overarching topic phrase."
    )


class MemoryExtractionList(BaseModel):
    memory_entries: List[MemoryEntry]
