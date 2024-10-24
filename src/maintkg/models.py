"""Models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BaseRecord(BaseModel):
    """Single base maintenance work order record model."""

    ids: List[int] = Field(..., description="The list of unique identifiers.")
    input: str = Field(..., description="The input text that is fed into the model.")
    original: str = Field(..., description="The original text.")


class Records(BaseModel):
    """Maintenance work order records model."""

    items: Dict[str, List[BaseRecord]] = Field(
        ..., description="The maintenance work order records."
    )


class Relation(BaseModel):
    """Relation model."""

    name: str
    type: Optional[str] = None
    properties: Optional[Dict[str, Any]] = Field(
        None, description="The key-value pair properties of the relation."
    )


class Node(BaseModel):
    """Node model."""

    name: str = Field(..., description="The surface form of the node.")
    type: str = Field(..., description="The type of the node.")
    properties: Optional[Dict[str, Any]] = Field(
        None, description="The key-value pair properties of the node."
    )


class Triple(BaseModel):
    """Triple model."""

    head: Node
    relation: Relation
    tail: Node


class RecordTypeSummary(BaseModel):
    """Summary of record types for a FLOC."""

    planned: str
    unplanned: str
    other: str
    total: int


class DateDurationSummary(BaseModel):
    """Summary of date durations for a FLOC."""

    min: str
    max: str


class ProcessingSummary(BaseModel):
    """Complete summary of data processing."""

    total_records_all: int
    unique_flocs_all: int
    unplanned_records_all: int
    focus_systems: Optional[List[str]]
    focus_system_count: Optional[int]
    record_types_by_floc: Dict[str, RecordTypeSummary]
    record_date_durations_by_floc: Dict[str, DateDurationSummary]
    total_records: int
    unplanned_records: int
    planned_records: int
    other_records: int
    unique_flocs_unplanned: int
    unque_floc_record_counts: Dict[str, int]
    unique_texts_not_processed: int
    unique_texts_not_processed_ave_len: float
    unique_texts: int
    unique_texts_ave_len: float
