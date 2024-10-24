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
