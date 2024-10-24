"""Refinement utilities.

These utilities are for refining automatically extracted semantic triples
from maintenance short texts.
"""

import json
from pathlib import Path
from typing import List, Tuple

from loguru import logger

from maintkg.models import Node, Relation, Triple
from maintkg.utils.node_normalisation import normalise
from maintkg.utils.resources import ENTITY_TYPES, RELATION_CONSTRAINTS, RELATION_TYPES

# Settings
ACTIVITIES_ARE_N_ARY = True  # If True, this will create binary assertions with Unknown head/tails if information is not present.
REMOVE_SUBSUMPTION = (
    True  # If True, triples with the "isA" relation will not be used in the graph
)

GAZETTEER_FILEPATH = Path(__file__).resolve().parent / "gazetteer.json"

with open(GAZETTEER_FILEPATH, "r") as f:
    gazetteer = json.load(f)
    # Transform gazetteer into a dictionary with inner-dictionary structure
    gazetteer = {
        outer_key: {pair[0]: pair[1] for pair in outer_value}
        for outer_key, outer_value in gazetteer.items()
    }
    logger.info(f"Gazetteer loaded: {len(gazetteer.keys())} keys")


def format_entity_type(type_str: str, capitalize: bool = True) -> str:
    """Refines the semantic typing of the entity into a standardised format.

    This removes the "<>" brackets used in NoisIE and capitalizes the first
    letter of the entity type to match the conventions used in Neo4J.
    """
    formatted_type_str = type_str.strip("<>")
    return formatted_type_str.capitalize() if capitalize else formatted_type_str


def format_relation_type(type_str: str, uppercase: bool = True) -> str:
    """Refines the semantic typing of the relation into a standardised format.

    This removes the "_" between relation phrases and uppercases the entire
    relation phrase to match the conventions used in Neo4J.
    """
    formatted_type_str = type_str.replace(" ", "_")
    return formatted_type_str.upper() if uppercase else formatted_type_str


def format_triple(
    triple: Tuple[str, str, str, str, str], capitalize_and_uppercase: bool = True
) -> Tuple[str, str, str, str, str]:
    """Refines the semantic typing of the triple into a standardised format.

    Capitalizes entities and uppercases relations for Neo4J best-practice.
    """
    head, head_type, tail, tail_type, relation = triple
    return (
        head,
        format_entity_type(head_type, capitalize_and_uppercase),
        tail,
        format_entity_type(tail_type, capitalize_and_uppercase),
        format_relation_type(relation, capitalize_and_uppercase),
    )


def validate_triple(triple: Tuple[str, str, str, str, str]) -> Tuple[bool, str]:
    """Validate a triple to ensure it meets the constraints of MaintKG."""
    head, head_type, tail, tail_type, relation = triple
    head = head.strip()
    tail = tail.strip()
    head_type = head_type.lower()
    tail_type = tail_type.lower()
    relation = relation.lower()

    if head == tail and head_type == tail_type:
        return False, "self-reference"

    if head == "" or tail == "":
        return False, "empty entity surface form"

    if (
        head_type.split("/")[0] not in ENTITY_TYPES
        or tail_type.split("/")[0] not in ENTITY_TYPES
    ):
        return False, "invalid entity type(s)"

    if relation not in RELATION_TYPES:
        return False, "invalid relation type"

    if (
        head_type.split("/")[0],
        tail_type.split("/")[0],
    ) not in RELATION_CONSTRAINTS.get(
        relation
    ):  # type: ignore
        logger.error(f"head_type: {head_type}, tail_type: {tail_type}")
        return False, "invalid triple pattern"

    return True, "valid"


def process_node(
    node: Node, use_gazetteer: bool = False
) -> Tuple[Node, List[str], bool]:
    """Process a node to refine its semantic typing and surface form.

    Node processing includes phrase normalisation, noun normalisation, verb
    normalisation, and humanizing node types.

    Parameters
    ----------
    node : Node
        The node to be processed.
    use_gazetteer : bool
        Whether to use the entity gazetteer to add fine-grained information to the node.

    Returns
    -------
    Node
        The processed node.
    """
    node.type = node.type.strip("<>").capitalize()

    # Normalise node
    new_name, norm_ops, name_changed = normalise(s=node.name, tag=node.type)
    if name_changed:
        node.name = new_name

    if use_gazetteer:
        gazetteer_mapping = gazetteer.get(node.type)
        if gazetteer_mapping is not None:
            hierarchy = gazetteer_mapping.get(node.name, node.type)
            node.type = hierarchy
            # node.properties = {
            #     **(node.properties if node.properties else {}),
            #     "hierarchy": hierarchy,
            # }

    return node, norm_ops, name_changed


def process_relation(relation: Relation) -> Relation:
    """Process a relation.

    - Verbal relation phrase normalisation.
    - humanize relations
    """
    relation.name = relation.name.replace(" ", "_").upper()
    return relation
