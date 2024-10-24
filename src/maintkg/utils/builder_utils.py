"""Builder utilities."""

from datetime import date, datetime
from typing import Any, Optional, Set

from loguru import logger
from neo4j import Transaction


def clear_database(tx: Transaction) -> None:
    """Clear the database.

    WARNING: This will delete all data in the database!
    """
    tx.run("MATCH (n) DETACH DELETE n")


def string_to_date(date_string: str) -> date:
    """Convert a string to a date."""
    return datetime.strptime(str(date_string), "%Y-%m-%d").date()


def create_or_get_node(
    tx: Transaction,
    node_name: str,
    node_type: str,
    node_properties: Optional[dict] = None,
    date_keys: Optional[Set[str]] = None,
) -> Any:
    """Create or get a node in the database."""
    try:
        if node_properties is None:
            node_properties = {}
        if date_keys is None:
            date_keys = set()

        # Convert properties that are dates
        for key in node_properties.keys():
            if key in date_keys:
                node_properties[key] = string_to_date(node_properties[key])

        # Dynamically build the properties part of the Cypher query
        properties_cypher = ", ".join([f"{k}: ${k}" for k in node_properties.keys()])
        if properties_cypher:
            properties_cypher = (
                ", " + properties_cypher
            )  # Prepend comma if there are properties

        cypher_query = (
            f"MERGE (n:`{node_type}` {{name: $name{properties_cypher}}}) " "RETURN n"
        )

        parameters = {"name": node_name, **node_properties}
        result = tx.run(cypher_query, parameters)
        return result.single()[0]
    except Exception as e:
        logger.error(f"Error with Neo4J node result ({node_name}, {node_type}) - {e}")


def create_relationship(
    tx: Transaction,
    head_node: dict,
    relation_name: str,
    tail_node: dict,
    relation_properties: Optional[dict] = None,
) -> Any:
    """Create a relationship between two nodes."""
    try:
        if relation_properties is None:
            relation_properties = {}
        cypher_query = (
            "MATCH (a), (b) "
            "WHERE a.name = $head_name AND b.name = $tail_name "
            f"MERGE (a)-[r:{relation_name} {{ {', '.join([f'{k}: ${k}' for k in relation_properties.keys()])} }}]->(b) "
            "RETURN r"
        )
        parameters = {
            "head_name": head_node["name"],
            "tail_name": tail_node["name"],
            **relation_properties,
        }
        result = tx.run(cypher_query, parameters)

        # Return relation element_id
        return result.single()["r"].id
    except Exception as e:
        logger.error(
            f"Error with Neo4J relationship result ({head_node}, {relation_name}, {tail_node}) - {e}"
        )
