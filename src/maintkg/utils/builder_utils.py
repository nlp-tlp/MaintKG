"""Builder utilities."""

import typing
from collections import Counter
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import pandas as pd
from loguru import logger
from neo4j import Transaction

from maintkg.models import (
    CounterItem,
    DateDurationSummary,
    EntityMention,
    EntityType,
    FrequencyItem,
    GenericAnalysis,
    ProcessingSummary,
    RecordTypeSummary,
    RelationType,
    Triple,
    TripleAnalysis,
    TriplePattern,
    TripleTuple,
)
from maintkg.settings import Settings
from maintkg.utils.general_utils import calculate_pmi


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
    head_node: Dict[str, str],
    relation_name: str,
    tail_node: Dict[str, str],
    relation_properties: Optional[Dict[str, Union[str, int, float, bool]]] = None,
) -> int:
    """
    Create a relationship between two nodes in Neo4j.

    Parameters
    ----------
    tx :
        Neo4j transaction object
    head_node :
        Dictionary containing source node properties (must include 'name')
    relation_name :
        Name of the relationship to create
    tail_node :
        Dictionary containing target node properties (must include 'name')
    relation_properties :
        Optional dictionary of relationship properties

    Returns
    -------
    int :
        The internal Neo4j ID of the created relationship

    Raises
    ------
    ValueError :
        If required node properties are missing
    RuntimeError :
        If relationship creation fails
    """
    if not head_node.get("name") or not tail_node.get("name"):
        raise ValueError("Both head_node and tail_node must contain 'name' property")

    # Initialise empty properties if None
    relation_properties = relation_properties or {}

    try:
        # Build property string for Cypher query
        property_string = (
            ", ".join(f"{k}: ${k}" for k in relation_properties.keys())
            if relation_properties
            else ""
        )

        # Construct base query
        cypher_query = (
            "MATCH (a), (b) " "WHERE a.name = $head_name AND b.name = $tail_name "
        )

        # Add MERGE clause with or without properties
        if property_string:
            cypher_query += f"MERGE (a)-[r:{relation_name} {{{property_string}}}]->(b) "
        else:
            cypher_query += f"MERGE (a)-[r:{relation_name}]->(b) "

        cypher_query += "RETURN r"

        # Combine parameters
        parameters = {
            "head_name": head_node["name"],
            "tail_name": tail_node["name"],
            **relation_properties,  # type: ignore
        }

        # Execute query
        result = tx.run(cypher_query, parameters)
        record = result.single()

        if record is None:
            raise RuntimeError("No relationship was created")

        relationship_id = int(record["r"].id)
        if relationship_id is None:
            raise RuntimeError("Created relationship has no ID")

        return relationship_id

    except Exception as e:
        error_msg = (
            f"Failed to create Neo4j relationship: "
            f"({head_node['name']}) -[{relation_name}]-> ({tail_node['name']})"
        )
        logger.error(f"{error_msg} - {str(e)}")
        raise RuntimeError(error_msg) from e


def analyse_category(
    category: Union[Literal["triples"], str],
    data: List[List[Union[TripleTuple, Any]]],
    top_n: int = 5,
) -> Union[TripleAnalysis, GenericAnalysis]:
    """
    Analyze data based on its category, with special handling for triple data.

    Parameters
    ----------
    category :
        Category of the data. Special handling for 'triples',
        otherwise treated as generic data.
    data :
        List of documents containing the data to analyze.
        For triples: List[List[Tuple[str, str, str, str, str]]] where each tuple is
            (head, head_type, tail, tail_type, relation)
        For others: List[List[Any]] where items will be converted to tuples for counting
    top_n :
        Number of top frequent items to return. Defaults to 5.

    Returns
    -------
        Dict[str, Any]: Analysis results containing statistics and frequency data.
            For triples category:
            - Total: Total number of triples
            - Total Unique: Number of unique triples
            - Average per Document: Average triples per document
            - Min/Max per Document: Min/max triples in any document
            - Entities: Counter of entity types
            - Total Unique Entity Mentions: Number of unique entity mentions
            - Relations: Counter of relations
            - Triple Patterns: Counter of (head_type, relation, tail_type) patterns
            - Top-n statistics for entities, relations, and patterns

            For other categories:
            - Total: Total number of items
            - Total Unique: Number of unique items
            - Average per Document: Average items per document
            - Min/Max per Document: Min/max items in any document
            - Top-n Most Frequent: Most frequent items

    Raises
    ------
        ValueError: If data is empty or malformed.
    """
    if not data:
        raise ValueError("Empty data provided for analysis")

    def counter_to_dict(counter: typing.Counter[CounterItem]) -> Dict[str, int]:
        """Convert Counter object to JSON-serializable dictionary."""
        return {str(key): value for key, value in counter.items()}

    def get_top_n(counter: typing.Counter[CounterItem]) -> List[FrequencyItem]:
        """Get top N items from a Counter in a structured format."""
        return [
            FrequencyItem(item=str(key), count=value)
            for key, value in counter.most_common(top_n)
        ]

    if category == "triples":
        # Initialize counters
        entity_type_counter: typing.Counter[EntityType] = Counter()
        relation_counter: typing.Counter[RelationType] = Counter()
        triple_pattern_counter: typing.Counter[TriplePattern] = Counter()
        entity_mentions: Set[EntityMention] = set()

        # Process each triple
        for document in data:
            for triple in document:
                head, head_type, tail, tail_type, relation = triple

                # Track unique entity mentions
                entity_mentions.add((head, head_type))
                entity_mentions.add((tail, tail_type))

                # Update counters
                entity_type_counter.update([head_type, tail_type])
                relation_counter[relation] += 1
                triple_pattern_counter[(head_type, relation, tail_type)] += 1

        # Calculate statistics
        doc_lengths = [len(doc) for doc in data]
        total_triples = sum(doc_lengths)
        unique_triples = len({tuple(triple) for doc in data for triple in doc})

        return TripleAnalysis(
            Total=total_triples,
            Total_Unique=unique_triples,
            Average_per_Document=total_triples / len(data),
            Min_per_Document=min(doc_lengths, default=0),
            Max_per_Document=max(doc_lengths, default=0),
            Entities=counter_to_dict(entity_type_counter),
            Total_Unique_Entity_Mentions=len(entity_mentions),
            Relations=counter_to_dict(relation_counter),
            Triple_Patterns=counter_to_dict(triple_pattern_counter),
            Top_n_Most_Frequent_Entities=get_top_n(entity_type_counter),
            Top_n_Most_Frequent_Relations=get_top_n(relation_counter),
            Top_n_Most_Frequent_Triple_Patterns=get_top_n(triple_pattern_counter),
        )

    # Process generic data
    all_items = [tuple(item) for doc in data for item in doc]
    doc_lengths = [len(doc) for doc in data]
    item_frequency: typing.Counter[Tuple[Any, ...]] = Counter(all_items)

    return GenericAnalysis(
        Total=len(all_items),
        Total_Unique=len(set(all_items)),
        Average_per_Document=len(all_items) / len(data),
        Min_per_Document=min(doc_lengths, default=0),
        Max_per_Document=max(doc_lengths, default=0),
        Top_n_Most_Frequent=item_frequency.most_common(top_n),
    )


def create_enriched_semantic_triples(
    triples: List[Triple], pmi_stats: dict
) -> List[Triple]:
    """Enrich semantic triples with PMI statistics."""
    enriched_triples = []
    for t in triples:
        t.head.properties = {
            "frequency": pmi_stats["frequency"]["node"][t.head.name],
            "probability": pmi_stats["probability"]["node"][t.head.name],
        }
        t.tail.properties = {
            "frequency": pmi_stats["frequency"]["node"][t.tail.name],
            "probability": pmi_stats["probability"]["node"][t.tail.name],
        }
        _pmi = calculate_pmi(
            triple=t,
            triple_prob=pmi_stats["probability"]["triple"],
            head_prob=pmi_stats["probability"]["head"],
            relation_prob=pmi_stats["probability"]["relation"],
            tail_prob=pmi_stats["probability"]["tail"],
        )
        t.relation.properties = {
            "frequency": pmi_stats["frequency"]["triple"][
                (t.head.name, t.relation.name, t.tail.name)
            ],
            "probability": pmi_stats["probability"]["triple"][
                (t.head.name, t.relation.name, t.tail.name)
            ],
            "pmi": _pmi,
        }

        enriched_triples.append(t)
    return enriched_triples


def load_and_clean_data(settings: Settings) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Load initial data and perform basic cleaning."""
    df = pd.read_csv(settings.input_dir / settings.input.csv_filename)
    init_df_size = len(df)
    logger.info(f"Loaded {init_df_size} records")

    df.rename(columns=settings.col_mapping, inplace=True)
    df = df.dropna(subset=[settings.processing.floc_col])
    df = df[df[settings.processing.text_col].apply(lambda x: isinstance(x, str))]

    logger.info(
        f"Dropped {init_df_size - len(df)} invalid rows (non-string texts and/or missing flocs)"
    )

    df = df[settings.cols_to_keep]
    return df, {
        "total_records_all": len(df),
        "unique_flocs_all": len(df[settings.processing.floc_col].unique()),
    }


def add_dummy_columns(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Add dummy columns if specified in settings."""
    if settings.processing.add_dummy_cols:
        logger.debug(settings.dummy_cols)
        for col_name, func in settings.dummy_cols.items():
            if col_name not in df.columns:
                df[col_name] = [func() for _ in range(len(df))]
    return df


def format_numeric_columns(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Format cost and time columns to numeric values."""
    for col in [settings.processing.cost_col, settings.processing.time_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].replace("[^0-9.]", "", regex=True).replace("", 0)
            )
    return df


def format_date_column(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Format date column according to specified format."""
    if settings.processing.start_date_col in df.columns:
        df[settings.processing.start_date_col] = pd.to_datetime(
            df[settings.processing.start_date_col], format=settings.input.date_format
        ).dt.strftime(settings.processing.date_format)
    return df


def calculate_record_summaries(
    df: pd.DataFrame, settings: Settings
) -> Tuple[Dict[str, RecordTypeSummary], Dict[str, DateDurationSummary]]:
    """Calculate record type and date duration summaries by FLOC."""
    record_types = {}
    date_durations = {}

    for floc, records in df.groupby(settings.processing.floc_col):
        planned_count = len(
            records[
                records[settings.processing.type_col].isin(
                    settings.input.planned_type_codes
                )
            ]
        )
        unplanned_count = len(
            records[
                records[settings.processing.type_col].isin(
                    settings.input.unplanned_type_codes
                )
            ]
        )
        other_count = len(records) - planned_count - unplanned_count
        total_count = planned_count + unplanned_count + other_count

        record_types[floc] = {
            "planned": f"{planned_count} ({planned_count/total_count*100:0.2f})",
            "unplanned": f"{unplanned_count} ({unplanned_count/total_count*100:0.2f})",
            "other": f"{other_count} ({other_count/total_count*100:0.2f})",
            "total": total_count,
        }

        date_durations[floc] = {
            "min": str(records[settings.processing.start_date_col].min()),
            "max": str(records[settings.processing.start_date_col].max()),
        }

    return record_types, date_durations  # type: ignore


def calculate_text_statistics(df: pd.DataFrame, settings: Settings) -> Dict[str, Any]:
    """Calculate statistics about text fields."""
    unique_texts_raw = set(df[settings.processing.text_col])
    unique_texts_processed = set(df["input"])

    return {
        "unique_texts_not_processed": len(unique_texts_raw),
        "unique_texts_not_processed_ave_len": sum(
            len(x.split()) for x in unique_texts_raw
        )
        / len(unique_texts_raw),
        "unique_texts": len(unique_texts_processed),
        "unique_texts_ave_len": sum(len(x.split()) for x in unique_texts_processed)
        / len(unique_texts_processed),
    }
