"""MaintKG Builder."""

import json
import shelve
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
from loguru import logger
from neo4j import GraphDatabase, Session
from tqdm import tqdm
from typing_extensions import Self

from maintkg.models import Node, ProcessingSummary, Relation, Triple
from maintkg.settings import Settings
from maintkg.utils import (
    builder_utils,
    extraction_utils,
    general_utils,
    refinement_utils,
    resources,
)


class GraphBuilder:
    """MaintKG Graph Builder."""

    def __init__(
        self: "GraphBuilder", settings: Settings, cleanup: bool = False
    ) -> None:
        """Initialise the GraphBuilder."""
        super().__init__()
        self.settings = settings
        self.cleanup = cleanup

        # Format datetime in a Windows-friendly format (e.g., YYYY-MM-DD_HH-MM-SS)
        self.folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        logger.debug(f"input_dir: {self.settings.input_dir}")
        logger.debug(f"output_dir: {self.settings.output_dir}")
        logger.debug(f"cache_file: {self.settings.cache_file}")

    def save(self: Self, filename: str, extension: str, data: Any) -> None:
        """General method for saving files."""
        assert extension in ["csv", "json"], "Unsupported file extension"

        # Create the new directory
        path = self.settings.output_dir / self.folder_name
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f"{filename}.{extension}"

        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=0)
        else:
            with open(file_path, "w", encoding="utf-8") as file:
                if extension == "json":
                    try:
                        json.dump(data, file, indent=2, sort_keys=True)
                    except Exception as e:
                        logger.error(f"Error saving file: {e}")
        logger.debug(f"File saved: {file_path}")

    def load(self: Self, filename: str, extension: str) -> Union[pd.DataFrame, Dict]:
        """General method for loading CSV or JSON files."""
        assert extension in [
            "csv",
            "json",
        ], "Unsupported file extension. Expected csv or json."

        path = self.settings.output_dir / self.folder_name
        file_path = path / f"{filename}.{extension}"

        if extension == "csv":
            logger.debug(f"CSV file loaded: {file_path}")
            return pd.read_csv(file_path)
        with open(file_path, "r", encoding="utf-8") as file:
            logger.debug(f"JSON file loaded: {file_path}")
            return json.load(file)

    def _setup(self: Self) -> None:
        """Perform initial setup and processing of maintenance work order data."""
        summary: ProcessingSummary = {}
        # Load and clean initial data
        df, initial_stats = builder_utils.load_and_clean_data(self.settings)
        summary.update(initial_stats)
        # Add dummy columns if needed
        df = builder_utils.add_dummy_columns(df, self.settings)
        # Format numeric and date columns
        df = builder_utils.format_numeric_columns(df, self.settings)
        df = builder_utils.format_date_column(df, self.settings)
        # Calculate unplanned records
        summary["unplanned_records_all"] = len(
            df[df[self.settings.type_col].isin(self.settings.unplanned_type_codes)]
        )
        # Filter by systems if specified
        if self.settings.systems:
            logger.info(f"Filtering for systems: {self.settings.systems}")
            pattern = "|".join(self.settings.systems)
            df = df[
                df[self.settings.floc_col].str.contains(pattern, case=False, na=False)
            ]
            summary["focus_systems"] = df[self.settings.floc_col].unique().tolist()
            summary["focus_system_count"] = len(summary["focus_systems"])

        if self.settings.limit:
            logger.info(f"Limiting analysis to {self.settings.limit} records")
            df = df.sample(n=self.settings.limit)
        logger.info(f"Filtered to {len(df)} records")

        # Calculate record summaries
        record_types, date_durations = builder_utils.calculate_record_summaries(
            df, self.settings
        )
        summary["record_types_by_floc"] = record_types
        summary["record_date_durations_by_floc"] = date_durations

        # Filter for unplanned work only
        if self.settings.unplanned_only:
            df = df[df[self.settings.type_col].isin(self.settings.unplanned_type_codes)]

        if len(df) == 0:
            raise RuntimeError("DataFrame has no data after filtering.")

        # Calculate final record counts
        summary.update(
            {
                "total_records": len(df),
                "unplanned_records": len(
                    df[
                        df[self.settings.type_col].isin(
                            self.settings.unplanned_type_codes
                        )
                    ]
                ),
                "planned_records": len(
                    df[
                        df[self.settings.type_col].isin(
                            self.settings.planned_type_codes
                        )
                    ]
                ),
                "unique_flocs_unplanned": len(df[self.settings.floc_col].unique()),
                "unque_floc_record_counts": df[self.settings.floc_col]
                .value_counts()
                .to_dict(),
            }
        )

        summary["other_records"] = (
            summary["total_records"]
            - summary["planned_records"]
            - summary["unplanned_records"]
        )

        # Preprocess text fields
        df["input"] = df[self.settings.text_col].apply(
            lambda x: general_utils.simple_preprocessing(
                text=str(x), non_semantic_codes=self.settings.non_semantic_codes
            )
        )

        # Calculate text statistics
        summary.update(builder_utils.calculate_text_statistics(df, self.settings))

        # Save results
        self.save(filename="mwos", extension="csv", data=df)
        self.save("setup_summary", "json", summary)

    def _extraction(self: Self) -> None:
        """
        Perform information extraction from text data using NoisIE model.

        The method extracts information using a model available at http://localhost:8000.
        Results are cached in a shelve file for optimization of rerunning the process.

        Raises
        ------
        RuntimeError :
            If prediction service fails or returns misaligned results
        IOError :
            If cache file operations fail
        """
        try:
            # Load initial data
            data = self.load("mwos", "csv")
            data["input"] = data["input"].astype(str)
            inputs = list(data["input"].unique())
            logger.debug(f"Processing {len(inputs)} unique inputs")

            # Load cached predictions and identify new inputs
            all_preds, new_inputs = extraction_utils.load_cached_predictions(
                inputs, self.settings.cache_file
            )

            # Process new inputs if any exist
            if new_inputs:
                batches = [
                    new_inputs[i : i + self.settings.IE_MODEL_BATCH_SIZE]
                    for i in range(
                        0, len(new_inputs), self.settings.IE_MODEL_BATCH_SIZE
                    )
                ]
                logger.info(f"Created {len(batches)} batches for processing")

                new_preds, failed_preds = extraction_utils.process_new_predictions(
                    batches, self.settings.cache_file
                )
                all_preds.update(new_preds)

                if failed_preds:
                    logger.info(
                        f"Failed to get predictions for {len(failed_preds)} items"
                    )
                    self.save("failed_ie_batches", "json", failed_preds)

            # Create final output records
            output, summary, stats = extraction_utils.create_output_records(
                data, self.settings.cache_file
            )

            # Save results
            self.save(
                "mwos",
                "json",
                {
                    floc: [record.model_dump() for record in records]
                    for floc, records in output.items()
                },
            )
            summary["other"] = stats
            self.save("ie_summary", "json", summary)

            # Analyze and save extraction results
            results = {
                key: builder_utils.analyse_category(key, value).model_dump()
                for key, value in summary.items()
                if key != "other"
            }
            self.save("ie_analysis", "json", results)

        except Exception as e:
            logger.error(f"Error in information extraction process: {e}")
            raise

    def _refinement(self: Self) -> None:
        """
        Perform refinement of the extracted information.

        This process includes:
        - Triple validation and formatting
        - Entity normalization
        - PMI calculation
        - Entity type enrichment
        """
        data = self.load("mwos", "json")
        logger.info(f"Loaded: {len(data.keys())} systems/subsystems")

        semantic_triples: list = []
        summary = {
            "entity_normalisation": [],
            "enriched_entity_types": {},
        }
        enriched_triples = 0
        for floc, _data in data.items():
            for idx in range(len(_data)):
                _triples = []
                for t in _data[idx]["preds"]["triples"]:
                    t = refinement_utils.format_triple(triple=t)

                    is_valid, message = refinement_utils.validate_triple(t)

                    if message in summary.keys():
                        if message == "valid":
                            summary[message] += 1
                        else:
                            summary[message]["count"] += 1
                            summary[message]["instances"].append(t)
                    else:
                        if message == "valid":
                            summary[message] = 1
                        else:
                            summary[message] = {"count": 1, "instances": [t]}

                    if self.settings.REMOVE_SUBSUMPTION & (t[4].lower() == "is_a"):
                        logger.error(
                            f"Triple {t} is invalid (reason: subsumption) - skipping."
                        )
                        if "remove_subsumption" in summary.keys():
                            summary["remove subsumption"] += 1
                        else:
                            summary["remove subsumption"] = 1

                        continue

                    if not is_valid:
                        logger.error(
                            f"Triple {t} is invalid (reason: {message}) - skipping."
                        )
                    else:
                        (
                            _head,
                            _head_norm_ops,
                            _head_name_changed,
                        ) = refinement_utils.process_node(
                            Node(name=t[0], type=t[1]),
                            use_gazetteer=True,
                        )

                        if _head_name_changed:
                            summary["entity_normalisation"].extend(_head_norm_ops)

                        (
                            _tail,
                            _tail_norm_ops,
                            _tail_name_changed,
                        ) = refinement_utils.process_node(
                            Node(name=t[2], type=t[3]),
                            use_gazetteer=True,
                        )
                        if _tail_name_changed:
                            summary["entity_normalisation"].extend(_tail_norm_ops)

                        if (
                            _head.type.lower() not in resources.ENTITY_TYPES
                            or _tail.type.lower() not in resources.ENTITY_TYPES
                        ):
                            # New head/tail type has been used on the triple.
                            enriched_triples += 1

                        # Check new triple is valid:
                        is_valid, message = refinement_utils.validate_triple(
                            (_head.name, _head.type, _tail.name, _tail.type, t[4])
                        )

                        if not is_valid:
                            logger.error(
                                f"Triple {_head.name} -> {t[4]} -> {_tail.name} is invalid (reason: {message}) - skipping."
                            )
                            continue

                        _new_triple = Triple(
                            head=_head,
                            relation=refinement_utils.process_relation(
                                Relation(name=t[4])
                            ),
                            tail=_tail,
                        )

                        _triples.append(_new_triple)

                        # Count the types of entities used in triples post gazetteering
                        if (
                            _new_triple.head.type
                            in summary["enriched_entity_types"].keys()
                        ):
                            summary["enriched_entity_types"][_new_triple.head.type] += 1
                        else:
                            summary["enriched_entity_types"][_new_triple.head.type] = 1

                        if (
                            _new_triple.tail.type
                            in summary["enriched_entity_types"].keys()
                        ):
                            summary["enriched_entity_types"][_new_triple.tail.type] += 1
                        else:
                            summary["enriched_entity_types"][_new_triple.tail.type] = 1

                # Convert Pydantic models to dictionaries for serialization
                triples_dicts = [triple.model_dump() for triple in _triples]

                data[floc][idx]["triples"] = triples_dicts

                semantic_triples.extend(_triples)

        summary["entity_normalisation"] = dict(Counter(summary["entity_normalisation"]))
        summary["enriched_triples"] = enriched_triples

        summary["enriched_entity_types_reduction"] = {}
        for label in ["Object", "Activity", "Process", "State", "Property"]:
            current_parent_count = summary["enriched_entity_types"].get(label, 0)
            current_child_count = sum(
                [
                    v
                    for k, v in summary["enriched_entity_types"].items()
                    if k.startswith(label)
                ]
            )
            start_parent_count = current_parent_count + current_child_count
            summary["enriched_entity_types_reduction"][label] = {  # type: ignore
                "start": start_parent_count,
                "end": current_parent_count,
                "reduction": round(
                    (1 - (current_parent_count / start_parent_count)) * 100
                ),
            }

        self.save("refinement_summary", "json", summary)

        # if self.settings.dev and self.settings.limit < 100:
        #     logger.debug(json.dumps(data, indent=2))

        # Calculate Pointwise Mutual Information of Semantic Triples.
        pmi_stats = general_utils.calc_pmi_probabilities(triples=semantic_triples)

        # Sanitize and save PMI statistics
        self.save(
            "triple_pmi_stats",
            "json",
            {
                "frequency": {
                    "head": dict(pmi_stats["frequency"]["head"]),
                    "tail": dict(pmi_stats["frequency"]["tail"]),
                    "relation": dict(pmi_stats["frequency"]["relation"]),
                    "node": dict(pmi_stats["frequency"]["node"]),
                    "triple": {
                        ", ".join(k): v
                        for k, v in dict(pmi_stats["frequency"]["triple"]).items()
                    },
                },
                "probability": {
                    "head": dict(pmi_stats["probability"]["head"]),
                    "tail": dict(pmi_stats["probability"]["tail"]),
                    "relation": dict(pmi_stats["probability"]["relation"]),
                    "node": dict(pmi_stats["probability"]["node"]),
                    "triple": {
                        ", ".join(k): v
                        for k, v in dict(pmi_stats["probability"]["triple"]).items()
                    },
                },
            },
        )

        semantic_triple_pmis = builder_utils.create_enriched_semantic_triples(
            triples=semantic_triples, pmi_stats=pmi_stats
        )

        self.save(
            "mwo_triples_pmi", "json", [t.model_dump() for t in semantic_triple_pmis]
        )

        pmi_values = [
            t.relation.properties["pmi"]
            for t in semantic_triple_pmis
            if t.relation.properties is not None
        ]
        pmi_median, pmi_lower, pmi_upper = general_utils.calculate_pmi_score_thresholds(
            data=pmi_values
        )

        self.pmi_median = pmi_median
        self.pmi_lower = pmi_lower
        self.pmi_upper = pmi_upper
        logger.info(
            f"PMI Range: {self.pmi_lower} : {self.pmi_median} : {self.pmi_upper}"
        )

        self.save("mwo_triples", "json", data)

        # Save figure of PMI distribution for analysis
        path = self.settings.output_dir / self.folder_name
        general_utils.plot_pmi_scores(
            pmi_scores=pmi_values, file_path=path / "pmi_scores.png"
        )

    def init_graph(self: Self) -> None:
        """Initialise the Neo4J driver."""
        self.driver = GraphDatabase.driver(
            uri=self.settings.NEO4J_URI,
            auth=(self.settings.NEO4J_USERNAME, self.settings.NEO4J_PASSWORD),
        )

    def close(self: Self) -> None:
        """Close the Neo4J driver."""
        self.driver.close()

    def create_session(self: Self) -> Session:
        """Create a session for the Neo4J database."""
        return self.driver.session(database=self.settings.NEO4J_DATABASE)

    def _create_db_triple(self: Self, session: Session, triple: Triple) -> Any:
        """Create a triple in the Neo4J database."""
        head_node = session.execute_write(
            builder_utils.create_or_get_node,
            triple.head.name,
            triple.head.type,
            triple.head.properties,
            {self.settings.start_date_col},
        )
        tail_node = session.execute_write(
            builder_utils.create_or_get_node,
            triple.tail.name,
            triple.tail.type,
            triple.tail.properties,
            {self.settings.start_date_col},
        )
        return session.execute_write(
            builder_utils.create_relationship,
            head_node,
            triple.relation.name,
            tail_node,
            triple.relation.properties,
        )

    def _build(self: Self) -> None:
        """Build and populate graph database."""
        # Load data
        data = self.load("mwo_triples", "json")

        # Load PMI values
        semantic_triple_pmis = self.load("mwo_triples_pmi", "json")

        # Parse PMI triples
        semantic_triple_pmis = [Triple(**t) for t in semantic_triple_pmis]

        semantic_triple_stats = {
            (
                t.head.name,
                t.head.type,
                t.relation.name,
                t.relation.type,
                t.tail.name,
                t.tail.type,
            ): t
            for t in semantic_triple_pmis
        }

        # Initialize the Neo4j driver
        self.init_graph()
        session = self.create_session()

        removed_triples = []
        semantic_triple_count = 0
        record_triple_count = 0
        system_triple_count = 0

        with session:
            session.execute_write(builder_utils.clear_database)
            for _, records in tqdm(data.items(), desc="Processing Systems"):
                for record in records:
                    # Create and populate semantic triples
                    relation_ids = []
                    _semantic_triples = []
                    for _triple in record["triples"]:
                        semantic_triple_count += 1
                        head_node = Node(**_triple["head"])
                        tail_node = Node(**_triple["tail"])
                        relation = Relation(**_triple["relation"])

                        # Find matching information in semantic_triples_pmi
                        _stat_triple = semantic_triple_stats[
                            (
                                head_node.name,
                                head_node.type,
                                relation.name,
                                relation.type,
                                tail_node.name,
                                tail_node.type,
                            )
                        ]

                        remove_triple = (
                            _stat_triple.relation.properties["frequency"] == 1
                        ) and (
                            (_stat_triple.relation.properties["pmi"] < self.pmi_lower)
                            or (
                                self.pmi_upper < _stat_triple.relation.properties["pmi"]
                            )
                        )

                        if remove_triple:
                            # logger.info(
                            #     f"Removed triple with PMI score: {_stat_triple.relation.properties['pmi']} (reason: PMI score out of bounds)"
                            # )
                            removed_triples.append(_stat_triple.model_dump())
                            continue
                        head_node.properties = {
                            **(
                                head_node.properties
                                if head_node.properties is not None
                                else {}
                            ),
                            **_stat_triple.head.properties,
                        }
                        tail_node.properties = {
                            **(
                                tail_node.properties
                                if tail_node.properties is not None
                                else {}
                            ),
                            **_stat_triple.tail.properties,
                        }
                        relation.properties = {
                            **(
                                relation.properties
                                if relation.properties is not None
                                else {}
                            ),
                            **_stat_triple.relation.properties,
                        }

                        _triple = Triple(
                            head=head_node,
                            relation=relation,
                            tail=tail_node,
                        )
                        _semantic_triples.append(_triple.model_dump())

                        relation_id = self._create_db_triple(
                            session=session, triple=_triple
                        )
                        relation_ids.append(relation_id)

                    # Create `(System)->(Record)`` triple
                    system_node = Node(
                        name=str(record["properties"]["floc"]), type="System"
                    )

                    # Create `Record` node
                    record_node = Node(
                        name=str(record["properties"]["id"]),
                        type="Record",
                        properties={
                            **record["properties"],
                            "relationIds": relation_ids,
                        },
                    )

                    self._create_db_triple(
                        session=session,
                        triple=Triple(
                            head=system_node,
                            relation=Relation(name="HAS_RECORD"),
                            tail=record_node,
                        ),
                    )
                    system_triple_count += 1

                    # Assign record node to semantic nodes via MENTIONS
                    # _fortuitous_triples = []
                    for t in _semantic_triples:
                        for name in ["head", "tail"]:
                            record_triple_count += 1
                            _node = Node(**t[name])

                            # Remove properties from record_node (no need to duplicate)
                            # record_node.properties = None
                            _triple = Triple(
                                head=record_node,
                                relation=Relation(name="MENTIONS"),
                                # properties={"relationIds": relation_ids},
                                tail=_node,
                            )
                            # _fortuitous_triples.append(_triple.model_dump())
                            self._create_db_triple(session=session, triple=_triple)

            logger.info("Graph creation complete.")
            self.close()

        self.save("pmi_filtered_triples", "json", removed_triples)
        logger.info(
            f"""Removed {len(removed_triples)}/{semantic_triple_count}
            ({len(removed_triples)/semantic_triple_count*100:0.2f}%)
            due to PMI threshold"""
        )
        logger.info(f"System Triples: {system_triple_count}")
        logger.info(f"Record Triples: {record_triple_count}")

    def _cleanup(self: Self) -> None:
        """Clean up the files in the `output` directory."""
        try:
            path = self.settings.output_dir / self.folder_name
            shutil.rmtree(path)
            logger.info(f"Successfully deleted the folder: {path}")
        except Exception as e:
            logger.info(f"Error occurred while deleting the folder: {e}")

    def create(self: Self) -> None:
        """Create a MaintKG graph."""
        self._setup()
        self._extraction()
        self._refinement()
        self._build()

        if self.cleanup:
            self._cleanup()
