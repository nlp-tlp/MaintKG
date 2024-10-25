"""Extraction utilities."""

import shelve
from typing import Any, Dict, List, Tuple

import pandas as pd
from loguru import logger

from maintkg.models import Prediction, RecordWithPreds
from noisie._base import NoisIE


def load_cached_predictions(
    inputs: List[str], cache_file: str
) -> Tuple[Dict[str, Prediction], List[str]]:
    """Load existing predictions from cache and identify new inputs."""
    all_preds: Dict[str, Prediction] = {}
    new_inputs: List[str] = []

    try:
        logger.info("Loading cached predictions")
        with shelve.open(cache_file) as shelf:
            for input_ in inputs:
                if input_ in shelf:
                    all_preds[input_] = shelf[input_]
                else:
                    new_inputs.append(input_)
        logger.info(f"Loaded {len(all_preds)} predictions from cache")
        return all_preds, new_inputs
    except Exception as e:
        logger.error(f"Error loading cache: {e}")
        raise IOError(f"Failed to load cache: {e}") from e


def process_new_predictions(
    model: NoisIE, batches: List[List[str]], cache_file: str
) -> Tuple[Dict[str, Prediction], List[Dict[str, Any]]]:
    """Process new inputs through the IE model and cache results."""
    all_preds: Dict[str, Prediction] = {}
    failed_predictions: List[Dict[str, Any]] = []

    for batch in batches:
        preds = model.inference(data=batch)

        if preds is None:
            raise RuntimeError("Failed to get predictions. Check service.")
        if len(preds) != len(batch):
            raise RuntimeError("Misaligned texts and predictions")

        with shelve.open(cache_file, writeback=True) as shelf:
            logger.info(f"Adding {len(preds)} predictions to cache")
            for idx, pred in enumerate(preds):
                if pred["passed"]:
                    shelf[batch[idx]] = pred
                    all_preds[batch[idx]] = pred
                else:
                    failed_predictions.append(pred)

    return all_preds, failed_predictions


def create_output_records(
    data: pd.DataFrame, cache_file: str
) -> Tuple[Dict[str, List[RecordWithPreds]], Dict[str, List[Any]], Dict[str, int]]:
    """Create final output records with predictions."""
    output: Dict[str, List[RecordWithPreds]] = {}
    summary: Dict[str, List[Any]] = {"norms": [], "entities": [], "triples": []}
    stats = {"texts_no_ie": 0, "texts_entity_only": 0}

    with shelve.open(cache_file) as shelf:
        for record in data.to_dict(orient="records"):
            floc = record["floc"]
            preds = shelf[record["input"]]

            entities = preds["entities"]
            triples = preds["relations"]

            # Update statistics
            if not triples and not entities:
                stats["texts_no_ie"] += 1
            elif not triples and entities:
                stats["texts_entity_only"] += 1

            record_with_preds = RecordWithPreds(
                properties=record,
                input=record["input"],
                output=preds["gen_output"],
                preds={
                    "norms": preds["norms"],
                    "entities": entities,
                    "triples": triples,
                    "issues": preds["issues"],
                },
            )

            output.setdefault(floc, []).append(record_with_preds)

            # Update summary
            summary["norms"].append(preds["norms"])
            summary["entities"].append(entities)
            summary["triples"].append(triples)

    return output, summary, stats
