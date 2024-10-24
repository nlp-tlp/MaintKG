"""Extraction utilities."""

import json
import shelve
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from loguru import logger

from maintkg.models import Prediction, RecordWithPreds


def perform_ie(texts: List[str]) -> Optional[Any]:
    """Perform information extraction via the NoisIE API.

    Ensure that this is running by visiting the NoisIE directory:
    `./NoisIE/maintnormie` and running `uvicorn server:app`.
    First ensure the venv is activated by going into `./rebel/rebel-venv`

    Parameters
    ----------
    texts
        List of texts to process

    Returns
    -------
    Optional[List[Dict[str, Any]]]
        The processed results if successful, None if there was an error.

    """
    url = "http://localhost:8000/predict/"  # TODO: settings.IE_MODEL_INFERENCE_API_ENDPOINT
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    data = {"texts": texts}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # This will raise an exception for HTTP error codes
        logger.debug("Success!")
        result: List[Dict[str, Any]] = response.json()
        return result

    except requests.exceptions.HTTPError as errh:
        logger.debug(response.json())
        logger.debug("Http Error:", errh)
    except requests.exceptions.ConnectionError as errc:
        logger.debug(
            "Error Connecting:", errc
        )  # This will catch errors related to the server being offline
    except requests.exceptions.Timeout as errt:
        logger.debug("Timeout Error:", errt)
    except requests.exceptions.RequestException as err:
        logger.debug("Oops: Something Else", err)

    return None  # Return None for all error cases


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
    batches: List[List[str]], cache_file: str
) -> Tuple[Dict[str, Prediction], List[Dict[str, Any]]]:
    """Process new inputs through the IE model and cache results."""
    all_preds: Dict[str, Prediction] = {}
    failed_predictions: List[Dict[str, Any]] = []

    for batch in batches:
        preds = perform_ie(texts=batch)

        if preds is None:
            raise RuntimeError("Failed to get predictions. Check service.")
        if len(preds["predictions"]) != len(batch):
            raise RuntimeError("Misaligned texts and predictions")

        with shelve.open(cache_file, writeback=True) as shelf:
            logger.info(f'Adding {len(preds["predictions"])} predictions to cache')
            for idx, pred in enumerate(preds["predictions"]):
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
