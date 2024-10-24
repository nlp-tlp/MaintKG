"""Extraction utilities."""

import json
from typing import Any, Dict, List, Optional

import requests
from loguru import logger


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
