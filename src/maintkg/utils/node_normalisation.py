"""Node normalisation utilities."""

import re
from typing import List, Tuple

from nltk.stem import WordNetLemmatizer

from maintkg.utils.resources import colloquialisms, entity_tag_to_pos, negations

wnl = WordNetLemmatizer()


def standardise_negations(s: str) -> str:
    """Standardise negations."""
    # Iterate over the dictionary items
    for key, value in negations.items():
        # Use a regular expression to replace all occurrences,
        # considering word boundaries
        s = re.sub(r"\b{}\b".format(re.escape(key)), value, s)
    return s


def standardise_colloquialism(s: str) -> str:
    """Standardise colloquialism."""
    tokens = s.split()
    word = tokens[-1]
    new_word = colloquialisms.get(word)
    if new_word:
        return " ".join(tokens[:-1] + [new_word])
    return s


def lemmatize(s: str, tag: str) -> str:
    """Lemmatize."""
    tokens = s.split()
    word = tokens[-1]
    pos = entity_tag_to_pos.get(tag)
    new_word = wnl.lemmatize(word, pos)
    return " ".join(tokens[:-1] + [new_word])


def normalise(s: str, tag: str) -> Tuple[str, List[str], bool]:
    """Normalise a string given its semantic tag (object, activity, ...)."""
    _ops = []

    tag = tag.lower()
    assert (
        tag in entity_tag_to_pos.keys()
    ), f"Tag ({tag}) not valid. Expected {entity_tag_to_pos.keys()}"

    s1 = standardise_colloquialism(s)
    if s != s1:
        _ops.append("colloquialism")

    if tag == "state":
        s2 = standardise_negations(s1)
        if s1 != s2:
            _ops.append("negation")
        s3 = lemmatize(s2, tag)
        if s2 != s3:
            _ops.append("lemmatize")

        return s3, _ops, len(_ops) > 0
    s2 = lemmatize(s1, tag)
    if s1 != s2:
        _ops.append("lemmatize")
    return s2, _ops, len(_ops) > 0
