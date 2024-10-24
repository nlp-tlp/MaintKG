"""General utilities."""

import math
import re
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from loguru import logger

from maintkg.models import Triple
from maintkg.utils.resources import corrections_dict


def preserve_chars(input_str: str, chars_to_preserve: List[str]) -> str:
    """Preserves alphanumeric characters and a list of specified characters in the input string.

    Parameters:
    - input_str :
        The string to process.
    - chars_to_preserve :
        A list of characters to preserve in the string, in addition to alphanumeric characters.

    Returns:
    - A string containing only the preserved characters.
    """
    # Create a pattern that matches characters to be preserved: alphanumeric and specified characters
    preserved_pattern = (
        f"[^a-zA-Z0-9 {''.join(re.escape(char) for char in chars_to_preserve)}]"
    )

    # Replace characters not in the preserved list with an empty string
    return re.sub(preserved_pattern, " ", input_str)


def correct_typos(text: str, corrections_dict: Dict[str, str]) -> str:
    """Corrects typos in a given string based on a mapping dictionary.

    Parameters:
    - text :
        The string containing potential typos.
    - corrections_dict :
    A dictionary mapping incorrect words to their correct versions.

    Returns:
    - The corrected string.

    Examples:
    >>> replacement_dict = {'craked': 'cracked'}
    >>> correct_typos("The wall was craked.", replacement_dict)
    "The wall was cracked."
    """
    # Sort corrections by length in descending order to replace longer phrases first
    corrections_sorted = sorted(
        corrections_dict.items(), key=lambda x: len(x[0]), reverse=True
    )

    corrected_text = text
    for incorrect, corrected in corrections_sorted:
        incorrect, corrected = str(incorrect), f" {str(corrected)} "
        replace = r"\b" + incorrect + r"\b"
        corrected_text = re.sub(replace, corrected, corrected_text)
    return corrected_text


def simple_preprocessing(
    text: str, non_semantic_codes: Optional[List[str]] = None
) -> str:
    """Perform simple preprocessing pipeline on a given text."""
    t = text
    # Replace characters with whitespace ($,)
    # Remove all characters except for alphanumerical and reserved special characters e.g. @, -, #, /, and .
    # Note: We're using regular expressions here, so some characters need to be 'escaped' as they are special 'metacharacters'
    # Refer to this for further information: https://www3.ntu.edu.sg/home/ehchua/programming/howto/Regexe.html
    # chars_to_keep = r"\,&\.\#\@\/-[]"

    if non_semantic_codes:
        regex_pattern = r"\b(" + "|".join(map(re.escape, non_semantic_codes)) + r")\b"
        t = re.sub(regex_pattern, " ", t, flags=re.IGNORECASE)

    chars_to_preserve = [
        ",",
        "&",
        ".",
        "#",
        "@",
        "/",
        "-",
        "[",
        "]",
        "(",
        ")",
    ]  # Excludes single quotes as these are used erroneously, although this introduces
    # technical LN errors e.g. 'won't' -> 'wont'

    t = t.replace(
        "'", ""
    )  # We don't want to add spaces around these as they are typically used contiguously.

    # fnc_rmv_chars = lambda text: regex.sub(rf"[^a-zA-Z0-9 {chars_to_keep}]", " ", text)
    # t = fnc_rmv_chars(t)
    t = preserve_chars(t, chars_to_preserve=chars_to_preserve)

    # Remove duplicate reserved special characters (like ##, @@@, etc.)
    # This way if we want to normalise # to number and we have ## we wont' get numbernumber
    _pattern = f"{''.join(re.escape(char) for char in chars_to_preserve)}"

    def fnc_rmv_dupe_chars(text: str) -> str:
        """Remove duplicate characters from a string."""
        return re.sub(rf"([{_pattern}])\1+", r"\1", text)

    t = fnc_rmv_dupe_chars(t)

    # Break any erroneous hyphenated or compound abbreviations
    # We want to keep things like 'o-ring' and 'c/o' but fix things like 'CV001-replace' → 'CV001 - replace'
    # and 'change-out' → 'change out', 'Plate/Lower' → 'plate / lower', 'inspect/replace' → 'inspect / replace'

    # fnc_fix_compound_hyphen = lambda text: regex.sub(r"(?<=\w{3,})-(?=\w{3,})", " - ", text)
    # fnc_fix_compound_fwd_slash = lambda text: regex.sub(
    #     r"(?<=\w{3,})\/(?=\w{3,})", " / ", text
    # )
    # fnc_fix_compound_chars = lambda text: fnc_fix_compound_fwd_slash(
    #     fnc_fix_compound_hyphen(text)
    # )

    # t = fnc_fix_compound_chars(t)

    # Dictionary normalisation (before modifying punctuation, etc.)
    t = correct_typos(text=t, corrections_dict=corrections_dict)

    # Replace alphanumeric identifiers with <id> tags
    t = re.sub(r"\b(?=\w*[a-zA-Z])(?=\w*\d)\w+\b", " ", t)  # " <id> "

    # Replace digits with special token <num>
    t = re.sub(r"\b\d+\b", " <num> ", t)

    # Remove punctuation at the end of the text
    t = t.rstrip(".")

    # Add space around commas
    t = t.replace(",", " , ")

    # Add space around periods ()
    t = t.replace(".", " ")

    # Add space around forward slashes
    # t = t.replace("/", " / ")

    # Replace brackets with parentheses
    t = t.replace("{", "(").replace("}", ")").replace("[", "(").replace("]", ")")

    # Add space around parentheses slashes
    t = t.replace("(", " ( ").replace(")", " ) ")

    # ampersand
    t = t.replace("&", " and ")

    # number
    t = t.replace("#", " number ")

    # at
    t = t.replace("@", " at ")

    # Remove training "at" # TODO: FIX
    t = t.replace(" at ", "") if t.rstrip().endswith(" at ") else t

    # selectively case words - if all upper - keep, if any lower - lower.
    # NOTE: This is not effective as MWOs can be entirely cased e.g. REPLACE ENGINE OIL which would be maintained.
    # t = " ".join([p if p.isupper() else p.lower() for p in t.split()])

    t = t.lower()

    # Dictionary normalisation (post punctuation normalisation, etc)
    t = correct_typos(text=t, corrections_dict=corrections_dict)

    # Remove superfluous whitespace
    # Corrected from \w+ to \s+ to target whitespace sequences
    return re.sub("\s+", " ", t).strip()


def calc_pmi_probabilities(triples: List[Triple]) -> Dict[str, Dict[str, dict]]:
    """Calculate Pointwise Mutual Information (PMI) probability for triples."""
    # Frequency dictionaries
    head_freq: DefaultDict = defaultdict(int)
    tail_freq: DefaultDict = defaultdict(int)
    relation_freq: DefaultDict = defaultdict(int)
    triple_freq: DefaultDict = defaultdict(int)
    node_freq: DefaultDict = defaultdict(int)
    total_triples: int = 0

    # Fill frequency dictionaries
    for triple in triples:
        head_freq[triple.head.name] += 1
        tail_freq[triple.tail.name] += 1
        relation_freq[triple.relation.name] += 1
        triple_freq[(triple.head.name, triple.relation.name, triple.tail.name)] += 1
        total_triples += 1

        node_freq[triple.head.name] += 1
        node_freq[triple.tail.name] += 1

    # Calculate probabilities
    head_prob = {k: v / total_triples for k, v in head_freq.items()}
    tail_prob = {k: v / total_triples for k, v in tail_freq.items()}
    relation_prob = {k: v / total_triples for k, v in relation_freq.items()}
    triple_prob = {k: v / total_triples for k, v in triple_freq.items()}
    node_prob = {k: v / total_triples for k, v in node_freq.items()}

    return {
        "frequency": {
            "head": head_freq,
            "tail": tail_freq,
            "relation": relation_freq,
            "triple": triple_freq,
            "node": node_freq,
        },
        "probability": {
            "head": head_prob,
            "tail": tail_prob,
            "relation": relation_prob,
            "triple": triple_prob,
            "node": node_prob,
        },
    }


def calculate_pmi(
    triple: Triple,
    triple_prob: dict,
    head_prob: dict,
    relation_prob: dict,
    tail_prob: dict,
) -> float:
    """Calculate the Pointwise Mutual Information (PMI) score for a given triple."""
    p_ijk = triple_prob[(triple.head.name, triple.relation.name, triple.tail.name)]
    p_i = head_prob[triple.head.name]
    p_k = relation_prob[triple.relation.name]
    p_j = tail_prob[triple.tail.name]
    return math.log(p_ijk / (p_i * p_k * p_j))


def plot_pmi_scores(pmi_scores: Union[List[float], np.array], file_path: str) -> None:
    """Plot and save a figure with a boxplot and histogram of PMI scores.

    Parameters:
    - pmi_scores :
        Array of PMI scores.
    - file_path :
        Path to save the figure.
    """
    # Set the style of the visualization
    sns.set(style="whitegrid")

    # Create a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
    f, (ax_box, ax_hist) = plt.subplots(
        2, sharex=True, gridspec_kw={"height_ratios": (0.15, 0.85)}
    )

    # Assigning a graph to each ax
    df = pd.DataFrame({"score": pmi_scores})

    # Both plots are now teal
    sns.boxplot(data=df, x="score", orient="h", ax=ax_box, color="teal")
    sns.histplot(data=df, x="score", ax=ax_hist, color="teal", kde=True)

    # Set titles and labels
    ax_hist.set_xlabel("PMI Score", fontsize=12)
    ax_hist.set_ylabel("Density", fontsize=12)

    # Remove x-axis name for the boxplot
    ax_box.set(xlabel="")

    # Improve layout and plot aesthetics
    plt.tight_layout()

    # Save the figure before showing it
    plt.savefig(file_path, dpi=300, format="png")

    # Show the plot
    # plt.show()

    # Close the plot
    plt.close()


def calculate_pmi_score_thresholds(data: List[float]) -> Tuple[float, float, float]:
    """Calculate the upper and lower thresholds based on the distribution of the data.

    Parameters:
    - data :
        The data for which to calculate the thresholds.

    Returns:
    - A tuple containing the median, lower threshold, and upper threshold.
    """
    # Calculate median, standard deviation, and quartiles
    median = np.median(data)
    std_dev = np.std(data)
    q1, q3 = np.percentile(data, [25, 75])

    # Perform Shapiro-Wilk test for normality
    shapiro_test = stats.shapiro(data)
    p_value = shapiro_test[1]

    logger.info(f"shapiro_test: {shapiro_test} (p : {p_value})")

    # Check if data is normally distributed with alpha = 0.05
    if p_value > 0.05:
        logger.info("Data is normally distributed")
        # Data is normally distributed
        lower_threshold = median - 2 * std_dev
        upper_threshold = median + 2 * std_dev
    else:
        logger.info("Data is NOT normally distributed")
        # Data is not normally distributed
        iqr = q3 - q1
        lower_threshold = q1 - 1.5 * iqr
        upper_threshold = q3 + 1.5 * iqr
        logger.info(f"Q1 {q1} Q3 {q3} IQR {iqr}")

    return median, lower_threshold, upper_threshold
