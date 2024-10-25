""""""

import json
from collections import Counter
from typing import Dict, List, Set, Tuple, Union

import torch
from loguru import logger
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset
from transformers.optimization import AdamW
from typing_extensions import Self

from noisie.scheduler import get_inverse_square_root_schedule_with_warmup

arg_to_scheduler = {
    "inverse_square_root": get_inverse_square_root_schedule_with_warmup,
}

entity_scheme = {"<object>", "<process>", "<state>", "<activity>", "<property>"}
relation_scheme = {
    "has agent",
    "has patient",
    "is a",
    "has part",
    "has property",
    "contains",
    "has participant",
}
relation_rules = [
    ("<object>", "<object>"),
    ("<activity>", "<object>"),
    ("<process>", "<object>"),
    ("<state>", "<object>"),
    ("<activity>", "<activity>"),
    ("<activity>", "<state>"),
    ("<activity>", "<process>"),
    ("<object>", "<property>"),
]


def process_triple(parts: tuple) -> Tuple[str, str, str, str, str]:
    """"""
    # Initialize variables to hold the parts of the triple
    head, head_type, tail, tail_type = "", "", "", ""
    relation = []
    processing_part = "head"  # Start with processing the head

    for part in parts:
        if part.startswith("<"):
            # This part is a type
            if processing_part == "head":
                head_type = part
                processing_part = "tail"  # Move to processing the tail next
            elif processing_part == "tail":
                tail_type = part
                processing_part = "relation"  # Move to processing the relation next
        else:
            # This part is either head, tail, or relation
            if processing_part == "head":
                head = f"{head} {part}".strip()
            elif processing_part == "tail":
                tail = f"{tail} {part}".strip()
            elif processing_part == "relation":
                relation.append(part)

    # Join the relation parts into a single string
    relation_str = " ".join(relation)

    return head, head_type, tail, tail_type, relation_str


def parse_and_categorize_sections(
    input_str: str,
) -> Dict[str, Union[List[Tuple[int, int]], List[str]]]:
    """Parse a structured string to categorise parts.

    Categorises a structured string into `norm`, `relation`, and `entity` sections.

    Parameters
    ----------
    input_str :
        A structured string with markers indicating `norm`, `relation`, and `entity` sections.

    Returns
    -------
    Dict[str, Union[List[Tuple[int, int]], List[str]]] :
        A dictionary with a 'spans' key containing a dictionary for `norm`, `relation`, and `entity`,
        each of which is a list of tuples indicating the span's positions (start, end) in the input.
        Also includes a 'tokens' key that lists all the tokens from the input string.

    Raises
    ------
    Exception :
        If an error occurs while parsing and categorising the sections.
    """
    parts: List[str] = input_str.split()
    spans: Dict[str, List[str]] = {"norm": [], "relation": [], "entity": []}
    current_span_type: List[str] = None
    start_index: int = None

    for idx, part in enumerate(parts):
        try:
            if part.startswith(("<norm>", "<relation>", "<entity>")):
                if start_index is not None:  # Close the previous span if it exists
                    spans[current_span_type].append((start_index, idx - 1))
                # Update the span type and start index for the new span
                current_span_type = part[1:-1]
                start_index = idx + 1
            elif idx == len(parts) - 1:  # Handle the last token as a special case
                spans[current_span_type].append((start_index, idx))
        except Exception as e:
            logger.error(f"Failed to parse and catagorize section. Error: {e}")
            raise
    return {"spans": spans, "tokens": parts}


def parse_abbreviation_full_form(
    input_str: str,
) -> List[Tuple[List[str], Union[List[str], str]]]:
    """Parse a string containing abbreviations and their full forms, separating them into pairs.

    The input string should contain abbreviations followed by their full forms, with each full form
    enclosed in square brackets. This function splits the string into parts where each abbreviation is
    followed by its corresponding full form. Spaces outside square brackets are treated as delimiters,
    while spaces inside square brackets are kept as part of the full form.

    Parameters
    ---------
    input_str :
        A string containing abbreviations and their full forms. Each full form must be
        enclosed in square brackets immediately following its abbreviation.
        For example: "NASA [National Aeronautics and Space Administration] ESA [European Space Agency]".

    Returns
    -------
    list of tuples :
        A list where each tuple contains an abbreviation and its full form as two elements.
        If an abbreviation does not have a corresponding full form in the input string,
        its full form is returned as an empty string.

    Example
    -------
    >>> parse_abbreviation_full_form("NASA [National Aeronautics and Space Administration] ESA [European Space Agency]")
    [('NASA', 'National Aeronautics and Space Administration'), ('ESA', 'European Space Agency')]

    Note
    ----
    - The function assumes that the input string is well-formed according to the described format.
    - If the input does not strictly follow the format, the output may not accurately reflect the intended
      abbreviation-full form pairs.
    """
    # Split the input string by spaces, keeping content inside square brackets intact
    parts: List[str] = []
    buffer: str = ""
    inside_brackets: bool = False
    for char in input_str:
        if char == "[":
            if buffer:
                parts.append(buffer.strip())
                buffer = ""
            inside_brackets = True
        elif char == "]":
            inside_brackets = False
            parts.append(buffer.strip())
            buffer = ""
        else:
            if not inside_brackets or (inside_brackets and char != " "):
                buffer += char
            elif inside_brackets and char == " ":
                buffer += char  # Include spaces inside brackets
    if buffer:  # Add any remaining part
        parts.append(buffer.strip())

    abbreviations: List[Tuple[List[str], Union[List[str], str]]] = []
    for i in range(0, len(parts), 2):
        abbreviation = parts[i]
        full_form = parts[i + 1] if (i + 1) < len(parts) else ""
        abbreviations.append((abbreviation, full_form))

    return abbreviations


def filter_relations_on_rules(
    relations: List[List[str]], rules: List[Tuple[str, str]]
) -> Tuple[List[List[str]], int]:
    """Filter a list of relationships based on a set of rules.

    Parameters:
    - relations: A list of relationships, where each relationship is represented as a list:
                 [head, head_type, tail, tail_type, relation]
    - rules: A list of tuples representing the allowed (head_type, tail_type) pairs.

    Returns:
    - A tuple containing the filtered list of relationships and the count of filtered out relations.

    Example:
    >>> relations = [["engine", "<object>", "engine", "<object>", "hasPart"]]
    >>> rules = [("<object>", "<object>")]
    >>> filter_relations_on_rules(relations, rules)
    ([["engine", "<object>", "engine", "<object>", "hasPart"]], 0)
    """
    filtered_relations = [
        relation for relation in relations if (relation[1], relation[3]) in rules
    ]
    issues = len(relations) - len(filtered_relations)
    return filtered_relations, issues


def filter_relations_on_self_referencing(
    relations: List[List[str]],
) -> Tuple[List[List[str]], int]:
    """Filter relations where the `head` is equal to the `tail` and the `head_type` is equal to the `tail_type`.

    Parameters
    ---------
    relations :
        A list of relationships, where each relationship is represented as a list:
        [head, head_type, tail, tail_type, relation]

    Returns
    -------
    Tuple[List[List[str]], int] :
        A tuple containing the filtered list of relationships and the count of self-referencing relations filtered out.
    """
    filtered_relations = [
        relation
        for relation in relations
        if not (relation[0] == relation[2] and relation[1] == relation[3])
    ]
    issues = len(relations) - len(filtered_relations)
    return filtered_relations, issues


def filter_normalisations(
    norms: List[tuple], input_str: str
) -> Tuple[List[tuple], Dict[str, int]]:
    """Filter out invalid normalisations based on the input string.

    norm [["srv", "srv"]]
    """
    issues: Dict[str, int] = {}

    # Filter out duplicates
    filtered_norms_duplicates = list(set(norms))
    issues["duplicates"] = len(norms) - len(filtered_norms_duplicates)

    # Filter out self-references e.g. ["repl", "repl"]
    filtered_norms_self_reference = [
        d for d in filtered_norms_duplicates if not d[0] == d[1]
    ]
    issues["self_references"] = len(filtered_norms_duplicates) - len(
        filtered_norms_self_reference
    )

    # Filter out pairs that have OOV forms outside of the input_vocab
    filtered_norms_oov_validation = [
        d for d in filtered_norms_self_reference if d[0] in input_str
    ]
    issues["invalid_oov"] = len(filtered_norms_self_reference) - len(
        filtered_norms_oov_validation
    )

    return filtered_norms_oov_validation, issues


def filter_entities(
    entities: List[tuple], vocab=Set[str]
) -> Tuple[List[tuple], Dict[str, int]]:
    """Filter entities based on the input vocab and entity scheme."""
    issues: Dict[str, int] = {}
    # Filter out duplicates
    filtered_entities_duplicates = list(set(entities))
    issues["duplicates"] = len(entities) - len(filtered_entities_duplicates)

    # Filter out entities with invalid types (outside schema definition)
    filtered_entities_invalid_type = [
        d for d in filtered_entities_duplicates if d[1] in entity_scheme
    ]
    issues["type_invalidation"] = len(filtered_entities_duplicates) - len(
        filtered_entities_invalid_type
    )
    # Filter out any entities surface forms that have hallucinated tokens
    # TODO
    return filtered_entities_invalid_type, issues


def filter_relations(
    relations: List[tuple], vocab=Set[str]
) -> Tuple[List[tuple], Dict[str, int]]:
    """Filter relations based on the input vocab and relation scheme."""
    issues: Dict[str, int] = {}

    # Filter out duplicates
    filtered_relations_duplicates = list(set(relations))
    issues["duplicates"] = len(relations) - len(filtered_relations_duplicates)

    # Filter out self-referencing relations
    (
        filtered_relations_self_reference,
        issues_relation_self_reference,
    ) = filter_relations_on_self_referencing(filtered_relations_duplicates)
    issues["self_references"] = issues_relation_self_reference

    # Filter relations based on rules
    filtered_relations_invalid_rules, issues_relation_rules = filter_relations_on_rules(
        filtered_relations_self_reference, rules=relation_rules
    )

    issues["rule_invalidation"] = issues_relation_rules

    # Filter relations based on invalid types
    filtered_relations_invalid_type = [
        r
        for r in filtered_relations_invalid_rules
        if (r[1] in entity_scheme and r[3] in entity_scheme and r[4] in relation_scheme)
    ]
    issues["type_invalidation"] = len(filtered_relations_invalid_rules) - len(
        filtered_relations_invalid_type
    )
    # Filter out any head/tail surface forms that have hallucinated tokens
    # TODO
    return filtered_relations_invalid_type, issues


def structure_noisie_string(input_str: str, noisie_str: str):
    """Structure a NoisIE string and perform validation and sanitisation on the output.

    Input and canonical tokens from the normalisation component are used to validate the head/tail of relations and the surface forms of entities.

    Parameters
    ----------
    input_str :
        The input string that was used to generate the NoisIE output.
    noisie_str :
        The NoisIE output string that contains the structured information.
    """
    input_tokens: List[str] = input_str.split()
    text_sections = parse_and_categorize_sections(noisie_str)
    text_spans = text_sections["spans"]
    text_tokens = text_sections["tokens"]
    data = {
        "norms": [],
        "entities": [],
        "relations": [],
        "issues": {"norms": {}, "entities": {}, "relations": {}, "general": {}},
    }
    try:
        for cat, spans in text_spans.items():
            for start, end in spans:
                span_tokens = tuple(text_tokens[start : end + 1])

                if cat == "norm":
                    norm_string = " ".join(text_tokens[start : end + 1])
                    norm_parts = parse_abbreviation_full_form(norm_string)
                    data["norms"] = norm_parts
                elif cat == "relation":
                    ie_item = tuple(text_tokens[start : end + 1])
                    data["relations"].append(process_triple(ie_item))
                elif cat == "entity":
                    data["entities"].append(span_tokens)

        norm_iv_tokens = [
            n[1] for n in data["norms"]
        ]  # extract the canonical tokens generated by the model
        norm_oov_tokens = set(
            [n[0] for n in data["norms"]]
        )  # extract the non-canonical tokens generated by the model

        # Creat vocab and remove OOV tokens from vocab (e.g. remove bad words from input str)
        vocab = set(input_tokens + norm_iv_tokens) - norm_oov_tokens

        # Sanitise output components - self-referencing, rules, scheme adherence, duplication.
        for key in ["norms", "entities", "relations"]:
            if key == "norms":
                filtered_norms, issues_norms = filter_normalisations(
                    norms=data[key], input_str=input_str
                )
                data[key] = filtered_norms
                data["issues"][key] = issues_norms

            elif key == "entities":
                filtered_entities, issues_entities = filter_entities(
                    entities=data[key], vocab=vocab
                )
                data[key] = filtered_entities
                data["issues"][key] = issues_entities

            elif key == "relations":
                filtered_relations, issues_relations = filter_relations(
                    relations=data[key], vocab=vocab
                )
                data[key] = filtered_relations
                data["issues"][key] = issues_relations
        return data
    except Exception as e:
        # TODO: investigate this issue.
        print(f"String structuring exception: {e}")
        data["issues"]["general"] = {"fatal_error": 1}
        return data


def evaluate_model(
    data: List[Dict],
) -> Tuple[
    Dict[str, Dict[str, Union[int, float]]], Dict[str, List[Tuple[int, List[str]]]]
]:
    """Evaluate the performance of a model by comparing its predictions with the ground truth labels.

    Parameters
    ----------
    data :
        Each data dict contains 'predictions' and 'labels' for norms, relations, and entities.

    Returns
    -------
    Tuple[Dict[str, Dict[str, Union[int, float]]], Dict[str, List[Tuple[int, List[str]]]]] :
        Contains the metrics dict with accuracy scores for each category and dicts for incorrect norms,
        relations, and entities with item indices.
    """
    metrics = {
        "loose_relation": {"correct": 0, "total": 0},
        "strict_relation": {"correct": 0, "total": 0},
        "loose_entity": {"correct": 0, "total": 0},
        "strict_entity": {"correct": 0, "total": 0},
        "normalisation": {"correct": 0, "total": 0},
    }

    incorrect_predictions = {
        "norms": [],
        "relations": {"loose": [], "strict": []},
        "entities": {"loose": [], "strict": []},
    }

    for idx, item in enumerate(data):
        predictions, labels = item["predictions"], item["labels"]

        # Evaluate normalisation
        pred_norms = Counter(tuple(norm) for norm in predictions["norms"])
        label_norms = Counter(tuple(norm) for norm in labels["norms"])
        metrics["normalisation"]["correct"] += sum((pred_norms & label_norms).values())
        metrics["normalisation"]["total"] += len(labels["norms"])
        incorrect_norms = pred_norms - label_norms
        if incorrect_norms:
            incorrect_predictions["norms"].append((idx, list(incorrect_norms)))

        # Evaluate relations
        for label_relation in labels["relations"]:
            label_loose = tuple(
                [label_relation[0], label_relation[2], label_relation[-1]]
            )
            metrics["strict_relation"]["total"] += 1
            metrics["loose_relation"]["total"] += 1
            relation_found_loose = False
            relation_found_strict = False

            for pred_relation in predictions["relations"]:
                pred_loose = tuple(
                    [pred_relation[0], pred_relation[2], pred_relation[-1]]
                )
                if pred_loose == label_loose:
                    relation_found_loose = True
                    metrics["loose_relation"]["correct"] += 1
                    if pred_relation == label_relation:
                        relation_found_strict = True
                        metrics["strict_relation"]["correct"] += 1
                        break

            if not relation_found_loose:
                incorrect_predictions["relations"]["loose"].append(
                    (idx, label_relation)
                )
            if not relation_found_strict:
                incorrect_predictions["relations"]["strict"].append(
                    (idx, label_relation)
                )

        # Evaluate entities
        for label_entity in labels["entities"]:
            label_loose = label_entity[0]
            metrics["strict_entity"]["total"] += 1
            metrics["loose_entity"]["total"] += 1
            entity_found_loose = False
            entity_found_strict = False

            for pred_entity in predictions["entities"]:
                pred_loose = pred_entity[0]
                if pred_loose == label_loose:
                    entity_found_loose = True
                    metrics["loose_entity"]["correct"] += 1
                    if pred_entity == label_entity:
                        entity_found_strict = True
                        metrics["strict_entity"]["correct"] += 1
                        break

            if not entity_found_loose:
                incorrect_predictions["entities"]["loose"].append((idx, label_entity))
            if not entity_found_strict:
                incorrect_predictions["entities"]["strict"].append((idx, label_entity))

    # Calculate scores
    for key, value in metrics.items():
        if value["total"] > 0:
            metrics[key]["score"] = value["correct"] / value["total"]
        else:
            metrics[key]["score"] = None

    return metrics, incorrect_predictions


class TranslationDataset(Dataset):
    """Dataset class for translation tasks."""

    def __init__(self: "TranslationDataset", encodings: Dict[str, List[float]]):
        """Initialise the dataset with the given encodings."""
        self.encodings = encodings

    def __getitem__(self: Self, idx: int) -> Dict[str, torch.Tensor]:
        """Return the item at the given index."""
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self: Self) -> int:
        """Return the length of the dataset."""
        return len(self.encodings.input_ids)


class ByT5Model(LightningModule):
    """LightningModule for the ByT5 model."""

    def __init__(
        self: "ByT5Model",
        model,
        config=None,
        tokenizer=None,
        output_dir=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.config = config

        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def generate(self, input_ids, attention_mask, **generate_kwargs):
        return self.model.generate(
            input_ids, attention_mask=attention_mask, **generate_kwargs
        )

    def forward(self, inputs, labels) -> dict:
        outputs = self.model(input_ids=inputs["input_ids"], labels=labels)
        logits = outputs["logits"]
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        output_dict = {"loss": loss, "logits": logits}

        return output_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        labels = batch.pop("labels")
        labels_original = labels.clone()

        batch["decoder_input_ids"] = torch.where(
            labels != -100, labels, self.config.pad_token_id
        )
        # labels = shift_tokens_left(labels, -100)
        labels[labels == self.config.pad_token_id] = -100

        forward_output = self.forward(batch, labels)
        self.log("loss", forward_output["loss"])
        batch["labels"] = labels_original
        # self.train_step_outputs.append(forward_output["loss"])

        # print(f'Train step loss: {forward_output["loss"]}')

        return forward_output["loss"]

    def val_test_step(self, batch: dict, return_labels: bool = False):
        gen_kwargs = {
            "max_length": self.config.max_length,
            "early_stopping": False,
            "no_repeat_ngram_size": 0,
            # "length_penalty": 0,
            "num_beams": 1,
        }
        generated_tokens = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **gen_kwargs,
        )
        decoded_preds = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        # print("\n Predictions:")
        # for pred in decoded_preds:
        #     print(pred, "\n")

        labels = batch.pop("labels")
        batch["decoder_input_ids"] = torch.where(
            labels != -100, labels, self.config.pad_token_id
        )

        with torch.no_grad():
            # compute loss on predict data
            forward_output = self.forward(batch, labels)

        decoded_labels = self.tokenizer.batch_decode(
            torch.where(labels != -100, labels, self.config.pad_token_id),
            skip_special_tokens=True,
        )
        if return_labels:
            return (
                decoded_preds,
                decoded_labels,
                float(forward_output["loss"].mean().detach()),
            )
        return decoded_preds, float(forward_output["loss"].mean().detach())

    def validation_step(self, batch: dict):
        decoded_preds, loss = self.val_test_step(batch)

        print("\n Predictions:")
        for pred in decoded_preds:
            print("\n", pred)
            # TODO: add input_str into structure fnc
            # print(structure_noisie_string(pred), "\n")

        self.log("val_loss", loss)

    def test_step(self, batch: dict):
        decoded_preds, decoded_labels, loss = self.val_test_step(
            batch, return_labels=True
        )

        outputs = {}
        # TODO: add input_str into structure fnc
        outputs[
            "predictions"
        ] = []  # [structure_noisie_string(p) for p in decoded_preds]
        outputs["labels"] = []  # [structure_noisie_string(l) for l in decoded_labels]

        outputs["inputs"] = self.tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )

        self.test_step_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        output = self.test_step_outputs

        # Format
        flat_data = [
            {"input": inp, "predictions": pred, "labels": labels}
            for item in output
            for pred, inp, labels in zip(
                item["predictions"], item["inputs"], item["labels"]
            )
        ]

        # Calc metrics
        metrics, incorrect_predictions = evaluate_model(data=flat_data)

        for metric, result in metrics.items():
            print(
                f"{metric}: {result.get('score', 0):.2f} ({result['correct']}/{result['total']})"
            )

        if self.output_dir == None:
            raise AssertionError("Output Directory not supplied (None)")

        with open(f"{self.output_dir}/test.json", "w") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "output": flat_data,
                    "incorrect_preds": incorrect_predictions,
                },
                f,
                indent=2,
            )

    def generate_samples(self, batch: dict):
        """Sample generation for inference."""

        gen_kwargs = {
            "max_length": self.config.max_length,
            "early_stopping": False,
            "no_repeat_ngram_size": 0,
            # "length_penalty": 0,
            "num_beams": 1,
        }

        generated_tokens = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **gen_kwargs,
        )
        decoded_preds = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        inputs = self.tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )

        return inputs, decoded_preds

    def configure_optimizers(self):
        """
        FROM PYTORCH LIGHTNING DOCUMENTATION

        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.

            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,  # self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "betas": (0.9, 0.999),
            "eps": 0.00000001,
            # "betas": (self.hparams.adam_beta1, self.hparams.adam_beta2),
            # "eps": self.hparams.adam_epsilon,
        }

        optimizer_kwargs["lr"] = 0.00005  # self.hparams.learning_rate

        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        lr_scheduler = self._get_lr_scheduler(
            1200000, optimizer  # self.hparams.max_steps,
        )

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def _get_lr_scheduler(self, num_training_steps, optimizer):
        schedule_func = arg_to_scheduler[
            "inverse_square_root"
            # self.hparams.lr_scheduler
        ]
        scheduler = schedule_func(
            optimizer,
            num_warmup_steps=1000,
            #   num_warmup_steps=self.hparams.warmup_steps
        )
        return scheduler
