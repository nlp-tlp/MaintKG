import uuid
import random
import string
import json
import re

# TODO:
# - Need to capture that has_part and contains (and their inverses) cannot hold between two of the same individual classes.

RELATION_CLASSES = {
    "has_part": {
        "domain": ["PhysicalObject"],
        "range": ["PhysicalObject"],
    },
    "is_part_of": {
        "inverse_of": "has_part",
        "domain": ["PhysicalObject"],
        "range": ["PhysicalObject"],
    },
    # "is_a_property": {"parent": "is_a", "domain": ["Property"], "range": ["Property"]},
    # "is_a_process": {"parent": "is_a", "domain": ["Process"], "range": ["Process"]},
    "has_agent": {},
    "is_agent_of": {"inverse_of": "has_agent"},
    # "state_has_agent": {
    #     "parent": "has_agent",
    #     "domain": ["State"],
    #     "range": ["PhysicalObject"],
    # },
    # "state_is_agent_of": {"inverse_of": "state_has_agent", "parent": "is_agent_of"},
    # "activity_has_agent": {
    #     "parent": "has_agent",
    #     "domain": ["Activity"],
    #     "range": ["PhysicalObject"],
    # },
    # "activity_is_agent_of": {
    #     "inverse_of": "activity_has_agent",
    #     "parent": "is_agent_of",
    # },
    # "process_has_agent": {
    #     "parent": "has_agent",
    #     "domain": ["Process"],
    #     "range": ["PhysicalObject"],
    # },
    # "process_is_agent_of": {"inverse_of": "process_has_agent", "parent": "is_agent_of"},
    "has_patient": {},
    "is_patient_of": {"inverse_of": "has_patient"},
    # "state_has_patient_physicalobject": {
    #     "parent": "has_patient",
    #     "domain": ["State"],
    #     "range": ["PhysicalObject"],
    # },
    # "physicalobject_is_patient_of_state": {
    #     "parent": "is_patient_of",
    #     "inverse_of": "state_has_patient_physicalobject",
    # },
    # "state_has_patient_activity": {
    #     "parent": "has_patient",
    #     "domain": ["State"],
    #     "range": ["Activity"],
    # },
    # "activity_has_patient_physicalobject": {
    #     "parent": "has_patient",
    #     "domain": ["Activity"],
    #     "range": ["PhysicalObject"],
    # },
    # "physicalobject_is_patient_of_activity": {
    #     "parent": "is_patient_of",
    #     "inverse_of": "activity_has_patient_physicalobject",
    # },
    # "activity_has_patient_activity": {
    #     "parent": "has_patient",
    #     "domain": ["Activity"],
    #     "range": ["Activity"],
    # },
    # "activity_has_patient_state": {
    #     "parent": "has_patient",
    #     "domain": ["Activity"],
    #     "range": ["State"],
    # },
    # "activity_has_patient_process": {
    #     "parent": "has_patient",
    #     "domain": ["Activity"],
    #     "range": ["Process"],
    # },
    # "process_has_patient_physicalobject": {
    #     "parent": "has_patient",
    #     "domain": ["Process"],
    #     "range": ["PhysicalObject"],
    # },
    "contains": {
        "domain": ["PhysicalObject"],
        "range": ["PhysicalObject"],
    },
    "is_contained_by": {
        "inverse_of": "contains",
        "domain": ["PhysicalObject"],
        "range": ["PhysicalObject"],
    },
    "has_property": {
        "domain": ["PhysicalObject"],
        "range": ["Property"],
    },
    "is_property_of": {
        "inverse_of": "has_property",
        "domain": ["Property"],
        "range": ["PhysicalObject"],
    },
}


PROPERTY_CHAIN_AXIOMS = {
    "can_be_patient_of_state_due_to_part": [
        "has_part",
        "physicalobject_is_patient_of_state",
    ],
    "can_be_patient_of_activity_due_to_part": [
        "has_part",
        "physicalobject_is_patient_of_activity",
    ],
}


def convert_format(input_string, include_brackets: bool = True):
    # Split the string by '/' and take the last part
    last_part = input_string.split("/")[-1]

    # Insert spaces before capital letters and convert to lowercase
    formatted_string = re.sub(r"(?<!^)(?=[A-Z])", " ", last_part).lower()

    return f"<{formatted_string}>" if include_brackets else f"{formatted_string}"


def transform_data(data, include_brackets: bool = True):
    result = []

    for item in data:
        # Transform the current item
        transformed_item = {
            "onto_name": item["name"],
            "annotation_name": convert_format(item["name"], include_brackets),
            "children": transform_data(
                item["children"], include_brackets
            ),  # Recursively transform children
        }
        result.append(transformed_item)

    return result


def process_maintie_scheme():
    with open("./maintie_scheme.json", "r") as f:
        scheme = json.load(f)

    entity_scheme = scheme["entity"]
    relation_scheme = scheme["relation"]

    relation_scheme_transformed = transform_data(
        relation_scheme, include_brackets=False
    )
    entity_scheme_transformed = transform_data(entity_scheme)
    return entity_scheme_transformed, relation_scheme_transformed


ENTITY_CLASSES, _ = process_maintie_scheme()

RELATION_PATTERN_MAPPING = {
    (
        "<state>",
        "has_patient",
        "<physical object>",
    ): "state_has_patient_physicalobject",
    ("<state>", "has_patient", "<activity>"): "state_has_patient_activity",
    (
        "<process>",
        "has_patient",
        "<physical object>",
    ): "process_has_patient_physicalobject",
    (
        "<activity>",
        "has_patient",
        "<physical object>",
    ): "activity_has_patient_physicalobject",
    ("<activity>", "has_patient", "<activity>"): "activity_has_patient_activity",
    ("<activity>", "has_patient", "<state>"): "activity_has_patient_state",
    ("<activity>", "has_patient", "<process>"): "activity_has_patient_process",
    ("<state>", "has_agent", "<physical object>"): "state_has_agent",
    ("<activity>", "has_agent", "<physical object>"): "activity_has_agent",
    ("<process>", "has_agent", "<physical object>"): "process_has_agent",
    # ("<process>", "is_a", "<process>"): "is_a_process",
    # ("<activity>", "is_a", "<activity>"): "is_a_activity",
    # ("<state>", "is_a", "<state>"): "is_a_state",
    # ("<physical object>", "is_a", "<physical object>"): "is_a_physical_object",
    # ("<property>", "is_a", "<property>"): "is_a_property",
}


def find_individual_by_label(onto, label, entity_class):
    # Helper function to find an individual by label
    for individual in onto.get_instances_of(entity_class):
        if label in individual.label:
            return individual
    return None


def generate_short_id(length=8):
    # Generates a random string of uppercase letters and digits
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def flatten_entity_classes(entity_classes):
    flat_dict = {}

    for class_data in entity_classes:
        onto_name = class_data["onto_name"]
        annotation_name = class_data["annotation_name"]

        # Add the ontology name and annotation name to the flat dictionary
        flat_dict[annotation_name] = onto_name

        # If there are children, recursively flatten them
        children = class_data.get("children", [])
        if children:
            child_dict = flatten_entity_classes(children)
            flat_dict.update(child_dict)

    return flat_dict


ACTIVITY_CLASSES = flatten_entity_classes(
    [x for x in ENTITY_CLASSES if x["annotation_name"] == "<activity>"]
)
STATE_CLASSES = flatten_entity_classes(
    [x for x in ENTITY_CLASSES if x["annotation_name"] == "<state>"]
)
PROCESS_CLASSES = flatten_entity_classes(
    [x for x in ENTITY_CLASSES if x["annotation_name"] == "<process>"]
)
PHYSICAL_OBJECT_CLASSES = flatten_entity_classes(
    [x for x in ENTITY_CLASSES if x["annotation_name"] == "<physical object>"]
)
PROPERTY_CLASSES = flatten_entity_classes(
    [x for x in ENTITY_CLASSES if x["annotation_name"] == "<property>"]
)

# Create a mapping from child class names to their parent class names
child_to_parent_class_name = {a: "<activity>" for a in ACTIVITY_CLASSES}
child_to_parent_class_name.update({s: "<state>" for s in STATE_CLASSES})
child_to_parent_class_name.update({p: "<process>" for p in PROCESS_CLASSES})
child_to_parent_class_name.update(
    {po: "<physical object>" for po in PHYSICAL_OBJECT_CLASSES}
)
child_to_parent_class_name.update({pr: "<property>" for pr in PROPERTY_CLASSES})
