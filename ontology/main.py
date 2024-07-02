# from owlready2 import *
import owlready2
from owlready2 import (
    types,
    get_ontology,
    OwlReadyError,
    Thing,
    ObjectProperty,
    PropertyChain,
)
import types
from typing import List, Tuple
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import csv
import json
import jsonlines
import itertools
import re
from collections import Counter

from utils import (
    ENTITY_CLASSES,
    RELATION_CLASSES,
    RELATION_PATTERN_MAPPING,
    find_individual_by_label,
    generate_short_id,
    flatten_entity_classes,
    child_to_parent_class_name,
    PROPERTY_CHAIN_AXIOMS,
)

owlready2.JAVA_EXE = r"C:\Program Files\Java\jre-1.8\bin\java.exe"

ENTITY_CLASS_MAPPING = flatten_entity_classes(ENTITY_CLASSES)

CREATE_FUNCTION_STATES = True
CONVERT_UNDESIRABLE_STATE_TO_INDETERMINATE_STATE = True


def load_triplets(file_path: str) -> List[tuple]:
    df = pd.read_csv(file_path, sep="|", dtype="str", quoting=csv.QUOTE_NONE)
    return [tuple(list(d.values())) for d in df.to_dict(orient="records")]


def to_camel_case(text):
    try:
        words = text.split()
        if len(words) == 0:
            return ""

        # Capitalize the first word and join it with the rest of the words
        camel_case = words[0].capitalize() + "".join(
            word.capitalize() for word in words[1:]
        )

        return camel_case
    except:
        import IPython

        IPython.embed()
        exit(1)


# Helper function to create classes recursively
def create_classes(onto, class_data, parent=None):
    class_name = class_data["onto_name"]
    if parent:
        class_name = class_name  # f"{parent.__name__}.{class_name}"

    # Use a with block to create classes within the ontology context
    with onto:
        if parent:
            new_class = types.new_class(class_name, (parent,))
        else:
            new_class = types.new_class(class_name, (Thing,))

    # Recursively create child classes
    for child_data in class_data.get("children", []):
        create_classes(onto, child_data, new_class)


def is_rdf_class_name_compliant(class_name):
    """
    Check if the given string is compliant with RDF class naming conventions.

    RDF class names typically start with an uppercase letter and are camel-cased.
    However, they can also contain numbers and underscores.

    Args:
    class_name (str): The class name to check.

    Returns:
    bool: True if the class name is compliant, False otherwise.
    """
    if not class_name:
        return False

    # Check if the first character is uppercase and the rest of the string follows the camel-case pattern
    return class_name[0].isupper() and all(x.isalnum() or x == "_" for x in class_name)


def create_ontology_base_classes(onto):
    # Iterate through ENTITY_CLASSES and create classes
    for class_data in ENTITY_CLASSES:
        create_classes(onto, class_data)
    # AllDisjoint([onto[entity_name] for entity_name in ENTITY_CLASSES.values()])


def create_ontology_object_properties(onto):
    # Create properties
    with onto:
        for relation, details in RELATION_CLASSES.items():
            print(f"Creating object property: {relation}")
            try:
                parent_class = (
                    ObjectProperty
                    if details.get("parent") is None
                    else onto[details["parent"]]
                )
                NewProp = types.new_class(relation, (parent_class,))
                if "domain" in details:
                    NewProp.domain = [
                        onto[domain_class] for domain_class in details.get("domain", [])
                    ]
                if "range" in details:
                    NewProp.range = [
                        onto[range_class] for range_class in details.get("range", [])
                    ]
                if "inverse_of" in details:
                    NewProp.inverse_property = onto[details["inverse_of"]]
            except:
                print("failed to create object property")
                import IPython

                IPython.embed()
                exit(1)

    # AllDisjoint([onto[relation] for relation in RELATION_CLASSES.keys()])


# Function to populate the ontology with individuals
def populate_onto(onto, triplets):
    with onto:
        for head, head_type, rel_type, tail, tail_type in tqdm(triplets):
            head_class = onto[to_camel_case(head)]  # ENTITY_CLASS_MAPPING[head_type]]
            tail_class = onto[to_camel_case(tail)]  # ENTITY_CLASS_MAPPING[tail_type]]

            # Check if the individuals already exist by label
            head_individual = find_individual_by_label(onto, head, head_class)
            if not head_individual:
                # Generate a unique ID for the head individual if it doesn't exist
                head_individual = head_class(generate_short_id())
                head_individual.label = head

            tail_individual = find_individual_by_label(onto, tail, tail_class)
            if not tail_individual:
                # Generate a unique ID for the tail individual if it doesn't exist
                tail_individual = tail_class(generate_short_id())
                tail_individual.label = tail

            # Normalise relation type to match ObjectProperty naming convention
            rel_type_normalised = rel_type.replace(" ", "_")

            rel_type = RELATION_PATTERN_MAPPING.get(
                (head_type, rel_type_normalised, tail_type), rel_type_normalised
            )

            # Ensure the relationship property exists in the ontology
            if onto[rel_type]:
                # Use the property to assign the tail individual to the head
                relation = getattr(head_individual, rel_type)
                if tail_individual not in relation:
                    relation.append(tail_individual)
            else:
                # Handle the case where the property is not defined
                print(f"The property {rel_type} is not defined in the ontology.")


def remove_circular_references(triples):
    parent_to_children = defaultdict(dict)
    child_to_parents = defaultdict(dict)

    # Building mappings with frequency data
    for head, _, relation, tail, _, frequency in triples:
        if relation == "is a":
            parent_to_children[tail][head] = frequency
            child_to_parents[head][tail] = frequency

    # Identifying and resolving circular references based on frequency
    cleaned_triples = []
    for triple in triples:
        head, _, relation, tail, _, frequency = triple
        if relation != "is a":
            cleaned_triples.append(triple)
            continue

        if tail in child_to_parents and head in child_to_parents[tail]:
            # Circular reference found
            reverse_frequency = child_to_parents[tail][head]
            if frequency <= reverse_frequency:
                print(f"Skipping due to lower frequency circular reference: {triple}")
                continue  # Skip this triple if its frequency is lower or equal

        cleaned_triples.append(triple)

    return cleaned_triples


def build_hierarchy(triples, lowercase: bool = True):
    temp_hierarchy = defaultdict(set)
    all_children = set()

    for head, head_type, relation, tail, tail_type, _ in triples:
        if relation == "is a":
            parent = (tail.lower() if lowercase else tail, tail_type)
            child = (head.lower() if lowercase else head, head_type)
            # Add child only if it's not the same as parent to avoid direct self-reference
            if child != parent:
                temp_hierarchy[parent].add(child)
                all_children.add(child)

    def build_nested_dict(node, ancestors):
        # Check if the node is in ancestors to prevent circular reference
        if node in ancestors:
            return {}

        # Update ancestors for the next level of recursion
        updated_ancestors = ancestors | {node}

        # Recursive call with updated ancestors
        return {
            child: build_nested_dict(child, updated_ancestors)
            for child in temp_hierarchy.get(node, [])
        }

    top_level_nodes = set(temp_hierarchy.keys()) - all_children
    hierarchy = {node: build_nested_dict(node, set()) for node in top_level_nodes}

    return hierarchy


def stringify_keys(d):
    """Recursively convert tuple keys to strings in a dictionary."""
    if not isinstance(d, dict):
        return d
    return {str(key): stringify_keys(value) for key, value in d.items()}


def print_hierarchy_as_json(hierarchy):
    stringified_hierarchy = stringify_keys(hierarchy)
    print(json.dumps(stringified_hierarchy, indent=4))


def print_hierarchy_prettified(hierarchy, node, indent=0):
    """
    Recursively prints the hierarchy in a prettified manner.

    Args:
    hierarchy (dict): The hierarchy to print.
    node (tuple): The current node being printed.
    indent (int): The current level of indentation.
    """
    # Print the current node with appropriate indentation
    print(" " * indent + str(node))

    # If the node has children, recursively print each one
    if node in hierarchy:
        for child in hierarchy[node]:
            print_hierarchy_prettified(hierarchy[node], child, indent + 4)


def create_class(parent_class, subclass_name, onto):
    # Create a new in the ontology
    subclass = types.new_class(subclass_name, (parent_class,))


def traverse_hierarchy_create_classes(onto, hierarchy, level=0, parent=None):
    for node, children in hierarchy.items():
        try:
            # Check if the current node is at the top level
            is_top_level = level == 0

            # Printing the node information, its parent, and whether it's at the top level
            # print(f"Node: {node}, Parent: {parent}, Top Level: {is_top_level}")

            entity_class_name = to_camel_case(node[0])
            if is_rdf_class_name_compliant(class_name=entity_class_name):
                if is_top_level:
                    try:
                        parent_class = types.new_class(
                            entity_class_name,
                            (onto[ENTITY_CLASS_MAPPING[node[1]]],),
                        )
                        print(f"Created class: {entity_class_name}")
                    except:
                        print(f"Failed to create parent class {entity_class_name}")
                else:
                    try:
                        parent_class = types.new_class(entity_class_name, (parent,))
                    except:
                        print(
                            f"Failed to create child class {entity_class_name} for {parent}"
                        )

                # Recursively traverse the children, incrementing the level and passing the current node as the parent
                traverse_hierarchy_create_classes(
                    onto, children, level + 1, parent_class
                )
        except:
            pass


def convert_undesirable_to_indeterminate_state(entity: tuple):
    if "undesirable" in entity[1]:
        return (entity[0], "<indeterminate state>")
    return entity


def sanitize_state(entity: tuple):
    if "state" in entity[1]:
        if CONVERT_UNDESIRABLE_STATE_TO_INDETERMINATE_STATE:
            return convert_undesirable_to_indeterminate_state(entity)
    return entity


def populate_onto_from_triple_docs(onto, docs, create_functional_states: bool = False):
    sanitize_state = lambda x: x.replace(
        "undesirable",
        f'{"indeterminate" if CONVERT_UNDESIRABLE_STATE_TO_INDETERMINATE_STATE else "undesirable"}',
    )

    with onto:
        for doc_idx, doc in tqdm(enumerate(docs)):
            assert "triples" in doc.keys(), 'Doc missing "triples" key'
            assert "input" in doc.keys(), 'Doc missing "input" key'

            triplets = [tuple(t.values()) for t in doc["triples"]]

            _classes = set(
                itertools.chain.from_iterable(
                    [
                        [(t[0], sanitize_state(t[1])), (t[3], t[4])]
                        for t in triplets
                        if t[2] != "is a"
                    ]
                )
            )

            # Associate states with physical objects to get "functional" states for futher state segregation
            state_to_object_map = [
                (
                    t[0],
                    sanitize_state(t[1]),
                    t[3],
                    t[4],
                )
                for t in triplets
                if "state" in t[1]
            ]  # assumes subject is always state

            # Create/find classes and return
            _classes_mapped = {}
            for c in _classes:
                class_name = c[0]
                class_name = c[0].replace(
                    "-", ""
                )  # Replace any hyphens as these are not RDF compliant
                entity_class_name = to_camel_case(class_name)

                if is_rdf_class_name_compliant(entity_class_name):
                    if onto[entity_class_name] == None:
                        print(f"{entity_class_name} not in ontology class hierarchy")

                        if create_functional_states:
                            for obj in state_to_object_map:
                                if obj[0] == c[0] and obj[1] == c[1]:
                                    # Create new functional state class name
                                    _functional_state_class_name = to_camel_case(
                                        obj[3]
                                        .replace("<", "")
                                        .replace(">", "")
                                        .replace("object", "")
                                        .capitalize()
                                    )
                                    _functional_state_class_name += f"{to_camel_case(c[1].replace('<','').replace('>','').capitalize())}"
                                    print(
                                        f"_functional_state_class_name: {_functional_state_class_name}"
                                    )

                                    _parent_class = types.new_class(
                                        _functional_state_class_name,
                                        (onto[ENTITY_CLASS_MAPPING[c[1]]],),
                                    )

                                    _class = types.new_class(
                                        entity_class_name, (_parent_class,)
                                    )

                                    individual = _class(generate_short_id())
                                    individual.label = (
                                        f"{doc_idx}_{class_name.replace(' ','_')}"
                                    )
                                    individual.comment = doc["input"]

                                    _classes_mapped[c] = individual

                        else:
                            # Create class based on its annotation entity class
                            _class = types.new_class(
                                entity_class_name, (onto[ENTITY_CLASS_MAPPING[c[1]]],)
                            )
                            print(f"Created new class for {entity_class_name}")

                            individual = _class(generate_short_id())
                            individual.label = (
                                f"{doc_idx}_{class_name.replace(' ','_')}"
                            )
                            individual.comment = doc["input"]

                            _classes_mapped[c] = individual

                    else:
                        _class = onto[entity_class_name]
                        individual = _class(generate_short_id())
                        individual.label = f"{doc_idx}_{class_name.replace(' ','_')}"
                        individual.comment = doc["input"]

                        _classes_mapped[c] = individual

            for head, head_type, rel_type, tail, tail_type in triplets:
                if (
                    _classes_mapped.get((head, head_type)) == None
                    or _classes_mapped.get((tail, tail_type)) == None
                ):
                    # Haven't been put into the ontology as classes yet.
                    continue

                # Normalise relation type to match ObjectProperty naming convention
                rel_type_normalised = rel_type.replace(" ", "_")

                rel_type = rel_type_normalised
                # rel_type = RELATION_PATTERN_MAPPING.get(
                #     (
                #         child_to_parent_class_name.get(head_type),
                #         rel_type_normalised,
                #         child_to_parent_class_name.get(tail_type),
                #     ),
                #     rel_type_normalised,
                # )

                # Ensure the relationship property exists in the ontology
                if onto[rel_type]:
                    # Check if the individuals already exist by label
                    head_individual = _classes_mapped.get((head, head_type))
                    tail_individual = _classes_mapped.get((tail, tail_type))

                    # Use the property to assign the tail individual to the head
                    relation = getattr(head_individual, rel_type)
                    if tail_individual not in relation:
                        relation.append(tail_individual)
                        print("Created new relation")
                else:
                    # Handle the case where the property is not defined
                    print(f"The property {rel_type} is not defined in the ontology.")


def create_populate_onto(name, onto, triplets, docs):
    try:
        onto.load()
        onto.destroy()
        onto = get_ontology(f"file://{name}")
    except OwlReadyError as e:
        print(f"Error loading ontology: {e}")
        # Handle error or create a new ontology

    print(f"Loaded ontology: {onto}")

    cleaned_triplets = remove_circular_references(triplets)

    ontology_hierarchy = build_hierarchy(cleaned_triplets)

    # Create MaintIE annotation schema classes
    create_ontology_base_classes(onto)

    # Create MaintIE annotation schema classes
    create_ontology_object_properties(onto)

    with onto:
        traverse_hierarchy_create_classes(onto, ontology_hierarchy)

    populate_onto_from_triple_docs(
        onto, docs=docs, create_functional_states=CREATE_FUNCTION_STATES
    )

    # with onto:
    #     # Create property chain axiom(s)
    #     for chain_name, properties in PROPERTY_CHAIN_AXIOMS.items():
    #         chain_properties = PropertyChain(
    #             [onto[prop_name] for prop_name in properties]
    #         )
    #         # new property
    #         _new_object_prop = types.new_class(chain_name, (ObjectProperty,))
    #         _new_object_prop.property_chain.append(chain_properties)

    onto.save(file=name)


def create_maintkb_ontology():
    """Creates ontology classes and populates from large automatically extracted maintenance corpus"""

    TRIPLET_FILE_PATH = f"../data/outputs/maintkb.psv"
    ONTOLOGY_NAME = f"maintkb.owl"
    onto = get_ontology(f"file://{ONTOLOGY_NAME}")

    triplets = load_triplets(TRIPLET_FILE_PATH)

    # POPULATION PROCESS
    MWO_PRED_FILE = r"D:\Repos\maintnet\data\preds.jsonl"
    with jsonlines.open(MWO_PRED_FILE) as reader:
        mwo_preds = list(reader)

    mwo_preds = [{"input": d["input"], "triples": d["preds"]} for d in mwo_preds]

    mwo_preds = mwo_preds[:100000]

    create_populate_onto(
        name=ONTOLOGY_NAME,
        onto=onto,
        triplets=triplets,
        docs=mwo_preds,
    )


def create_test_ontology():
    """Creates ontology classes and populates with limited set of individuals for testing"""

    print(f"TEST_MODE")
    TRIPLET_FILE_PATH = f"../data/outputs/maintkb_test.psv"
    ONTOLOGY_NAME = f"maintkb_test.owl"
    onto = get_ontology(f"file://{ONTOLOGY_NAME}")

    TEST_TRIPLETS = [
        (
            "requires",
            "<undesirable state>",
            "has patient",
            "replace",
            "<replace>",
            1,
        ),
        # (
        #     "oil centrifugal pump",
        #     "<physical object>",
        #     "is a",
        #     "centrifugal pump",
        #     "<physical object>",
        #     1,
        # ),
        # ("piston pump", "<physical object>", "is a", "pump", "<physical object>", 1),
        # (
        #     "centrifugal pump",
        #     "<physical object>",
        #     "is a",
        #     "pump",
        #     "<physical object>",
        #     1,
        # ),
        # (
        #     "lube centrifugal pump",
        #     "<physical object>",
        #     "is a",
        #     "centrifugal pump",
        #     "<physical object>",
        #     1,
        # ),
        # ("oil cap", "<physical object>", "is a", "cap", "<physical object>", 1),
        # ("replace", "<replace>", "has patient", "cap", "<physical object>", 1),
        # ("replace", "<replace>", "has patient", "pump", "<physical object>", 1),
        # ("adjust", "<adjust>", "has patient", "pump", "<physical object>", 1),
        # ("adjustment", "<adjust>", "has patient", "pump", "<physical object>", 1),
        # ("replace", "<replace>", "has agent", "technician", "<physical object>", 1),
        # ("replace", "<replace>", "has patient", "bolt", "<physical object>", 1),
    ]

    TEST_TRIPLETS_V2 = [
        (
            "centrifugal pump",
            "<physical object>",
            "is a",
            "pump",
            "<physical object>",
            1,
        ),
        (
            "oil centrifugal pump",
            "<physical object>",
            "is a",
            "centrifugal pump",
            "<physical object>",
            5,
        ),
        (
            "centrifugal pump",
            "<physical object>",
            "is a",
            "oil centrifugal pump",
            "<physical object>",
            1,
        ),
    ]

    TEST_DOCS = [
        {
            "input": "pump requires replacing",
            "triples": [
                {
                    "head": "requires",
                    "head_type": "<undesirable state>",
                    "relation": "has patient",
                    "tail": "replacing",
                    "tail_type": "<replace>",
                },
                {
                    "head": "requires",
                    "head_type": "<undesirable state>",
                    "relation": "has agent",
                    "tail": "pump",
                    "tail_type": "<physical object>",
                },
            ],
        },
        # {
        #     "input": "replace cap",
        #     "triples": [
        #         {
        #             "head": "replace",
        #             "head_type": "<replace>",
        #             "relation": "has patient",
        #             "tail": "cap",
        #             "tail_type": "<physical object>",
        #         }
        #     ],
        # },
        # {
        #     "input": "pump replacement",
        #     "triples": [
        #         {
        #             "head": "replacement",
        #             "head_type": "<replace>",
        #             "relation": "has patient",
        #             "tail": "pump",
        #             "tail_type": "<physical object>",
        #         },
        #     ],
        # },
        # {
        #     "input": "technician adjust pump",
        #     "triples": [
        #         {
        #             "head": "adjust",
        #             "head_type": "<adjust>",
        #             "relation": "has patient",
        #             "tail": "pump",
        #             "tail_type": "<physical object>",
        #         },
        #         {
        #             "head": "adjust",
        #             "head_type": "<adjust>",
        #             "relation": "has agent",
        #             "tail": "technician",
        #             "tail_type": "<physical object>",
        #         },
        #     ],
        # },
        # {
        #     "input": "centrifugal pump adjustment",
        #     "triples": [
        #         {
        #             "head": "adjustment",
        #             "head_type": "<adjust>",
        #             "relation": "has patient",
        #             "tail": "centrifugal pump",
        #             "tail_type": "<physical object>",
        #         },
        #         {
        #             "head": "centrifugal pump",
        #             "head_type": "<physical object>",
        #             "relation": "is a",
        #             "tail": "pump",
        #             "tail_type": "<physical object>",
        #         },
        #     ],
        # },
        # {
        #     "input": "replace pump bolt",
        #     "triples": [
        #         {
        #             "head": "replace",
        #             "head_type": "<replace>",
        #             "relation": "has patient",
        #             "tail": "bolt",
        #             "tail_type": "<physical object>",
        #         },
        #         {
        #             "head": "pump",
        #             "head_type": "<physical object>",
        #             "relation": "has part",
        #             "tail": "bolt",
        #             "tail_type": "<physical object>",
        #         },
        #     ],
        # },
        # {
        #     "input": "centrifugal pump blown",
        #     "triples": [
        #         {
        #             "head": "blown",
        #             "head_type": "<failed state>",
        #             "relation": "has patient",
        #             "tail": "centrifugal pump",
        #             "tail_type": "<physical object>",
        #         },
        #         {
        #             "head": "centrifugal pump",
        #             "head_type": "<physical object>",
        #             "relation": "is a",
        #             "tail": "pump",
        #             "tail_type": "<physical object>",
        #         },
        #     ],
        # },
        # {
        #     "input": "pump bolt sheared",
        #     "triples": [
        #         {
        #             "head": "sheared",
        #             "head_type": "<failed state>",
        #             "relation": "has patient",
        #             "tail": "bolt",
        #             "tail_type": "<physical object>",
        #         },
        #         {
        #             "head": "pump",
        #             "head_type": "<physical object>",
        #             "relation": "has part",
        #             "tail": "bolt",
        #             "tail_type": "<physical object>",
        #         },
        #     ],
        # },
        # {
        #     "input": "technican to repair bolt",
        #     "triples": [
        #         {
        #             "head": "repair",
        #             "head_type": "<repair>",
        #             "relation": "has patient",
        #             "tail": "bolt",
        #             "tail_type": "<physical object>",
        #         },
        #         {
        #             "head": "repair",
        #             "head_type": "<repair>",
        #             "relation": "has agent",
        #             "tail": "technician",
        #             "tail_type": "<physical object>",
        #         },
        #     ],
        # },
    ]

    create_populate_onto(
        name=ONTOLOGY_NAME, onto=onto, triplets=TEST_TRIPLETS, docs=TEST_DOCS
    )


def create_gold_annotated_ontology(limit: int = None):
    """Creates ontology classes and populates from gold MaintIE annotations. This provides a deeper hierarchical classification than automatic processes (ATM)

    Params
    ------
    limit : number of documents to use to create the onto.
    """

    ONTOLOGY_NAME = f"maintkb_gold.owl"
    onto = get_ontology(f"file://{ONTOLOGY_NAME}")

    # Convert .json entity/relation format to triple format
    with open(r"D:\Repos\maintnet\data\maintie\gold_release_v0.json", "r") as f:
        mwo_gold = json.load(f)

    def convert_format(input_string, include_brackets: bool = True):
        # Split the string by '/' and take the last part
        last_part = input_string.split("/")[-1]

        # Insert spaces before capital letters and convert to lowercase
        formatted_string = re.sub(r"(?<!^)(?=[A-Z])", " ", last_part).lower()

        return f"<{formatted_string}>" if include_brackets else f"{formatted_string}"

    mwo_gold = [
        {
            "input": d["text"],
            "triples": [
                {
                    "head": " ".join(
                        d["tokens"][
                            d["entities"][r["head"]]["start"] : d["entities"][
                                r["head"]
                            ]["end"]
                        ]
                    ),
                    "head_type": convert_format(d["entities"][r["head"]]["type"]),
                    "rel_type": convert_format(r["type"], include_brackets=False),
                    "tail": " ".join(
                        d["tokens"][
                            d["entities"][r["tail"]]["start"] : d["entities"][
                                r["tail"]
                            ]["end"]
                        ]
                    ),
                    "tail_type": convert_format(
                        d["entities"][r["tail"]]["type"].split("/")[-1]
                    ),
                }
                for r in d["relations"]
            ],
        }
        for d in mwo_gold
    ]

    if limit:
        mwo_gold = mwo_gold[:limit]
        print(f"Limited MWO docs to {limit}")

    # Convert mwo_gold into triples to create ontology classes
    mwo_gold_triples = list(
        itertools.chain.from_iterable([d["triples"] for d in mwo_gold])
    )
    mwo_gold_triplet_counts = Counter([tuple(t.values()) for t in mwo_gold_triples])
    mwo_gold_triplets = [
        (
            *key,
            value,
        )
        for key, value in mwo_gold_triplet_counts.items()
    ]

    create_populate_onto(
        name=ONTOLOGY_NAME,
        onto=onto,
        triplets=mwo_gold_triplets,
        docs=mwo_gold,
    )


if __name__ == "__main__":
    # create_test_ontology()
    create_gold_annotated_ontology()    # limit=50
    # create_maintkb_ontology()

    #   sync_reasoner(infer_property_values = True)
