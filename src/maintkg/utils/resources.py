corrections_dict = {
    "preasure": "pressure",
    "wotking": "working",
    "2-WAY": "two-way",
    "2WAY": "two-way",
    "2-way": "two-way",
    "2way": "two-way",
    "2 way": "two-way",
    "2 WAY": "two-way",
    "a/c": "air conditioner",
    "a/f": "after cooler",
    "als": "ALS",
    "anti cav": "anti-cavitation",
    "anti-cav": "anti-cavitation",
    "assy": "assembly",
    "auto retard": "auto-retarder",
    "bcs": "BCS",
    "boken": "broken",
    "brke": "brake",
    "brohen": "broken",
    "broked": "broken",
    "c/body": "car body",
    "c/o": "change out",
    "cab": "cabin",
    "cav": "cavitation",
    "cha": "change",
    "chan": "change",
    "chng": "change",
    "cons": "conditioners",
    "contam": "contamination",
    "cyl": "cylinder",
    "cylnder": "cylinder",
    "dcv": "DCV",
    "g . e . t": "GET",
    "g.e.t": "GET",
    "G.E.T": "GET",
    "glr": "GLR",
    "h.p": "high pressure",
    "handpump": "hand pump",
    "hydrolics": "hydraulics",
    "hid": "HID",
    "lhbank": "left hand bank",
    "mech": "mechanical",
    "meteal": "metal",
    "mp": "MP",
    "mstr": "master",
    "o/h": "overhaul",
    "o/stroke": "overstroke",
    "overtemp": "over temperature",
    "ovhl": "overhaul",
    "prssr": "pressure",
    "repl": "replace",
    "rev control": "reverse control",
    "rollar": "roller",
    "rx": "receiving",
    "releif": "relief",
    "schrod": "shroud",
    "schroud": "shroud",
    "schroud": "shroud",
    "senor": "sensor",
    "sht": "shut",
    "shut off": "shut-off",
    "sps": "SPS",
    "stc": "STC",
    "t/stat": "thermostat",
    "temp": "temperature",
    "temps": "temperatures",
    "trvl": "travel",
    "tx": "transmission",
    "u/s": "unserviceable",
    "up/both": "up / both",
    "vhf": "VHF",
    "was n't": "wasn't",
    "wasn t": "wasn't",
    "wind shroud": "wing shroud",
    "cct": "CCT",
    "O&K RH 170": "<id>",
    "O&K": "",
    "L.H.S": "left hand side",
    "l.h.s": "left hand side",
    "ove": "over",
    "intermitently": "intermittently",
    "blwn": "blown",
    "xover": "cross-over",
    "wireing": "wiring",
    "waterpump": "water pump",
    "cambuss": "cambus",
    "windscreane": "windscreen",
    "wieght": "weight",
}

entity_tag_to_pos = {
    "activity": "v",
    "process": "v",
    "object": "n",
    "property": "n",
    "state": "v",
}
colloquialisms = {
    "conditioning": "conditioner",
    "piping": "pipe",
    "leaky": "leak",
}
negations = {
    "wont": "will not",
    "won't": "will not",
    "wouldnt": "would not",
    "wouldn't": "would not",
    "dont": "does not",
    "don't": "does not",
    "doesnt": "does not",
    "doesn't": "does not",
}

MAPPING = {"hasPart": "has part", "isA": "is a", "hasProperty": "has property"}
ENTITY_TYPES = {"object", "activity", "process", "property", "state"}

RELATION_CONSTRAINTS = {
    "has_patient": {
        ("activity", "activity"),
        ("activity", "state"),
        ("activity", "process"),
        ("activity", "object"),
        ("process", "object"),
        ("state", "activity"),
        ("state", "object"),
    },
    "has_agent": {("state", "object"), ("process", "object"), ("activity", "object")},
    "contains": {("object", "object")},
    "has_part": {("object", "object")},
    "has_property": {("object", "property"), ("state", "property")},
    "is_a": {(e, e) for e in ENTITY_TYPES},
}

RELATION_TYPES = {
    "has_part",
    "is_a",
    "has_property",
    "has_patient",
    "contains",
    "has_agent",
}
