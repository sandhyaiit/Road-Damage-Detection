"""The 16 training mixes from Arya et al., arXiv:2008.13101, Table 2.

Each mix is named the way the paper names it. Counts are images, not damage boxes.
The test list is the countries that mix is scored on. Every test country uses
1,000 held-out images.
"""

EXPERIMENTS = {
    "Ja_2k": {"Japan": 2000, "India": 0, "Czech": 0, "test": ["Japan", "India", "Czech"]},
    "Ja_4k": {"Japan": 4000, "India": 0, "Czech": 0, "test": ["Japan", "India", "Czech"]},
    "Ja_6k": {"Japan": 6000, "India": 0, "Czech": 0, "test": ["Japan", "India", "Czech"]},
    "Ja_8k": {"Japan": 8000, "India": 0, "Czech": 0, "test": ["Japan", "India", "Czech"]},
    "Ja_10k": {"Japan": 10000, "India": 0, "Czech": 0, "test": ["Japan", "India", "Czech"]},
    "In_2k": {"Japan": 0, "India": 2000, "Czech": 0, "test": ["India"]},
    "In_4k": {"Japan": 0, "India": 4000, "Czech": 0, "test": ["India"]},
    "In_6k": {"Japan": 0, "India": 6000, "Czech": 0, "test": ["India"]},
    "In_8k": {"Japan": 0, "India": 8000, "Czech": 0, "test": ["India"]},
    "Cz_2k": {"Japan": 0, "India": 0, "Czech": 2000, "test": ["Czech"]},
    "InJa_12k": {"Japan": 10000, "India": 2000, "Czech": 0, "test": ["India", "Japan"]},
    "InJa_14k": {"Japan": 10000, "India": 4000, "Czech": 0, "test": ["India", "Japan"]},
    "InJa_16k": {"Japan": 10000, "India": 6000, "Czech": 0, "test": ["India", "Japan"]},
    "InJa_18k": {"Japan": 10000, "India": 8000, "Czech": 0, "test": ["India", "Japan"]},
    "JaCz_11k": {"Japan": 10000, "India": 0, "Czech": 1000, "test": ["Czech"]},
    "JaCz_12k": {"Japan": 10000, "India": 0, "Czech": 2000, "test": ["Czech"]},
}

COUNTRIES = ("Japan", "India", "Czech")
CLASSES = ("D00", "D10", "D20", "D40")
CLASS_TO_LABEL = {name: index + 1 for index, name in enumerate(CLASSES)}
LABEL_TO_CLASS = {label: name for name, label in CLASS_TO_LABEL.items()}
