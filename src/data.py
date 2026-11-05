"""Read the Road Damage Dataset the way the paper uses it.

Images are Pascal VOC: one XML per photo, boxes named D00, D10, D20, and D40.
Other labels in the Japanese files (D01, D43, D44, D50) are ignored, which is
what Arya et al. do. Photos are resized to 300x300.
"""

import random
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src.experiments import CLASS_TO_LABEL, CLASSES, COUNTRIES

IMAGE_SIZE = 300
TEST_SIZE = 1000
SEED = 42
DATASET_PAGE = "https://github.com/sekilab/RoadDamageDetector"


def parse_xml(path):
    root = ET.parse(path).getroot()
    size = root.find("size")
    width = int(size.findtext("width") or 0) if size is not None else 0
    height = int(size.findtext("height") or 0) if size is not None else 0
    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in CLASS_TO_LABEL:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin") or 0)
        ymin = float(box.findtext("ymin") or 0)
        xmax = float(box.findtext("xmax") or 0)
        ymax = float(box.findtext("ymax") or 0)
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append((name, xmin, ymin, xmax, ymax))
    return width, height, boxes


def _image_for(image_dir, stem):
    for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _index_country(country_dir):
    xml_dir = country_dir / "annotations" / "xmls"
    image_dir = country_dir / "images"
    if not xml_dir.is_dir() or not image_dir.is_dir():
        return []
    records = []
    for xml_path in sorted(xml_dir.glob("*.xml")):
        image_path = _image_for(image_dir, xml_path.stem)
        if image_path is None:
            continue
        _width, _height, boxes = parse_xml(xml_path)
        records.append(
            {
                "id": xml_path.stem,
                "image": image_path,
                "boxes": boxes,
                "country": country_dir.name,
            }
        )
    return records


def collect_records(root):
    """Find Japan, India, and Czech folders anywhere under root and merge them."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset not found at {root}. Download RDD2020 from {DATASET_PAGE} "
            "and unpack it so Japan, India, and Czech each have images/ and annotations/xmls/."
        )
    merged = {country: {} for country in COUNTRIES}
    for path in root.rglob("*"):
        if not path.is_dir() or path.name not in merged:
            continue
        for record in _index_country(path):
            merged[path.name][record["id"]] = record
    counts = {country: len(items) for country, items in merged.items()}
    if sum(counts.values()) == 0:
        raise FileNotFoundError(
            f"No road images found under {root}. Expected folders named Japan, India, and Czech, "
            f"each with images/ and annotations/xmls/. Dataset: {DATASET_PAGE}"
        )
    return {country: list(items.values()) for country, items in merged.items()}, counts


def hold_out_tests(records_by_country, test_size=TEST_SIZE, seed=SEED):
    """Hold out the same 1,000 test photos per country for every experiment.

    The remaining photos stay in a fixed order, so Ja_2k is the first part of
    Ja_4k, and so on. That matches the paper's check of training-set size.
    """
    pools = {}
    tests = {}
    for country, records in records_by_country.items():
        order = list(records)
        random.Random(seed).shuffle(order)
        if len(order) <= test_size:
            raise ValueError(
                f"{country} has {len(order)} images, which is not enough to hold out {test_size}."
            )
        tests[country] = order[:test_size]
        pools[country] = order[test_size:]
    return pools, tests


def sample_mix(pools, experiment):
    chosen = []
    used = {}
    for country in COUNTRIES:
        count = experiment[country]
        pool = pools[country]
        if count > len(pool):
            raise ValueError(
                f"Need {count} {country} training images, but {len(pool)} remain after the test holdout."
            )
        chosen.extend(pool[:count])
        used[country] = count
    return chosen, used


class RoadDamageDataset(torch.utils.data.Dataset):
    def __init__(self, records, image_size=IMAGE_SIZE):
        self.records = list(records)
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["image"]).convert("RGB")
        width, height = image.size
        image = image.resize((self.image_size, self.image_size))
        scale_x = self.image_size / width
        scale_y = self.image_size / height
        boxes = []
        labels = []
        for name, xmin, ymin, xmax, ymax in record["boxes"]:
            boxes.append([xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y])
            labels.append(CLASS_TO_LABEL[name])
        if boxes:
            box_tensor = torch.tensor(boxes, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)
            label_tensor = torch.zeros((0,), dtype=torch.int64)
        target = {
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": torch.tensor([index]),
        }
        return TF.to_tensor(image), target


def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def write_synthetic(directory, per_country=6):
    """Small stand-in photos so the training loop can be checked without RDD2020."""
    directory = Path(directory)
    rng = random.Random(0)
    for country_index, country in enumerate(COUNTRIES):
        image_dir = directory / country / "images"
        xml_dir = directory / country / "annotations" / "xmls"
        image_dir.mkdir(parents=True, exist_ok=True)
        xml_dir.mkdir(parents=True, exist_ok=True)
        for item in range(per_country):
            stem = f"{country}_{item:02d}"
            color = (
                40 + 30 * country_index,
                80 + 15 * item,
                60,
            )
            Image.new("RGB", (120, 120), color).save(image_dir / f"{stem}.jpg")
            name = CLASSES[item % len(CLASSES)]
            xmin = 10 + rng.randint(0, 20)
            ymin = 10 + rng.randint(0, 20)
            xmax = xmin + 40
            ymax = ymin + 30
            xml_dir.joinpath(f"{stem}.xml").write_text(
                f"""<annotation>
  <filename>{stem}.jpg</filename>
  <size><width>120</width><height>120</height><depth>3</depth></size>
  <object>
    <name>{name}</name>
    <bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin><xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox>
  </object>
</annotation>
""",
                encoding="utf-8",
            )
    return directory
