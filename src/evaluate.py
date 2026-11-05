"""Precision, recall, and F1 at IoU 0.5, the paper's decision rule."""

import torch

from src.experiments import CLASSES, LABEL_TO_CLASS


def box_iou(box, other):
    left = max(box[0], other[0])
    top = max(box[1], other[1])
    right = min(box[2], other[2])
    bottom = min(box[3], other[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection == 0.0:
        return 0.0
    area = (box[2] - box[0]) * (box[3] - box[1])
    other_area = (other[2] - other[0]) * (other[3] - other[1])
    return intersection / (area + other_area - intersection)


def _scores(true_positive, false_positive, false_negative):
    precision = None
    recall = None
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def score_country(predictions, ground_truth, iou_threshold):
    """predictions and ground_truth are lists of {class_name: [(box, score), ...]}."""
    totals = {name: [0, 0, 0] for name in CLASSES}
    for guessed, truth in zip(predictions, ground_truth):
        for name in CLASSES:
            true_boxes = [box for box, _score in truth.get(name, [])]
            candidates = sorted(guessed.get(name, []), key=lambda item: item[1], reverse=True)
            matched = set()
            for box, _score in candidates:
                best_index = None
                best_overlap = 0.0
                for index, truth_box in enumerate(true_boxes):
                    if index in matched:
                        continue
                    overlap = box_iou(box, truth_box)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_index = index
                if best_index is not None and best_overlap >= iou_threshold:
                    totals[name][0] += 1
                    matched.add(best_index)
                else:
                    totals[name][1] += 1
            totals[name][2] += len(true_boxes) - len(matched)
    return {name: _scores(*counts) for name, counts in totals.items()}


def predict(model, loader, device, score_threshold):
    model.eval()
    ordered = []
    with torch.no_grad():
        for images, targets in loader:
            images = [image.to(device) for image in images]
            outputs = model(images)
            for target, output in zip(targets, outputs):
                truth = {name: [] for name in CLASSES}
                for box, label in zip(target["boxes"].tolist(), target["labels"].tolist()):
                    name = LABEL_TO_CLASS.get(int(label))
                    if name is not None:
                        truth[name].append((box, 1.0))
                guessed = {name: [] for name in CLASSES}
                for box, label, score in zip(
                    output["boxes"].tolist(),
                    output["labels"].tolist(),
                    output["scores"].tolist(),
                ):
                    if score < score_threshold:
                        continue
                    name = LABEL_TO_CLASS.get(int(label))
                    if name is not None:
                        guessed[name].append((box, score))
                ordered.append((int(target["image_id"]), guessed, truth))
    ordered.sort(key=lambda item: item[0])
    return [item[1] for item in ordered], [item[2] for item in ordered]
