"""Train one of the paper's country mixes and write precision, recall, and F1."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import (
    IMAGE_SIZE,
    RoadDamageDataset,
    collate,
    collect_records,
    hold_out_tests,
    sample_mix,
    write_synthetic,
)
from src.evaluate import predict, score_country
from src.experiments import EXPERIMENTS
from src.models import build_ssd_mobilenet


def parse_args():
    parser = argparse.ArgumentParser(
        description="SSD MobileNet road damage detection, following Arya et al. arXiv:2008.13101."
    )
    parser.add_argument("--data", default="data", help="Folder that contains Japan, India, and Czech.")
    parser.add_argument("--experiment", default="InJa_14k", choices=sorted(EXPERIMENTS))
    parser.add_argument("--list", action="store_true", help="Print the 16 mixes and exit.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate used in the paper.")
    parser.add_argument("--iou", type=float, default=0.5, help="Box overlap required for a match.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Minimum detector score. The paper does not publish this cutoff.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip ImageNet MobileNet weights. Useful only for a short code check.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Train on a handful of drawn images. The report from this run is not RDD2020.",
    )
    return parser.parse_args()


def print_experiments():
    print(f"{'name':<12} {'Japan':>7} {'India':>7} {'Czech':>7}  test")
    for name in EXPERIMENTS:
        item = EXPERIMENTS[name]
        tests = ", ".join(item["test"])
        print(f"{name:<12} {item['Japan']:>7} {item['India']:>7} {item['Czech']:>7}  {tests}")


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    steps = 0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        loss = sum(model(images, targets).values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        steps += 1
    return total / max(steps, 1)


def main():
    args = parse_args()
    if args.list:
        print_experiments()
        return

    if args.synthetic:
        data_root = write_synthetic(Path(args.output_dir or "outputs/smoke") / "images")
        records, counts = collect_records(data_root)
        train_records = []
        test_records = {}
        for country, items in records.items():
            train_records.extend(items[:-2])
            test_records[country] = items[-2:]
        experiment_name = "synthetic"
        train_counts = counts
        test_countries = list(test_records)
        source = "synthetic"
    else:
        records, counts = collect_records(args.data)
        pools, tests = hold_out_tests(records, seed=args.seed)
        experiment = EXPERIMENTS[args.experiment]
        train_records, train_counts = sample_mix(pools, experiment)
        test_records = {country: tests[country] for country in experiment["test"]}
        experiment_name = args.experiment
        test_countries = experiment["test"]
        source = "rdd2020"

    random.Random(args.seed).shuffle(train_records)
    device = torch.device("cpu")
    model = build_ssd_mobilenet(pretrained_backbone=not args.no_pretrained).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    loader = DataLoader(
        RoadDamageDataset(train_records, IMAGE_SIZE),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    print(
        f"source={source} experiment={experiment_name} train={len(train_records)} "
        f"countries={train_counts}"
    )
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f"epoch={epoch} loss={loss:.4f}")

    results = {}
    for country in test_countries:
        test_loader = DataLoader(
            RoadDamageDataset(test_records[country], IMAGE_SIZE),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        guessed, truth = predict(model, test_loader, device, args.score_threshold)
        results[country] = score_country(guessed, truth, args.iou)
        f1_text = ", ".join(
            f"{name}={scores['f1']}" for name, scores in results[country].items()
        )
        print(f"test={country} f1: {f1_text}")

    output_dir = Path(args.output_dir or Path("outputs") / experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "experiment": experiment_name}, output_dir / "model.pt")
    report = {
        "paper": "Arya et al., Transfer Learning-based Road Damage Detection for Multiple Countries, arXiv:2008.13101",
        "data_source": source,
        "experiment": experiment_name,
        "detector": "SSD MobileNet",
        "image_size": IMAGE_SIZE,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "iou_threshold": args.iou,
        "score_threshold": args.score_threshold,
        "train_images": len(train_records),
        "train_counts": train_counts,
        "results": results,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(1) from None
