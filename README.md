# Road damage detection across countries

This repo follows Arya, Maeda, Ghosh, Toshniwal, Mraz, Kashiyama, and Sekimoto, [Transfer Learning-based Road Damage Detection for Multiple Countries](https://arxiv.org/abs/2008.13101).

The dataset has 26,620 smartphone photos from Japan, India, and the Czech Republic. Each damage box is one of four types:

| Label | What it is |
| --- | --- |
| D00 | Longitudinal crack |
| D10 | Transverse crack |
| D20 | Alligator crack |
| D40 | Pothole |

The detector is SSD with a MobileNet backbone. Training starts from a MobileNet that has already been trained on ImageNet, then the detector is trained on road photos. Images are resized to 300×300. The learning rate is 0.003, the value given in the paper. A predicted box is counted as correct when its overlap with a labeled box (IoU) is at least 0.5. Precision, recall, and F1 are computed for each damage type.

The paper's main result is that a model trained only on Japanese roads does poorly on Indian and Czech roads. Adding local photos to the Japanese photos is the setup they recommend. The 16 mixes below are the ones in their Table 2. `InJa_14k` (10,000 Japanese images and 4,000 Indian images) is the mix they single out for India and Japan together.

The paper trained with the TensorFlow Object Detection API. This repo trains the same mixes in PyTorch with torchvision's MobileNet-SSD. The paper does not say how many epochs it trained, or which score cutoff it used for F1. Epochs default to 10. The score cutoff defaults to 0.5.

## Dataset

Download RDD2020 from [sekilab/RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector). Unpack it under `data/` so the country folders look like this:

```text
data/
  train/
    Japan/
      images/
      annotations/xmls/
    India/
    Czech/
  test1/
    Japan/
    India/
    Czech/
```

`train` and `test1` are both read. Photos from the same country are pooled, then 1,000 images per country are held out as the test set. Smaller mixes are the first part of the larger mix for that country, so `Ja_2k` is contained in `Ja_10k`. Labels other than D00, D10, D20, and D40 are skipped.

## The 16 mixes

| Name | Japan | India | Czech | Scored on |
| --- | ---: | ---: | ---: | --- |
| Ja_2k | 2000 | 0 | 0 | Japan, India, Czech |
| Ja_4k | 4000 | 0 | 0 | Japan, India, Czech |
| Ja_6k | 6000 | 0 | 0 | Japan, India, Czech |
| Ja_8k | 8000 | 0 | 0 | Japan, India, Czech |
| Ja_10k | 10000 | 0 | 0 | Japan, India, Czech |
| In_2k | 0 | 2000 | 0 | India |
| In_4k | 0 | 4000 | 0 | India |
| In_6k | 0 | 6000 | 0 | India |
| In_8k | 0 | 8000 | 0 | India |
| Cz_2k | 0 | 0 | 2000 | Czech |
| InJa_12k | 10000 | 2000 | 0 | India, Japan |
| InJa_14k | 10000 | 4000 | 0 | India, Japan |
| InJa_16k | 10000 | 6000 | 0 | India, Japan |
| InJa_18k | 10000 | 8000 | 0 | India, Japan |
| JaCz_11k | 10000 | 0 | 1000 | Czech |
| JaCz_12k | 10000 | 0 | 2000 | Czech |

## Setup

PyTorch and torchvision are required. A GPU is the practical way to train the full mixes. The code runs on CPU.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision
pip install -r requirements.txt
```

List the 16 mixes:

```powershell
python -m src.train --list
```

## Train

```powershell
python -m src.train --data data --experiment InJa_14k
```

The checkpoint and `report.json` go to `outputs/InJa_14k/`. The F1 numbers in that file are measured on the held-out photos. They are not copied from the paper.

A short run that draws its own images, for checking that training and scoring work:

```powershell
python -m src.train --synthetic --epochs 1 --no-pretrained --output-dir outputs/smoke
```

Those numbers are not RDD2020 results.

## Citation

```text
Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Mraz, A., Kashiyama, T., and Sekimoto, Y.
Transfer Learning-based Road Damage Detection for Multiple Countries.
arXiv:2008.13101, 2020.
```
