"""SSD with a MobileNet backbone.

Arya et al. trained this detector with the TensorFlow Object Detection API and
a MobileNet backbone, starting from a model that had already been trained on a
general image set. This file is that same setup in current PyTorch: SSDLite
with MobileNetV3, ImageNet backbone weights, and a new head for the four
damage classes. Class 0 is the background class the detector expects.
"""

from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from src.experiments import CLASSES


def build_ssd_mobilenet(pretrained_backbone=True):
    backbone_weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    return ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=backbone_weights,
        num_classes=len(CLASSES) + 1,
        trainable_backbone_layers=6,
    )
