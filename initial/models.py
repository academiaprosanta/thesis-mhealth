"""ResNet-18 construction and turning a trained model into a feature extractor."""
import copy
import torch.nn as nn
from torchvision import models


def build_model(n_out, pretrained=True):
    """
    ResNet-18 with the final layer swapped for one that outputs n_out logits.

    'pretrained' means: start from weights already trained on ImageNet.
    This is stage 1 of the three-stage picture -- a generic starting point,
    identical for every dataset, so it carries no pair-specific information.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, n_out)
    return net


def as_backbone(net):
    """
    Strip the classifier. What comes out is a 512-number vector per image --
    the 'features' or 'embedding'. All geometry metrics are computed on these.

    Returns a copy, so the original model is untouched.
    """
    bb = copy.deepcopy(net)
    bb.fc = nn.Identity()
    return bb
