"""ResNet-18 for the shared binary task, plus backbone extraction."""
import copy
import torch.nn as nn
from torchvision import models


def build_model(pretrained=True, n_out=2):
    """Every domain uses the SAME 2-way head -- that is what enables zero-shot."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, n_out)
    return net


def split_model(net):
    """
    Return (backbone, head).

    backbone: outputs the 512-number feature vector
    head:     the trained 2-way classifier, kept SEPARATELY so we can apply it
              to those features and get zero-shot predictions from the same
              forward pass. One pass gives features AND predictions.
    """
    bb = copy.deepcopy(net)
    head = copy.deepcopy(net.fc)
    bb.fc = nn.Identity()
    return bb, head
