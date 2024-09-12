import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class ClipLoss(nn.Module):

    def __init__(self):
        super(ClipLoss, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_logits(self, image_features, text_features):
        logits_per_image = self.logit_scale.exp() * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features)
        labels = torch.arange(logits_per_image.shape[0]).to(device)
        return F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)
