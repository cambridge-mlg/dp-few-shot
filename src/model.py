import sys

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import vit_base_patch16_224_in21k

from bit_resnet import KNOWN_MODELS
from film import enable_film, get_film_parameter_names


def create_feature_extractor(feature_extractor_name, learnable_params):
    num_classes = 0

    if feature_extractor_name == 'vit-b-16':
        feature_extractor = vit_base_patch16_224_in21k(pretrained=True, num_classes=num_classes)
    elif 'BiT' in feature_extractor_name:
        feature_extractor = KNOWN_MODELS[feature_extractor_name](head_size=num_classes, zero_head=True)
        feature_extractor.load_from(np.load(f"{feature_extractor_name}.npz"))
    else:
        print("Invalid feature extractor option.")
        sys.exit()

    film_param_names = None
    if learnable_params == 'film':
        # freeze all the model parameters
        for param in feature_extractor.parameters():
            param.requires_grad = False

        film_param_names = get_film_parameter_names(
            feature_extractor_name=feature_extractor_name,
            feature_extractor=feature_extractor
        )

        enable_film(film_param_names, feature_extractor)
    elif learnable_params == 'none':
        # freeze all the model parameters
        for param in feature_extractor.parameters():
            param.requires_grad = False

    return feature_extractor, film_param_names


class DpFslLinear(nn.Module):
    def __init__(self, feature_extractor_name, num_classes, learnable_params):
        super(DpFslLinear, self).__init__()

        self.feature_extractor, _ = create_feature_extractor(
            feature_extractor_name=feature_extractor_name,
            learnable_params=learnable_params
        )

        # send a test signal through the feature extractor to get its feature dimension
        feature_extractor_dim = self.feature_extractor(torch.Tensor(1, 3, 224, 224)).size(1)

        self.head = nn.Linear(feature_extractor_dim, num_classes)
        self.head.weight.data.fill_(0.0)
        self.head.bias.data.fill_(0.0)

    def forward(self, x):
        return self.head(self.feature_extractor(x))
