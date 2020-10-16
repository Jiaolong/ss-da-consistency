import os
from collections import OrderedDict
from itertools import chain

import torch
from torch import nn as nn

from networks.alexnet import Id
from networks.model_utils import ReverseLayerF


class AlexNetCaffe(nn.Module):
    def __init__(self, aux_classes=1000, n_classes=100, domains=3, dropout=True, feature_only=False):
        super(AlexNetCaffe, self).__init__()
        print("Using Caffe AlexNet")
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.feature_only = feature_only
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout() if dropout else Id()),
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout() if dropout else Id())]))

        self.feat_dim = 4096
        if not feature_only:
            self.aux_classifier = nn.Linear(4096, aux_classes)
            self.class_classifier = nn.Linear(4096, n_classes)
        # self.domain_classifier = nn.Sequential(
        #     nn.Linear(256 * 6 * 6, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(1024, domains))

    def get_params(self, base_lr):
        return [{"params": self.features.parameters(), "lr": 0.},
                {"params": chain(self.classifier.parameters(), self.aux_classifier.parameters()
                                 , self.class_classifier.parameters()#, self.domain_classifier.parameters()
                                 ), "lr": base_lr}]

    def is_patch_based(self):
        return False

    def forward(self, x, lambda_val=0):
        x = self.features(x*57.6)  #57.6 is the magic number needed to bring torch data back to the range of caffe data, based on used std
        x = x.view(x.size(0), -1)
        #d = ReverseLayerF.apply(x, lambda_val)
        x = self.classifier(x)
        if self.feature_only:
            return x
        else:
            return self.aux_classifier(x), self.class_classifier(x)#, self.domain_classifier(d)

def caffenet(aux_classes=1000, classes=100, feature_only=False):
    model = AlexNetCaffe(aux_classes, classes, feature_only=feature_only)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, .1)
            nn.init.constant_(m.bias, 0.)

    state_dict = torch.load(os.path.join(os.path.dirname(__file__), "pretrained/alexnet_caffe.pth.tar"))
    del state_dict["classifier.fc8.weight"]
    del state_dict["classifier.fc8.bias"]
    model.load_state_dict(state_dict, strict=False)

    return model
