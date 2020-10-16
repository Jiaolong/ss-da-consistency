from torch import nn
from torch.utils import model_zoo
from torchvision import models
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck

from networks.model_utils import init_weights

resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, 
        "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}

class ResNetFc(nn.Module):
    def __init__(self, resnet_name='ResNet50', use_bottleneck=True, bottleneck_dim=256,
            aux_classes=1000, classes=100, output='all'):
        super(ResNetFc, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.output = output
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                             self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.use_bottleneck = use_bottleneck
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
            self.bottleneck.apply(init_weights)
            self.feat_dim = bottleneck_dim
        else:
            self.feat_dim = model_resnet.fc.in_features

        self.fc = nn.Linear(self.feat_dim, classes)
        self.fc.apply(init_weights)

        self.aux_classifier = nn.Linear(self.feat_dim, aux_classes)
        self.aux_classifier.apply(init_weights)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        if self.use_bottleneck:
            x = self.bottleneck(x)

        if self.output == 'feature':
            return x
        elif self.output == 'feature+class_logits':
            return x, self.fc(x)
        else:
            return self.aux_classifier(x), self.fc(x)

    def output_num(self):
        return self.feat_dim

    def get_parameters(self):
        if self.use_bottleneck:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                        {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                        {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2},
                        {"params":self.aux_classifier.parameters(), "lr_mult":10, 'decay_mult':2},
                        ]
        else:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                        {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2},
                        {"params":self.aux_classifier.parameters(), "lr_mult":10, 'decay_mult':2},
                        ]
        return parameter_list

class ResNet(nn.Module):
    def __init__(self, block, layers, aux_classes=1000, classes=100, output='all'):
        self.output = output
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # print(block.expansion)
        self.feat_dim = 512 * block.expansion
        
        self.aux_classifier = nn.Linear(512 * block.expansion, aux_classes)
        self.class_classifier = nn.Linear(512 * block.expansion, classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.normal_(m.bias)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def get_parameters(self):
        feat_param_list = [
                {"params":self.conv1.parameters(), "lr_mult":1, 'decay_mult':2},
                {"params":self.bn1.parameters(), "lr_mult":1, 'decay_mult':2},
                {"params":self.layer1.parameters(), "lr_mult":1, 'decay_mult':2},
                {"params":self.layer2.parameters(), "lr_mult":1, 'decay_mult':2},
                {"params":self.layer3.parameters(), "lr_mult":1, 'decay_mult':2},
                {"params":self.layer4.parameters(), "lr_mult":1, 'decay_mult':2}
                ]

        class_param_list = [ {"params":self.class_classifier.parameters(), "lr_mult":10, 'decay_mult':2} ]
        aux_param_list = [ {"params":self.aux_classifier.parameters(), "lr_mult":10, 'decay_mult':2} ]

        if self.output == 'feature':
            return feat_param_list
        elif self.output == 'feature+class_logits':
            return feat_param_list + class_param_list
        else:
            return feat_param_list + class_param_list + aux_param_list

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # B x C
        if self.output == 'feature':
            return x
        elif self.output == 'feature+class_logits':
            return x, self.class_classifier(x)
        else:
            return self.aux_classifier(x), self.class_classifier(x)

def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model

def resnet50_fc(resnet_name='ResNet50', **kwargs):
    model = ResNetFc(resnet_name=resnet_name, **kwargs)
    return model

def resnet18_fc(resnet_name='ResNet18', **kwargs):
    model = ResNetFc(resnet_name=resnet_name, **kwargs)
    return model
