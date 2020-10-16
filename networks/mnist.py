from torch import nn

class LeNet(nn.Module):
    def __init__(self, aux_classes=1000, classes=100, feature_only=False):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )
        
        self.feat_dim = 500
        self.fc = nn.Sequential(nn.Linear(50*4*4, 500), nn.ReLU(), nn.Dropout(p=0.5))

        self.feature_only = feature_only
        if not self.feature_only:
            self.aux_classifier = nn.Linear(self.feat_dim, aux_classes)
            self.class_classifier = nn.Linear(self.feat_dim, classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.feature_only:
            return x
        else:
            return self.aux_classifier(x), self.class_classifier(x)

# built as https://github.com/ricvolpi/generalize-unseen-domains/blob/master/model.py
class MnistModel(nn.Module):
    def __init__(self, aux_classes=1000, n_classes=100):
        super().__init__()

        self.feat_dim = 1024
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
        )
#         outfeats = 100
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, 5),
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(32, 48, 5),
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2)
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(48 * 4 * 4, 100),
#             nn.ReLU(True),
#             nn.Linear(100, outfeats),
#             nn.ReLU(True),
#         )
        print("Using LeNet (%d)" % outfeats)
        self.aux_classifier = nn.Linear(outfeats, aux_classes)
        self.class_classifier = nn.Linear(outfeats, n_classes)

    def get_params(self, base_lr):
        raise "No pretrained exists for LeNet - use train all"

    def is_patch_based(self):
        return False

    def forward(self, x, lambda_val=0):
        # print(x.shape)
        x = self.features(x)
        # print(x.shape)
        x = self.classifier(x.view(x.size(0), -1))
        return self.aux_classifier(x), self.class_classifier(x)


def lenet(aux_classes=4, classes=10, **kwargs):
    model = LeNet(aux_classes, classes, **kwargs)
    return model
