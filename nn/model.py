import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
torch.manual_seed(12321)


class Model1(nn.Module):
    '''
    Uses MobileNet_v2 as the CNN architecture
    and Feedforward layers to get a 30 x 2 (60x1 in row major) vector as output
    '''
    def __init__(self):
        super(Model1, self).__init__()
        model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT)
        model.features[0][0] = nn.Conv2d(
            2, 32, kernel_size=(3, 3), stride=(2, 2),
            padding=(1, 1), bias=False
        )

        # Feature Extractor
        self.fe = model.features

        # Fully Connected
        self.linear = nn.Sequential(
            nn.Dropout(0.2),
            # nn.Linear(1280, 1280),
            # nn.BatchNorm1d(1280),
            # nn.ReLU6(True),
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 60)
        )

    def forward(self, occ_map):
        features = self.fe(occ_map)
        # Same as torchvison
        # Ref: https://github.com/pytorch/vision/blob/main/torchvision/\
        #     models/mobilenetv2.py

        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)

        output = self.linear(features)

        return output


class Model_Temporal(nn.Module):
    '''
    Uses MobileNet_v2 as the CNN architecture
    and Feedforward layers to get a 30 x 2 (60x1 in row major) vector as output
    '''
    def __init__(self, past_frames):
        super(Model_Temporal, self).__init__()
        model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT)
        model.features[0][0] = nn.Conv2d(
            past_frames+1, 32, kernel_size=(3, 3), stride=(2, 2),
            padding=(1, 1), bias=False
        )

        # Feature Extractor
        self.fe = model.features

        # Fully Connected
        self.linear = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(True),
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 60)
        )

    def forward(self, occ_map):
        features = self.fe(occ_map)

        # Same as torchvison
        # Ref: https://github.com/pytorch/vision/blob/main/torchvision/\
        #     models/mobilenetv2.py
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)

        output = self.linear(features)

        return output
