import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
torch.manual_seed(42)

class Model1(nn.Module):
    '''
    Uses MobileNet_v2 as the CNN architecture
    and Feedforward layers to get a 30 x 2 vector as output
    '''
    def __init__(self, feature_size=256):
            super(Model1, self).__init__()
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            # l = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
            # print(model)
            # print(list(model.children()))
            # quit()
            model.features[0][0] = nn.Conv2d(
                1, 32, kernel_size=(3, 3), stride=(2,2), padding=(1,1), bias=False
            )
            # print(model.features[0][0])
            modules = list(model.children())[:-1]
            # print(modules[-1])
            self.cnn = nn.Sequential(*modules)

            self.to_linear = nn.Sequential(
                nn.Dropout(0.2),
                # nn.Linear(81920, 1280, bias=True),
                # nn.ReLU(True),
                # nn.Linear(40960, 20480, bias=True),
                # nn.ReLU(True),
                # nn.Linear(20480, 1280, bias=True),
                # nn.ReLU(True),
                nn.Linear(1280, feature_size, bias=True)
            )

            self.output = nn.Sequential(
                nn.Linear(feature_size, int(feature_size/2)),
                nn.BatchNorm1d(int(feature_size/2)),
                nn.ReLU(True),
                nn.Linear(int(feature_size/2),int(feature_size/2)),
                nn.BatchNorm1d(int(feature_size/2)),
                nn.ReLU(True),
                nn.Linear(int(feature_size/2),60)
            )

    def forward(self, occ_map):
        features = self.cnn(occ_map)
        # features = features.flatten()
        features = features.mean([2, 3])
        features = self.to_linear(features)
        output = self.output(features)
        return output

