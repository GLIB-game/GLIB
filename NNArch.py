import torch
import torch.nn as nn
cfg = [16, 16, 'M', 16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M']


class CNN(nn.Module):
    def __init__(self, init_weights=True):
        super(CNN, self).__init__()
        # convolutional layer
        self.features = make_layers(cfg)

        # full-connected layer
        self.classifier = nn.Sequential(
            nn.Linear(16 * 8 * 128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            
        )
        # output
        self.out = nn.Linear(128, 2)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  
        x = self.classifier(x)
        output = self.out(x)
        return output, x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':  # pooling
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == "__main__":
    cnn = CNN()
    for parameters in cnn.parameters():
        print(parameters)
    print(sum(p.numel() for p in cnn.parameters() if p.requires_grad))

