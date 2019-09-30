# This code is mainly based on https://github.com/developer0hye/PyTorch-Darknet53/blob/master/model.py

from torch import nn

def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet53(nn.Module):
    def __init__(self, block, num_classes=1000):
        super(Darknet53, self).__init__()

        self.num_classes = num_classes

        features = [conv_batch(3, 32), ]
        darknet_setting = [
            # conv_in_channels, conv_out_channels, num_blocks
            [32, 64, 1],
            [64, 128, 2],
            [128, 256, 8],
            [256, 512, 8],
            [512, 1024, 4]
        ]
        for idx in range(5):
            features.extend(
                [conv_batch(darknet_setting[idx][0], darknet_setting[idx][1], stride=2),
                 self.make_layer(block, in_channels=darknet_setting[idx][1], num_blocks=darknet_setting[idx][2])
                 ]
            )
        self.features = nn.Sequential(*features)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)

        return out

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


def darknet53(num_classes):
    return Darknet53(DarkResidualBlock, num_classes)
