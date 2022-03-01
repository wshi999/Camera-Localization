from multiprocessing import pool
import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle


def init(key, module, weights=None):
    if weights == None:
        return module

    # initialize bias.data: layer_name_1 in weights
    # initialize  weight.data: layer_name_0 in weights
    module.bias.data = torch.from_numpy(weights[(key + "_1").encode()])
    module.weight.data = torch.from_numpy(weights[(key + "_0").encode()])

    return module


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        n1x1,
        n3x3red,
        n3x3,
        n5x5red,
        n5x5,
        pool_planes,
        key=None,
        weights=None,
    ):
        super(InceptionBlock, self).__init__()

        # TODO: Implement InceptionBlock
        # Use 'key' to load the pretrained weights for the correct InceptionBlock

        # 1x1 conv branch
        self.b1 = nn.Sequential(
            init(
                key + "1x1",
                nn.Conv2d(in_channels, n1x1, kernel_size=1, stride=1, padding=0),
                weights,
            ),
            nn.ReLU(),
        )

        # 1x1 -> 3x3 conv branch
        self.b2 = nn.Sequential(
            init(
                key + "3x3_reduce",
                nn.Conv2d(in_channels, n3x3red, kernel_size=1, stride=1, padding=0),
                weights,
            ),
            nn.ReLU(),
            init(
                key + "3x3",
                nn.Conv2d(in_channels, n3x3, kernel_size=3, stride=1, padding=1),
                weights,
            ),
            nn.ReLU(),
        )

        # 1x1 -> 5x5 conv branch
        self.b3 = nn.Sequential(
            init(
                key + "5x5_reduce",
                nn.Conv2d(in_channels, n5x5red, kernel_size=1, stride=1, padding=0),
                weights,
            ),
            nn.ReLU(),
            init(
                key + "5x5",
                nn.Conv2d(in_channels, n5x5, kernel_size=5, stride=1, padding=2),
                weights,
            ),
            nn.ReLU(),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1),
            init(
                key + "pool_proj",
                nn.Conv2d(in_channels, pool_planes, kernel_size=1, stride=1, padding=0),
                weights,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        # TODO: Feed data through branches and concatenate
        x1 = self.b1(x)
        x2 = self.b2(x)
        x3 = self.b3(x)
        x4 = self.b4(x)

        x = torch.cat((x1, x2, x3, x4))

        return x


class LossHeader(nn.Module):
    def __init__(self, key, weights=None):
        super(LossHeader, self).__init__()

        # TODO: Define loss headers

    def forward(self, x):
        # TODO: Feed data through loss headers

        return xyz, wpqr


class PoseNet(nn.Module):
    def __init__(self, load_weights=True):
        super(PoseNet, self).__init__()

        # Load pretrained weights file
        if load_weights:
            print("Loading pretrained InceptionV1 weights...")
            file = open("pretrained_models/places-googlenet.pickle", "rb")
            weights = pickle.load(file, encoding="bytes")
            file.close()
        # Ignore pretrained weights
        else:
            weights = None

        # TODO: Define PoseNet layers

        self.pre_layers = nn.Sequential(
            # Example for defining a layer and initializing it with pretrained weights
            init(
                "conv1/7x7_s2",
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                weights,
            ),
            nn.MaxPool2d(3, 2),
            nn.LocalResponseNorm(2),
            # init("conv1"),
        )

        # Example for InceptionBlock initialization
        self._3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32, "3a", weights)

        print("PoseNet model created!")

    def forward(self, x):
        # TODO: Implement PoseNet forward

        if self.training:
            return loss1_xyz, loss1_wpqr, loss2_xyz, loss2_wpqr, loss3_xyz, loss3_wpqr
        else:
            return loss3_xyz, loss3_wpqr


class PoseLoss(nn.Module):
    def __init__(self, w1_xyz, w2_xyz, w3_xyz, w1_wpqr, w2_wpqr, w3_wpqr):
        super(PoseLoss, self).__init__()

        self.w1_xyz = w1_xyz
        self.w2_xyz = w2_xyz
        self.w3_xyz = w3_xyz
        self.w1_wpqr = w1_wpqr
        self.w2_wpqr = w2_wpqr
        self.w3_wpqr = w3_wpqr

    def forward(self, p1_xyz, p1_wpqr, p2_xyz, p2_wpqr, p3_xyz, p3_wpqr, poseGT):
        # TODO: Implement loss
        # First 3 entries of poseGT are ground truth xyz, last 4 values are ground truth wpqr

        return loss
