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
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
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

        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class LossHeader(nn.Module):
    def __init__(self, inchannel, key, weights=None):
        super(LossHeader, self).__init__()

        # TODO: Define loss headers
        self.sequential = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            init(
                key + "conv",
                nn.Conv2d(in_channels=inchannel, out_channels=128, kernel_size=1),
                weights,
            ),
            nn.ReLU(),
            nn.Flatten(),
            init(
                key + "fc",
                nn.Linear(in_features=2048, out_features=1024),
                weights,
            ),
            nn.ReLU(),
            nn.Dropout(0.7),
        )
        self.xyz = nn.Linear(in_features=1024, out_features=3)
        self.wpqr = nn.Linear(in_features=1024, out_features=4)

    def forward(self, x):
        # TODO: Feed data through loss headers
        x = self.sequential(x)
        xyz = self.xyz(x)
        wpqr = self.wpqr(x)
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
            init(
                "conv2/3x3_reduce",
                nn.Conv2d(64, 64, kernel_size=1),
                weights,
            ),
            nn.ReLU(),
            init(
                "conv2/3x3",
                nn.Conv2d(64, 192, kernel_size=3),
                weights,
            ),
            nn.ReLU(),
            nn.LocalResponseNorm(2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Example for InceptionBlock initialization
        self._3a = InceptionBlock(
            192, 64, 96, 128, 16, 32, 32, "inception_3a/", weights
        )

        self._3b = InceptionBlock(
            256, 128, 128, 192, 32, 96, 64, "inception_3b/", weights
        )

        self._4a = InceptionBlock(
            480, 192, 96, 208, 16, 48, 64, "inception_4a/", weights
        )
        self._4b = InceptionBlock(
            512, 160, 112, 224, 24, 64, 64, "inception_4b/", weights
        )
        self._4c = InceptionBlock(
            512, 128, 128, 256, 24, 64, 64, "inception_4c/", weights
        )
        self._4d = InceptionBlock(
            512, 112, 144, 288, 32, 64, 64, "inception_4d/", weights
        )
        self._4e = InceptionBlock(
            528, 256, 160, 320, 32, 128, 128, "inception_4e/", weights
        )
        self._5a = InceptionBlock(
            832, 256, 160, 320, 32, 128, 128, "inception_5a/", weights
        )
        self._5b = InceptionBlock(
            832, 384, 192, 384, 48, 128, 128, "inception_5b/", weights
        )

        self.loss1 = LossHeader(512, "loss1/", weights)
        self.loss2 = LossHeader(528, "loss2/", weights)
        self.loss3_header = nn.Sequential(
            nn.AvgPool2d(kernel_size=7),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.loss3_xyz = nn.Linear(2048, 3)
        self.loss3_wpqr = nn.Linear(2048, 4)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        print("PoseNet model created!")

    def forward(self, x):
        # TODO: Implement PoseNet forward
        x = self.pre_layers(x)
        x = self._3a(x)
        x = self._3b(x)
        x = self.maxpool(x)
        x = self._4a(x)

        if self.training:
            loss1_xyz, loss1_wpqr = self.loss1(x)

        x = self._4b(x)
        x = self._4c(x)
        x = self._4d(x)

        if self.training:
            loss2_xyz, loss2_wpqr = self.loss2(x)

        x = self._4e(x)
        x = self.maxpool(x)

        x = self._5a(x)
        x = self._5b(x)

        x = self.loss3_header(x)
        loss3_xyz = self.loss3_xyz(x)
        loss3_wpqr = self.loss3_wpqr(x)

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
        def _lossi(x, q, GTx, GTq):
            # nn.functional.mse_loss()
            return nn.functional.mse_loss(x, GTx) + 300 * nn.functional.mse_loss(q, GTq)

        GTx = poseGT[:, 0:3]
        GTq = poseGT[:, 3:]
        loss = (
            0.3 * _lossi(p1_xyz, p1_wpqr, GTx, GTq)
            + 0.3 * _lossi(p2_xyz, p2_wpqr, GTx, GTq)
            + _lossi(p3_xyz, p3_wpqr, GTx, GTq)
        )
        return loss

    # def forward(self, p1_xyz, p1_wpqr, p2_xyz, p2_wpqr, p3_xyz, p3_wpqr, poseGT):
    #     # TODO: Implement loss
    #     # First 3 entries of poseGT are ground truth xyz, last 4 values are ground truth wpqr
    #     def _lossi(x, q, GTx, GTq):
    #         return (x - GTx).norm(dim=1) + 300 * (
    #             q - GTq / GTq.norm(dim=1).unsqueeze(1).expand_as(GTq)
    #         ).norm(dim=1)

    #     GTx = poseGT[:, 0:3]
    #     GTq = poseGT[:, 3:]
    #     loss = (
    #         0.3 * _lossi(p1_xyz, p1_wpqr, GTx, GTq)
    #         + 0.3 * _lossi(p2_xyz, p2_wpqr, GTx, GTq)
    #         + _lossi(p3_xyz, p3_wpqr, GTx, GTq)
    #     )
    #     return loss.mean()
