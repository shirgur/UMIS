import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.resnet import *


class DeepVess(nn.Module):
    def __init__(self):
        super(DeepVess, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, 3),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2), stride=(2, 2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, (1, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(64, 64, (1, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2), stride=(2, 2, 2))
        )

        self.fc1 = nn.Linear(64 * 1 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 2 * 1 * 4 * 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], 64 * 1 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.reshape(x.shape[0], 2, 1, 4, 4)

        return x


class VessNN(nn.Module):
    def __init__(self):
        super(VessNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 24, (2, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(24, 24, (2, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(24, 24, (2, 3, 3)),
            nn.Tanh(),
            nn.MaxPool3d((1, 2, 2), stride=(1, 1, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(24, 36, (1, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(36, 36, (1, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2), stride=(1, 1, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(36, 48, (2, 3, 3), padding=(1, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(48, 48, (2, 3, 3), padding=(1, 0, 0)),
            nn.Tanh(),
            nn.MaxPool3d((2, 2, 2), stride=(1, 1, 1))
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(48, 60, (2, 3, 3), padding=(1, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(60, 60, (2, 3, 3), padding=(1, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(60, 100, (2, 3, 3), padding=(1, 0, 0)),
            nn.ReLU()
        )
        self.drop = nn.Dropout(0.5)

        self.fc = nn.Linear(100 * 5 * 62 * 62, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.drop(x)
        x = x.reshape(x.shape[0], 100 * 5 * 62 * 62)
        x = self.fc(x)
        x = x.reshape(x.shape[0], 2, 1, 1, 1)

        return x


class SegmentNet3D_Resnet(nn.Module):
    def __init__(self):
        super(SegmentNet3D_Resnet, self).__init__()

        self.base = resnet34()
        self.du = nn.Dropout(0.2)

        # Seg
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(512, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 64, (1, 3, 3), 1, padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(256 + 64, 32, 4, 2, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 32, (1, 3, 3), 1, padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(128 + 32, 8, 4, 2, 1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.ConvTranspose3d(8, 8, (1, 3, 3), 1, padding=(0, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.out_conv = nn.Sequential(
            nn.ConvTranspose3d(8, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        # Rec
        self.deconv1_rec = nn.Sequential(
            nn.ConvTranspose3d(512, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 64, 1, 1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.deconv2_rec = nn.Sequential(
            nn.ConvTranspose3d(256 + 64, 32, 4, 2, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 32, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.deconv3_rec = nn.Sequential(
            nn.ConvTranspose3d(128 + 32, 8, 4, 2, 1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.ConvTranspose3d(8, 8, 1, 1),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.rec_conv = nn.Sequential(
            nn.ConvTranspose3d(8, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        c1, c2, c3, c4 = self.base(x)

        c4 = self.du(c4)

        dc1 = self.deconv1(c4)
        cat = torch.cat((c3, dc1), 1)
        cat = F.interpolate(cat, size=(x.shape[2] // 8, x.shape[3] // 8, x.shape[4] // 8), mode='trilinear')
        dc2 = self.deconv2(cat)
        _c2 = F.interpolate(c2, size=dc2.shape[2:], mode='trilinear')
        cat = torch.cat((_c2, dc2), 1)
        out = self.deconv3(cat)
        out = F.interpolate(out, size=x.shape[2:], mode='trilinear')
        out = self.out_conv(out)

        dcr1 = self.deconv1_rec(c4)
        dcr2 = self.deconv2_rec(torch.cat((c3, dcr1), 1))
        rec = self.deconv3_rec(torch.cat((c2, dcr2), 1))
        rec = F.interpolate(rec, size=x.shape[2:], mode='trilinear')
        rec = self.out_conv(rec)

        return out, rec
