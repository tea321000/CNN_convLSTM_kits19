import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLayer(nn.Module):
    def __init__(self, C_in, C_out):
        super(CNNLayer, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv3d(C_in, C_out, 3, 1, 1),
            torch.nn.BatchNorm3d(C_out),
            torch.nn.ReLU(),
            torch.nn.Conv3d(C_out, C_out, 3, 1, 1),
            torch.nn.BatchNorm3d(C_out),
            torch.nn.ReLU()
        )
    def forward(self, x):
        return self.layer(x)

class Downsample(nn.Module):
    def __init__(self, kernel_size):
        super(Downsample, self).__init__()
        self.layer = torch.nn.MaxPool3d(kernel_size)
    def forward(self, x):
        return self.layer(x)

class Upsample(nn.Module):
    def __init__(self, C):
        super(Upsample, self).__init__()
        self.C = torch.nn.Conv3d(C, C // 2, 1, 1)

    def forward(self, x):
        up = F.interpolate(x, scale_factor=2, mode='trilinear')
        return self.C(up)

class ThreeDCNN(nn.Module):
    def __init__(self):
        super(ThreeDCNN, self).__init__()
        self.Conv1 = CNNLayer(1, 32)
        self.Downsample1 = Downsample(2)
        self.Conv2 = CNNLayer(32, 64)
        self.Downsample2 = Downsample(2)
        self.Conv3 = CNNLayer(64, 128)
        self.Downsample3 = Downsample(2)
        self.Conv4 = CNNLayer(128, 256)
        self.Downsample4 = Downsample(2)
        self.Conv5 = CNNLayer(256, 512)
        self.Upsample1 = Upsample(512)
        self.Conv6 = CNNLayer(512, 256)
        self.Upsample2 = Upsample(256)
        self.Conv7 = CNNLayer(256, 128)
        self.Upsample3 = Upsample(128)
        self.Conv8 = CNNLayer(128, 64)
        self.Upsample4 = Upsample(64)
        self.Conv9 = CNNLayer(64, 32)
        self.final = torch.nn.Conv3d(32, 3, 3, 1, 1)

    def forward(self, x):
        x1= self.Conv1(x)
        x=self.Downsample1(x1)
        x2=self.Conv2(x)
        x=self.Downsample2(x2)
        x3=self.Conv3(x)
        x=self.Downsample3(x3)
        x4=self.Conv4(x)
        x=self.Downsample4(x4)
        x=self.Conv5(x)
        x=self.Upsample1(x)
        x=torch.cat([x,x4],dim=1)
        x=self.Conv6(x)
        x=self.Upsample2(x)
        x=torch.cat([x,x3],dim=1)
        x=self.Conv7(x)
        x=self.Upsample3(x)
        x=torch.cat([x,x2],dim=1)
        x=self.Conv8(x)
        x=self.Upsample4(x)
        x=torch.cat([x,x1],dim=1)
        x=self.Conv9(x)
        x=self.final(x)
        return x
