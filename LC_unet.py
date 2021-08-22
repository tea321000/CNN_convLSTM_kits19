import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLayer(nn.Module):
    def __init__(self, C_in, C_out):
        super(CNNLayer, self).__init__()
        self.conv = torch.nn.Conv2d(C_in, C_out, 3, 1, 1)
        # self.norm = torch.nn.BatchNorm3d(C_out)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        res = self.conv(x[:,:,0,:,:]).unsqueeze(2)
        for depth in range(1, x.shape[2]):
            res = torch.cat((res, self.conv(x[:,:,depth,:,:]).unsqueeze(2)),dim=2)

        return self.relu(res)




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
        # return torch.cat((x, r), 1)


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.num_features = num_features
        self.conv = self._make_layer(in_channels+num_features, num_features*4,
                                       kernel_size, padding, stride)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, inputs):
        '''
        :param inputs: (B, S, C, H, W)
        :param hidden_state: (hx: (B, S, C, H, W), cx: (B, S, C, H, W))
        :return:
        '''
        inputs = inputs.permute(0, 2, 1, 3, 4)
        outputs = []
        B, S, C, H, W = inputs.shape
        hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        for t in range(S):
            combined = torch.cat([inputs[:, t], # (B, C, H, W)
                                  hx], dim=1)
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            outputs.append(hy)
            hx = hy
            cx = cy

        return torch.stack(outputs).permute(1, 2, 0, 3, 4).contiguous() # (S, B, C, H, W) -> (B, S, C, H, W)


class LC_UNet(torch.nn.Module):
    def __init__(self):
        super(LC_UNet, self).__init__()
        kernel_size = (3, 3)
        num_layers = 1
        self.Conv1 = CNNLayer(1, 32)
        self.convLSTM1 = ConvLSTM(32, 32, kernel_size, num_layers)
        self.Downsample1 = Downsample(2)
        self.Conv2 = CNNLayer(32, 64)
        self.convLSTM2 = ConvLSTM(64, 64, kernel_size, num_layers)
        self.Downsample2 = Downsample(2)
        self.Conv3 = CNNLayer(64, 128)
        self.convLSTM3 = ConvLSTM(128, 128, kernel_size, num_layers)
        self.Downsample3 = Downsample(2)
        self.Conv4 = CNNLayer(128, 256)
        self.convLSTM4 = ConvLSTM(256, 256, kernel_size, num_layers)
        self.Downsample4 = Downsample(2)
        self.Conv5 = CNNLayer(256, 512)
        self.convLSTM5 = ConvLSTM(512, 512, kernel_size, num_layers)
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
        x = self.Conv1(x)
        lstm1 = self.convLSTM1(x)
        x = self.Downsample1(lstm1)
        x = self.Conv2(x)
        lstm2 = self.convLSTM2(x)
        x = self.Downsample2(lstm2)
        x = self.Conv3(x)
        lstm3 = self.convLSTM3(x)
        x = self.Downsample3(lstm3)
        x = self.Conv4(x)
        lstm4 = self.convLSTM4(x)
        x = self.Downsample4(lstm4)
        x = self.Conv5(x)
        x = self.convLSTM5(x)
        upsample1 = self.Upsample1(x)
        concat1 = torch.cat([upsample1, lstm4], dim=1)
        x = self.Conv6(concat1)
        upsample2 = self.Upsample2(x)
        concat2 = torch.cat([upsample2, lstm3], dim=1)
        x = self.Conv7(concat2)
        upsample3 = self.Upsample3(x)
        concat3 = torch.cat([upsample3, lstm2], dim=1)
        x = self.Conv8(concat3)
        upsample4 = self.Upsample4(x)
        concat4 = torch.cat([upsample4, lstm1], dim=1)
        x = self.Conv9(concat4)
        x = self.final(x)
        # x = self.sigmoid(x)
        return x


