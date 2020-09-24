import torch
import torch.nn as nn
from ConvLSTM import ConvLSTM_LayerNorm


class C3DFeatures(nn.Module):
    def __init__(self, inplace=False):
        super(C3DFeatures, self).__init__()
        self.convolution1_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=inplace)
        )

        self.convolution1_2 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=inplace)
        )

        self.convolution2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=inplace),
        )

    def forward(self, dis, err):
        # input batch_size, seq_len, channel, depth, height, width
        # output batch_size, seq_len, channel, height, width
        features = []
        for t_index in range(dis.size(1)):
            x = self.convolution1_1(dis[:, t_index, :, :, :, :])
            y = self.convolution1_2(err[:, t_index, :, :, :, :])
            x = x.squeeze(2)
            y = y.squeeze(2)
            out = torch.cat((x, y), dim=1)
            out = self.convolution2(out)
            features.append(out)
        features = torch.stack(features, dim=1)
        return features


class FeatureConvolution(nn.Module):
    def __init__(self, channels=64, size=28):
        super(FeatureConvolution, self).__init__()
        self.convlstm = ConvLSTM_LayerNorm(input_size=(channels, size, size),
                                           hidden_size=(channels, size, size), elementwise_affine=True)

    def forward(self, x):
        out = self.convlstm(x)
        return out


class Many2One(nn.Module):
    def __init__(self, inplace=False):
        super(Many2One, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=inplace)
        )

    def forward(self, x):
        # batch first
        # output batch_size, seq_len, channel, height, width
        video_map = []
        for t_index in range(x.size(1)):
            out = self.convolution(x[:, t_index, :, :, :])
            video_map.append(out)
        video_map = torch.stack(video_map, dim=1)
        return video_map


class ScoreRegression(nn.Module):
    def __init__(self):
        super(ScoreRegression, self).__init__()
        self.R = nn.Sequential(
            nn.Linear(in_features=1, out_features=4),
            nn.LeakyReLU(),
            nn.Linear(in_features=4, out_features=1),
            nn.LeakyReLU()
        )

    def forward(self, score):
        out = self.R(score)
        return out


class SQA(nn.Module):
    def __init__(self):
        super(SQA, self).__init__()
        self.part1 = C3DFeatures()
        self.part2 = FeatureConvolution()
        self.part3 = Many2One()

    def forward(self, dis, err, err_4):
        out = self.part1(dis, err)
        out = self.part2(out)
        out = self.part3(out)
        score = out*err_4
        # batch_size, seq_len, channel, height, width
        score = torch.mean(score[:, :, :, 4:-4, 4:-4], dim=[1, 3, 4])
        return score, out


class TQA(nn.Module):
    def __init__(self):
        super(TQA, self).__init__()
        self.lstm = nn.LSTM(input_size=1, bidirectional=True, hidden_size=64, batch_first=True)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64*2, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=1),
            nn.LeakyReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1, out_features=1),
            nn.LeakyReLU(),
        )

    def forward(self, predict):
        # batch_size, len, size
        c0 = h0 = torch.rand(2, predict.shape[0], 64).to(predict.device)
        out, _ = self.lstm(predict, (h0, c0))
        out1 = self.fc1(out)

        out2 = self.fc2(predict)

        out = out1*out2
        out = torch.mean(out, dim=1)
        return out