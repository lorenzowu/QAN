import torch
import torch.nn as nn
import numpy as np
import os
from scipy import stats
import random


class ReadY(object):
    def __init__(self, path, width, height):
        self.path = path
        self.width = width
        self.height = height

    def __call__(self, video_name):
        img_size = int(self.width * self.height * 1.5)
        y_size = self.width * self.height
        yuv_name = self.path + video_name
        frames = os.path.getsize(yuv_name) / img_size
        if frames % 1 == 0:
            frames = int(frames)
        else:
            raise Exception("Wrong width and height!")
        yuv = np.fromfile(yuv_name, dtype=np.uint8)
        bgr_all = []
        for k in range(frames):
            frame_yuv = yuv[k * img_size: (k + 1) * img_size]
            y_frame = frame_yuv[0: y_size].reshape(self.height, self.width)
            bgr_all.append(y_frame)
        bgr_all = np.stack(bgr_all, axis=0)
        return bgr_all, frames


def log_diff(dis, ref, epsilon=1.0):
    return torch.log(1.0/((dis-ref)**2+epsilon/65025.0))/torch.log(torch.tensor(65025.0/epsilon))


class ImageDownSample(nn.Module):
    def __init__(self, level=2):
        super(ImageDownSample, self).__init__()
        self.level = level
        self.convolution = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=2, bias=False)
        kernel = torch.Tensor([1, 4, 6, 4, 1])
        kernel = torch.ger(kernel, kernel)
        kernel = kernel/torch.sum(kernel)
        kernel = torch.reshape(kernel, (1, 1, 5, 5))
        self.convolution.weight = nn.Parameter(kernel)
        [para.requires_grad_(False) for para in self.convolution.parameters()]

    def forward(self, input_tensor):
        out = input_tensor
        for k in range(self.level):
            out = self.convolution(out)
        return out


class VideoTemporalSample(object):
    """Transform a video segment into clips."""
    def __init__(self, clip_len=None, clip_size=None):
        assert isinstance(clip_len, int)
        assert isinstance(clip_size, int)
        self.clip_len = clip_len
        self.clip_size = clip_size

    # input: (seq_len, 1, height, width)
    # output: (seq_len, depth, 1, height, width)
    def __call__(self, videos):
        video_len = len(videos)
        start_index = np.linspace(start=self.clip_size-1, stop=video_len - 1, num=self.clip_len, dtype=np.int16)
        video_clips = []
        for idx in start_index:
            video_clips.append(videos[(idx-self.clip_size+1):(idx+1), ])
        video_clips = torch.stack(video_clips, dim=0)
        # (seq_len, depth, 1, height, width) --> (seq_len, 1, depth, height, width)
        video_clips = video_clips.permute(0, 2, 1, 3, 4)
        return video_clips


def fit_function(predict, mos):
    params = np.polyfit(predict, mos, deg=3)
    p1 = np.poly1d(params)
    predict_fitted = p1(predict)
    return predict_fitted


def plcc_srcc(predict, label):
    predict_fitted = fit_function(predict, label)
    plcc, _ = stats.pearsonr(predict_fitted, label)
    srcc, _ = stats.spearmanr(predict_fitted, label)
    return np.float32(abs(plcc)), np.float32(abs(srcc))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

