import torch
import torch.nn as nn
import fvcore.nn.weight_init as weight_init

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        # init
        for layer in self.fc.modules():
            if isinstance(layer, nn.Conv2d):
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # init
        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    code is modified from https://github.com/luuuyi/CBAM.PyTorch and
    https://github.com/Jongchan/attention-module
    It will initialize when it's created.
    """
    def __init__(self, in_channels, reduction_ratio=16, use_spatial=True):
        super(CBAM, self).__init__()
        self.channel_attention_module = ChannelAttention(in_channels, reduction_ratio)
        self.use_spatial = use_spatial
        if use_spatial:
            self.spatial_attention_module = SpatialAttention()
        
    def forward(self, x):
        x = self.channel_attention_module(x)
        if self.use_spatial:
            x = self.spatial_attention_module(x)
        return x
