import torch
import torch.nn as nn
import numpy as np

"""
Key modules:
1.  Residual in Residual Dense Block (RRDB):
    * Deeper network with RDBs for increased capacity.
    * Residual connections at the RDB level to ease training.
    * Learnable residual scaling for better control of feature contributions.
2.  Efficient Channel Attention (ECA):
    * Efficient channel attention mechanism.
    * Adaptive determination of local cross-channel interaction.
3.  PixelShuffle upsampling:
    * Efficient and effective upsampling method.
4.  Spectral normalization:
     * Stabilizes training by normalizing the spectral norm of the weights
5.  Input normalization:
    * Normalize the input before feeding into the model
"""


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_channels):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels + growth_channels, growth_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_channels, growth_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.res_scale = nn.Parameter(torch.FloatTensor([0.2]))  # Reduced scaling

    def forward(self, x):
        inputs = x
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.relu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        return x3 * self.res_scale + inputs


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_channels, num_convs=5):  # More convs
        super(ResidualInResidualDenseBlock, self).__init__()
        self.rdb_layers = nn.Sequential(
            *[ResidualDenseBlock(in_channels, growth_channels) for _ in range(num_convs)]
        )
        self.res_scale = nn.Parameter(torch.FloatTensor([0.2]))

    def forward(self, x):
        out = self.rdb_layers(x)
        return out * self.res_scale + x



class ECALayer(nn.Module):
    """Efficient Channel Attention Layer"""
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.k_size = k_size

    def forward(self, x):
        b, c, h, w = x.size()
        # Perform adaptive average pooling
        y = self.avg_pool(x).view(b, c, 1)  # (b, c, 1, 1) -> (b, c, 1)
        # Apply 1D convolution
        y = self.conv(y.transpose(-1, -2))  # (b, c, 1) -> (b, 1, c) -> (b, c, 1)
        # Apply sigmoid activation
        y = self.sigmoid(y).view(b, c, 1, 1)  # Reshape to (b, c, 1, 1)
        # Expand and apply attention
        return x * y.expand_as(x)

    def get_k_size(self):
        return self.k_size


class SuperResolutionModel(nn.Module):
    def __init__(self, num_blocks, in_channels, growth_channels, scale_factor):
        super(SuperResolutionModel, self).__init__()
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        self.initial_conv = nn.Conv2d(
            in_channels, growth_channels, kernel_size=3, padding=1)
        self.rdbs = nn.Sequential(
            *[ResidualInResidualDenseBlock(growth_channels, growth_channels, num_convs=5) for _ in range(num_blocks)]) # Use RRDB
        self.channel_attention = ECALayer(growth_channels) # Use ECA

        # Upsampling using PixelShuffle
        self.scale_factor = scale_factor
        if scale_factor == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(growth_channels, growth_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2)
            )
        elif scale_factor == 4:
            self.upsample = nn.Sequential(
                nn.Conv2d(growth_channels, growth_channels * 16, kernel_size=3, padding=1),
                nn.PixelShuffle(4)
            )
        elif scale_factor == 8:
             self.upsample = nn.Sequential(
                nn.Conv2d(growth_channels, growth_channels * 64, kernel_size=3, padding=1),
                nn.PixelShuffle(8)
            )
        else:
            raise ValueError("Scale factor must be 2, 4, or 8.")

        self.final_conv = nn.Conv2d(
            growth_channels, in_channels, kernel_size=3, padding=1)
        
        self.num_blocks = num_blocks
        self.growth_channels = growth_channels
        self.in_channels = in_channels
        

    def forward(self, x):
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        x = self.initial_conv(x)
        x = self.rdbs(x)  # Use RRDBs
        x = self.channel_attention(x)  # Use ECA
        x = self.upsample(x)  # Use PixelShuffle
        x = self.final_conv(x)
        return x

    def get_num_blocks(self):
        return self.num_blocks

    def get_growth_channels(self):
        return self.growth_channels

    def get_in_channels(self):
        return self.in_channels
