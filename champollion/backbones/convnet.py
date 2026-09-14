from collections import OrderedDict

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
import numpy as np

def ComputeOutputDim(dimension, depth):
    """Compute the output resolution
    """
    if depth==0:
        return dimension
    else:
        return(ComputeOutputDim(dimension//2+dimension%2, depth-1))

class Conv3dSame(nn.Conv3d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((np.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: Tensor) -> Tensor:
        ih, iw, id = x.size()[-3:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])
        pad_d = self.calc_same_pad(i=id, k=self.kernel_size[2], s=self.stride[2], d=self.dilation[2])

        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            x = F.pad(
                x, [int(pad_d // 2), int(pad_d - pad_d // 2),
                    int(pad_w // 2), int(pad_w - pad_w // 2),
                    int(pad_h // 2), int(pad_h - pad_h // 2)]
            )
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ConvNet(pl.LightningModule):

    def __init__(self, in_channels, encoder_depth, block_depth,
                 num_representation_features, adaptive_pooling,
                 filters, initial_kernel_size, initial_stride,
                 max_pool, drop_rate, in_shape):

        super(ConvNet, self).__init__()

        self.num_representation_features = num_representation_features
        self.drop_rate = drop_rate

        # Decoder part
        self.in_shape = in_shape
        c, h, w, d = in_shape
        self.encoder_depth = encoder_depth
        self.filters = filters
        self.block_depth = block_depth
        self.initial_kernel_size = initial_kernel_size
        self.initial_stride = initial_stride
        self.max_pool = max_pool
        assert len(self.filters) >= encoder_depth, "Incomplete filters list given."

        if adaptive_pooling is None:
            # receptive field downsampled
            self.z_dim_h = ComputeOutputDim(h, self.encoder_depth)
            self.z_dim_w = ComputeOutputDim(w, self.encoder_depth)
            self.z_dim_d = ComputeOutputDim(d, self.encoder_depth)
            self.out_dim = self.z_dim_h*self.z_dim_w*self.z_dim_d
        else:
            self.out_dim = np.prod(adaptive_pooling[1])

        modules_encoder = []
        layer_name = ['', 'a', 'b', 'c']
        for step in range(encoder_depth):
            for depth in range(block_depth-1):
                name = layer_name[depth]
                in_channels = 1 if (step == 0 and depth==0) else out_channels
                kernel_size = self.initial_kernel_size if (step == 0 and depth==0) else 3
                stride = self.initial_stride if (step==0 and depth==0) else 1
                out_channels = filters[step]
                #out_channels = 16 if step == 0 else 16 * (2**step)
                modules_encoder.append(
                    (f'conv{step}{name}',
                    nn.Conv3d(in_channels, out_channels,
                            kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
                    ))
                modules_encoder.append(
                    (f'norm{step}{name}', nn.BatchNorm3d(out_channels)))
                modules_encoder.append((f'LeakyReLU{step}{name}', nn.LeakyReLU()))
                if (self.max_pool and step == 0 and depth==0):
                    modules_encoder.append(('MaxPool', nn.MaxPool3d((2,2,2))))
                modules_encoder.append(
                    (f'DropOut{step}{name}', nn.Dropout3d(p=drop_rate)))

            name=layer_name[block_depth-1]
            modules_encoder.append(
                (f'conv{step}{name}',
                Conv3dSame(in_channels=out_channels, out_channels=out_channels,
                        kernel_size=(3,3,3), stride=(2,2,2), groups=1, bias=True)
                ))
            modules_encoder.append(
                (f'norm{step}{name}', nn.BatchNorm3d(out_channels)))
            modules_encoder.append((f'LeakyReLU{step}{name}', nn.LeakyReLU()))
            modules_encoder.append(
                (f'DropOut{step}{name}', nn.Dropout3d(p=drop_rate)))
            self.num_features = out_channels
        # adaptive pool to ensure a fixed size linear layer accross regions
        if adaptive_pooling is not None:
            if adaptive_pooling[0]=='max':
                    modules_encoder.append(('AdaptiveMaxPool', nn.AdaptiveMaxPool3d(output_size=adaptive_pooling[1])))
            elif adaptive_pooling[0]=='average':
                    modules_encoder.append(('AdaptiveAvgPool', nn.AdaptiveAvgPool3d(output_size=adaptive_pooling[1])))
            else:
                raise ValueError("Wrong pooling name argument")
        # flatten and reduce to the desired dimension
        modules_encoder.append(('Flatten', nn.Flatten()))
        modules_encoder.append(
            ('Linear',
            nn.Linear(
                self.num_features*self.out_dim,
                self.num_representation_features)
            ))
        self.encoder = nn.Sequential(OrderedDict(modules_encoder))

    def forward(self, x):
        out = self.encoder(x)
        return out.squeeze(dim=1)
