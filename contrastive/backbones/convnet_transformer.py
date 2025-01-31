from collections import OrderedDict

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
import torch
import numpy as np


class _DropoutNd(nn.Module):
    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(_DropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)


class Dropout3d_always(_DropoutNd):
    r"""Randomly zero out entire channels (a channel is a 3D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Alwyas applies dropout also during evaluation

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout3d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zeroed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    Shape:
        - Input: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`.
        - Output: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`
                  (same shape as input).

    Examples::

        >>> m = nn.Dropout3d(p=0.2)
        >>> input = torch.randn(20, 16, 4, 32, 32)
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       https://arxiv.org/abs/1411.4280
    """

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout3d(input, self.p, True, self.inplace)


class ConvNet_Transformer(pl.LightningModule):
    r"""3D-ConvNet model class, based on

    Attributes:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first
            convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate
        num_classes (int) - number of classification classes
            (if 'classifier' mode)
        in_channels (int) - number of input channels (1 for sMRI)
        mode (str) - specify in which mode DenseNet is trained on,
            must be "encoder" or "classifier"
        memory_efficient (bool) - If True, uses checkpointing. Much more memory
            efficient, but slower. Default: *False*.
            See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, in_channels=1, encoder_depth=3, block_depth=2,
                 num_representation_features=256, linear=True,
                 adaptive_pooling=None, filters=[16,32,64], initial_kernel_size=3,
                 initial_stride=1, max_pool=False, nhead=4, num_transformer_layers=1, drop_rate=0.1, memory_efficient=False,
                 in_shape=None):

        super(ConvNet_Transformer, self).__init__()

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
        self.nhead = nhead
        self.num_transformer_layers = num_transformer_layers
        assert len(self.filters) >= encoder_depth, "Incomplete filters list given."

        if adaptive_pooling is None:
            # receptive field downsampled
            self.z_dim_h = h//2**self.encoder_depth
            self.z_dim_w = w//2**self.encoder_depth
            self.z_dim_d = d//2**self.encoder_depth
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
                nn.Conv3d(out_channels, out_channels,
                        kernel_size=4, stride=2, padding=1)
                ))
            modules_encoder.append(
                (f'norm{step}{name}', nn.BatchNorm3d(out_channels)))
            modules_encoder.append((f'LeakyReLU{step}{name}', nn.LeakyReLU()))
            modules_encoder.append(
                (f'DropOut{step}{name}', nn.Dropout3d(p=drop_rate)))
            self.num_features = out_channels

        self.encoder = nn.Sequential(OrderedDict(modules_encoder))


        # Positional Encoding
        self.positional_encoding = self._posemb_sincos_3d(self.z_dim_h, self.z_dim_w, self.z_dim_d, self.filters[-1])
        # Transformer Encoder Layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.filters[-1],
            nhead=self.nhead,
            dropout=self.drop_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_transformer_layers)

        # flatten
        self.flatten = nn.Flatten()

        # linear
        self.linear = nn.Linear(self.filters[-1]*self.out_dim, self.num_representation_features)
    
    def _posemb_sincos_3d(self, h, w, d, dim, temperature = 10000, dtype = torch.float32):

        z, y, x = torch.meshgrid(
            torch.arange(h),
            torch.arange(w),
            torch.arange(d),
        indexing = 'ij')

        fourier_dim = dim // 6

        omega = torch.arange(fourier_dim) / (fourier_dim - 1)
        omega = 1. / (temperature ** omega)

        z = z.flatten()[:, None] * omega[None, :]
        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :] 

        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)

        pe = F.pad(pe, (0, dim - (fourier_dim * 6))) # pad if feature dimension not cleanly divisible by 6
        return pe.type(dtype)


    def forward(self, x):
        x = self.encoder(x)
        # Flatten into sequences
        batch_size, embed_dim, H, W, D = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, embed_dim)  # Shape: (batch_size, num_patches, embed_dim)
        # Add fixed positional encoding
        x += self.positional_encoding.to(x.device)
        # Transformer Encoder
        x = self.transformer_encoder(x)  # Shape: (batch_size, num_patches, embed_dim)
        # Global average pooling over sequence length
        #x = x.mean(dim=1)  # Shape: (batch_size, embed_dim)
        x = self.flatten(x)
        # linear
        x = self.linear(x)

        return x.squeeze(dim=1)
    

# sinusoidal positional encoding : to fix !
"""
    def _generate_fixed_positional_encoding(self, embed_dim, H, W, D):
        
        pe = torch.zeros(embed_dim, H, W, D)
        div_term = torch.exp(torch.arange(0, embed_dim // 3 * 2, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        
        grid_z, grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, D),
            torch.linspace(0, 1, H),
            torch.linspace(0, 1, W),
            indexing="ij"
        )
        
        grid_x = grid_x.unsqueeze(0).repeat(embed_dim // 3, 1, 1, 1) * div_term[:, None, None, None]
        grid_y = grid_y.unsqueeze(0).repeat(embed_dim // 3, 1, 1, 1) * div_term[:, None, None, None]
        grid_z = grid_z.unsqueeze(0).repeat(embed_dim // 3, 1, 1, 1) * div_term[:, None, None, None]
        
        pe[0::3, :, :, :] = torch.sin(grid_x)
        pe[1::3, :, :, :] = torch.cos(grid_y)
        pe[2::3, :, :, :] = torch.sin(grid_z)
        return pe.unsqueeze(0)  # Add batch dimension
"""

"""
    def _generate_fixed_positional_encoding(self, embed_dim, H, W, D):
    
        grid_z, grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, D),
            torch.linspace(0, 1, H),
            torch.linspace(0, 1, W),
            indexing="ij"
        )
        # Combine the three dimensions into positional encoding
        pe = torch.stack((grid_x, grid_y, grid_z), dim=0)  # Shape: (3, H, W, D)
    
        # Repeat or interpolate to match embed_dim
        pe = pe.repeat(embed_dim // 3 + 1, 1, 1, 1)[:embed_dim, :, :, :]
        return pe.unsqueeze(0)  # Add batch dimension

# in the init :
# self.fixed_positional_encoding = self._generate_fixed_positional_encoding(self.filters[-1], self.z_dim_w, self.z_dim_d, self.z_dim_h) # TODO : h,w,d are not ordered properly
"""

""" # 2D case here ?
    def _get_sinusoid_encoding(self, num_tokens, token_len):
         Make Sinusoid Encoding Table

            Args:
                num_tokens (int): number of tokens
                token_len (int): length of a token
                
            Returns:
                (torch.FloatTensor) sinusoidal position encoding table
        

        def get_position_angle_vec(i):
            return [i / np.power(10000, 2 * (j // 2) / token_len) for j in range(token_len)]

        sinusoid_table = np.array([get_position_angle_vec(i) for i in range(num_tokens)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) 

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
"""
