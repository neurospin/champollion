from collections import OrderedDict

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor, ones, linspace, cat, rand, mul, empty
import numpy as np

from contrastive.backbones.coord_conv import CoordConv3d


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
    

class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 3D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W x D.
                     Can be a tuple (H, W, D) or a single H for a square image H x H x H
                     H, W and D can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return nn.functional.adaptive_avg_pool3d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'
    

class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(ones(1) * norm)


class GeneralizedMeanPoolingPerMap_LNP(nn.Module):
    r"""Applies a 3D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W x D.
                     Can be a tuple (H, W, D) or a single H for a square image H x H x H
                     H, W and D can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    Here p is variable, and specific to each feature map.
    """

    def __init__(self, n_features, init_min=1.5, init_max=5., output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingPerMap_LNP, self).__init__()
        #self.p = nn.Parameter(ones(n_features) * init_norm, requires_grad=True) ### TODO: change the init
        self.p = nn.Parameter(linspace(init_min, init_max, steps=n_features), requires_grad=True) # better to start randomly ?
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        p_broadcast = self.p.to(x.device).view(1, -1, 1, 1, 1)
        x = x.clamp(min=self.eps).pow(p_broadcast)
        out = nn.functional.adaptive_avg_pool3d(x, self.output_size).pow(1. / p_broadcast)
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'
    

class GeneralizedMeanPoolingPerMap_Mix(nn.Module):
    r"""Applies a 3D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W x D.
                     Can be a tuple (H, W, D) or a single H for a square image H x H x H
                     H, W and D can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    Here p is variable, and specific to each feature map.
    """

    def __init__(self, n_features, output_size=1):
        super(GeneralizedMeanPoolingPerMap_Mix, self).__init__()
        self.p = nn.Parameter(rand(n_features), requires_grad=True)
        self.output_size = output_size

    def forward(self, x):
        p_broadcast = self.p.to(x.device).view(1, -1, 1, 1, 1)
        out = p_broadcast*nn.functional.adaptive_max_pool3d(x, self.output_size) + (1-p_broadcast)*nn.functional.adaptive_avg_pool3d(x, self.output_size)
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'
    

class GeneralizedMeanPoolingPerMap_Atten(nn.Module):
    r"""Applies a 3D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W x D.
                     Can be a tuple (H, W, D) or a single H for a square image H x H x H
                     H, W and D can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    Here p is variable, and specific to each feature map.
    """

    def __init__(self, n_features, h, w, d, output_size=1):
        super(GeneralizedMeanPoolingPerMap_Atten, self).__init__()
        self.size = h*w*d
        #self.p = nn.Parameter(rand((n_features, h, w, d)), requires_grad=True) ## Kaiming uniform ? Xavier ? [0,1] ?
        self.p = nn.Parameter(empty((n_features, h, w, d)), requires_grad=True)
        nn.init.xavier_uniform_(self.p) # xavier is good with softmax ?
        self.output_size = output_size

    def forward(self, x):
        p_norm = nn.functional.softmax(self.p, dim=1)
        p_broadcast = p_norm.to(x.device).unsqueeze(0)
        out = self.size*nn.functional.adaptive_avg_pool3d(mul(x, p_broadcast), self.output_size) # need to sum pool instead of average pool
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'



class ConvNet_Test(pl.LightningModule):
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
                 initial_stride=1, max_pool=False, drop_rate=0.1, memory_efficient=False,
                 in_shape=None, init_min=1.5, init_max=5, coordconv_out_channels=4, scaling=1, with_r=False):

        super(ConvNet_Test, self).__init__()

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
        if adaptive_pooling is not None and adaptive_pooling[0]=="generalized_mean_per_map":
             self.init_min = init_min
             self.init_max = init_max

        # receptive field downsampled
        self.z_dim_h = h//2**self.encoder_depth
        self.z_dim_w = w//2**self.encoder_depth
        self.z_dim_d = d//2**self.encoder_depth
        if adaptive_pooling is None:
            self.out_dim = self.z_dim_h*self.z_dim_w*self.z_dim_d
        else:
            self.out_dim = np.prod(adaptive_pooling[1])

        self.coordconv_out_channels = coordconv_out_channels
        self.with_r = with_r
        self.scaling = scaling

        modules_encoder = []
        # add coordconv
        modules_encoder.append(
                        (f'CoordConv',
                        CoordConv3d(dim_x=h, dim_y=w, dim_z=d, scaling=self.scaling,
                                    in_channels=4, out_channels=self.coordconv_out_channels,
                                kernel_size=1, stride=1, padding=0, with_r=self.with_r)
                        ))

        layer_name = ['', 'a', 'b', 'c']
        for step in range(encoder_depth):
            for depth in range(block_depth-1):
                name = layer_name[depth]
                in_channels = self.coordconv_out_channels if (step == 0 and depth==0) else out_channels
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
        # adaptive pool to ensure a fixed size linear layer accross regions
        if adaptive_pooling is not None:
            if adaptive_pooling[0]=='max':
                    modules_encoder.append(('AdaptiveMaxPool', nn.AdaptiveMaxPool3d(output_size=adaptive_pooling[1])))
            elif adaptive_pooling[0]=='average':
                    modules_encoder.append(('AdaptiveAvgPool', nn.AdaptiveAvgPool3d(output_size=adaptive_pooling[1])))
            elif adaptive_pooling[0]=='generalized_mean':
                    modules_encoder.append(('GeneralizedAvgPool', GeneralizedMeanPoolingP(output_size=adaptive_pooling[1])))
            elif adaptive_pooling[0]=='generalized_mean_per_map_LNP':
                    modules_encoder.append(('GeneralizedAvgPoolPerMap_LNP', GeneralizedMeanPoolingPerMap_LNP(n_features=filters[-1],
                                                                                                     init_min=self.init_min,
                                                                                                     init_max=self.init_max,
                                                                                                     output_size=adaptive_pooling[1])))
            elif adaptive_pooling[0]=='generalized_mean_per_map_Mix':
                    modules_encoder.append(('GeneralizedAvgPoolPerMap_Mix', GeneralizedMeanPoolingPerMap_Mix(n_features=filters[-1],
                                                                                                     output_size=adaptive_pooling[1])))
            elif adaptive_pooling[0]=='generalized_mean_per_map_Atten':
                    modules_encoder.append(('GeneralizedAvgPoolPerMap_Atten', GeneralizedMeanPoolingPerMap_Atten(n_features=filters[-1],
                                                                                                                 h=self.z_dim_h,
                                                                                                                 w=self.z_dim_w,
                                                                                                                 d=self.z_dim_d,
                                                                                                     output_size=adaptive_pooling[1])))
            else:
                raise ValueError("Wrong pooling name argument")
            
        #modules_encoder.append(
        #    (f'DropOut', nn.Dropout3d(p=drop_rate))) # TODO: remove dropout from conv, keep only before the linear layer!
        # flatten and reduce to the desired dimension
        modules_encoder.append(('Flatten', nn.Flatten()))

        if linear:
            modules_encoder.append(
                ('Linear',
                nn.Linear(
                    self.num_features*self.out_dim,
                    self.num_representation_features)
                ))
            
        self.encoder = nn.Sequential(OrderedDict(modules_encoder))

        #self.flatten = nn.Flatten()

        #self.pool = nn.AdaptiveAvgPool3d(output_size=1)

        #self.fc1 = nn.Linear(self.num_features*self.out_dim, 128)

        #self.fc2 = nn.Linear(256, self.num_representation_features)

        ########
        ## TODO : design a reducer !!
        #modules_reducer = []
        #input_dim = self.num_features*self.out_dim
        #for idx, k in enumerate(range(2,-1,-1)):
        #    output_dim = 128*(4**k)
        #    modules_reducer.append(
        #        (f'Reducer_Linear{idx}',
        #        nn.Linear(
        #            input_dim, # TODO : ne marche que si on met u 
        #            output_dim)
        #        ))
        #    if idx < 2: # TODO : do not add dropout right before latent space...
        #        modules_reducer.append(
        #            (f'Reducer_DropOut{idx}', nn.Dropout(p=drop_rate)))
        #        input_dim = output_dim

        #self.reducer = nn.Sequential(OrderedDict(modules_reducer))
        #self.linear = nn.Linear(128, self.num_representation_features) # TODO : linear must be in backbone right now...

    def forward(self, x):
        out = self.encoder(x)
        #x1 = self.encoder(x)
        #x2 = self.flatten(x1)
        #x_pool = self.pool(x1)
        #x3 = self.fc1(x2)
        #x_pool1 = self.flatten(x_pool)
        #out = cat((x3, x_pool1), dim=1)
        #out = self.fc2(out)
        #out = self.reducer(out)
        #out = self.linear(out)
        return out.squeeze(dim=1)
