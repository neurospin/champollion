import torch
import torch.nn as nn


class AddCoords3d(nn.Module):
    """Add coords to a tensor"""
    def __init__(self, dim_x, dim_y, dim_z, scaling=1, with_r=False):
        super(AddCoords3d, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.scaling = scaling
        self.with_r = with_r

    def forward(self, input_tensor):
        
        batch_size = input_tensor.shape[0]
        x_range = torch.linspace(-1, 1, steps=self.dim_x) * self.scaling # Normalize to [-1, 1]
        y_range = torch.linspace(-1, 1, steps=self.dim_y) * self.scaling
        z_range = torch.linspace(-1, 1, steps=self.dim_z) * self.scaling

        # 3D tensors
        x_coords, y_coords, z_coords = torch.meshgrid(x_range, y_range, z_range, indexing="ij")

        # add channel dimension
        x_coords = x_coords.unsqueeze(0)
        y_coords = y_coords.unsqueeze(0)
        z_coords = z_coords.unsqueeze(0)

        # add batch dimension
        x_coords = x_coords.expand(batch_size, -1, -1, -1, -1).to(input_tensor.device)
        y_coords = y_coords.expand(batch_size, -1, -1, -1, -1).to(input_tensor.device)
        z_coords = z_coords.expand(batch_size, -1, -1, -1, -1).to(input_tensor.device)

        out = torch.cat([input_tensor, x_coords, y_coords, z_coords], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(x_coords, 2) +
                            torch.pow(y_coords, 2) +
                            torch.pow(z_coords, 2))
            out = torch.cat([out, rr], dim=1)

        return out


class CoordConv3d(nn.Module):
    def __init__(self, dim_x, dim_y, dim_z, scaling, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, with_r=False):
        super(CoordConv3d, self).__init__()
        
        self.addcoords = AddCoords3d(dim_x, dim_y, dim_z, scaling, with_r)
        self.conv = nn.Conv3d(in_channels + int(with_r), out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out