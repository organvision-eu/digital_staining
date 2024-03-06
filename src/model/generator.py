from typing import Callable

import torch
import torch.nn as nn


class UNet(nn.Module):
    """U-net architecture for 2D or 3D data."""
    def __init__(
        self,
        ndim: int = 2,
        activation_fn: Callable = nn.ReLU,
        activation_kwargs: dict = dict(inplace=True),
        dropout: float = 0,
        depth=3,
        n_in_channels=1,
        out_channels=1,
        mult_chan=64,
    ):
        """Constructor for UNet class.
        Args:
            ndim: Dimensionality of input data (2 or 3).
            activation_fn: Activation function to use.
            activation_kwargs: Additional arguments for the activation function.
            dropout: Dropout probability.
            depth: Depth of the U-net.
            n_in_channels: Number of input channels.
            out_channels: Number of output channels.
            mult_chan: Factor to determine number of output channels."""
        super().__init__()
        mult_chan = mult_chan
        self.depth = depth

        if ndim == 2:
            ConvNd = nn.Conv2d
            DropoutNd = nn.Dropout2d
        elif ndim == 3:
            ConvNd = nn.Conv3d
            DropoutNd = nn.Dropout3d

        self.net_recurse = NetRecurse(
            n_in_channels=n_in_channels,
            mult_chan=mult_chan,
            depth=depth,
            ndim=ndim,
            activation_fn=activation_fn,
            activation_kwargs=activation_kwargs,
            dropout=dropout,
        )
        self.conv_out = ConvNd(mult_chan, out_channels,
                               kernel_size=3, padding=1)

    def forward(self, x):
        x_rec = self.net_recurse(x)
        return self.conv_out(x_rec)


class NetRecurse(nn.Module):
    """Recursive definition of U-network."""
    def __init__(
        self,
        n_in_channels,
        mult_chan=2,
        depth=0,
        ndim: int = 2,
        activation_fn: Callable = nn.ReLU,
        activation_kwargs: dict = dict(inplace=True),
        dropout: float = 0,
    ):
        """Class for recursive definition of U-network.p

        Parameters
        ----------
        in_channels
            Number of channels for input.
        mult_chan
            Factor to determine number of output channels
        depth
            If 0, this subnet will only be convolutions that double the channel
            count.

        """
        super().__init__()
        self.depth = depth
        n_out_channels = n_in_channels * mult_chan
        if ndim == 2:
            ConvNd = nn.Conv2d
            InstanceNormNd = nn.InstanceNorm2d
            ConvTransposeNd = nn.ConvTranspose2d
            DropoutNd = nn.Dropout2d
        elif ndim == 3:
            ConvNd = nn.Conv3d
            InstanceNormNd = nn.InstanceNorm3d
            ConvTransposeNd = nn.ConvTranspose3d
            DropoutNd = nn.Dropout3d

        self.sub_2conv_more = SubNet2Conv(
            n_in_channels,
            n_out_channels,
            ndim=ndim,
            activation_fn=activation_fn,
            activation_kwargs=activation_kwargs,
            dropout=dropout,
        )

        if depth > 0:
            self.sub_2conv_less = SubNet2Conv(
                2 * n_out_channels,
                n_out_channels,
                ndim=ndim,
                activation_fn=activation_fn,
                activation_kwargs=activation_kwargs,
                dropout=dropout,
            )
            self.conv_down = ConvNd(
                n_out_channels, n_out_channels, kernel_size=2, stride=2)
            self.bn0 = InstanceNormNd(n_out_channels, affine=True)
            self.relu0 = activation_fn(*activation_kwargs)
            self.convt = ConvTransposeNd(
                2 * n_out_channels, n_out_channels, kernel_size=2, stride=2
            )
            self.bn1 = InstanceNormNd(n_out_channels, affine=True)
            self.relu1 = activation_fn(*activation_kwargs)
            self.sub_u = NetRecurse(
                n_out_channels,
                mult_chan=2,
                depth=(depth - 1),
                ndim=ndim,
                activation_fn=activation_fn,
                activation_kwargs=activation_kwargs,
                dropout=dropout,
            )

    def forward(self, x):
        if self.depth == 0:
            return self.sub_2conv_more(x)
        else:  # depth > 0
            x_2conv_more = self.sub_2conv_more(x)
            x_conv_down = self.conv_down(x_2conv_more)
            x_bn0 = self.bn0(x_conv_down)
            x_relu0 = self.relu0(x_bn0)
            x_sub_u = self.sub_u(x_relu0)
            x_convt = self.convt(x_sub_u)
            x_bn1 = self.bn1(x_convt)
            x_relu1 = self.relu1(x_bn1)
            x_cat = torch.cat((x_2conv_more, x_relu1), 1)  # concatenate
            x_2conv_less = self.sub_2conv_less(x_cat)
        return x_2conv_less


class SubNet2Conv(nn.Module):
    """Subnetwork with two convolutional layers."""
    def __init__(
        self,
        n_in,
        n_out,
        ndim: int = 2,
        activation_fn: Callable = nn.ReLU,
        activation_kwargs: dict = dict(inplace=True),
        dropout: float = 0,
    ):
        """Constructor for SubNet2Conv class.
        Args:
            n_in: Number of input channels.
            n_out: Number of output channels.
            ndim: Dimensionality of input data (2 or 3).
            activation_fn: Activation function to use.
            activation_kwargs: Additional arguments for the activation function.
            dropout: Dropout probability."""
        super().__init__()

        if ndim == 2:
            ConvNd = nn.Conv2d
            InstanceNormNd = nn.InstanceNorm2d
            DropoutNd = nn.Dropout2d
        elif ndim == 3:
            ConvNd = nn.Conv3d
            InstanceNormNd = nn.InstanceNorm3d
            DropoutNd = nn.Dropout3d

        self.conv1 = ConvNd(n_in, n_out, kernel_size=3, padding=1)
        self.bn1 = InstanceNormNd(n_out, affine=True)
        self.relu1 = activation_fn(*activation_kwargs)
        self.conv2 = ConvNd(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = InstanceNormNd(n_out, affine=True)
        self.relu2 = activation_fn(*activation_kwargs)
        self.dropout_layer = DropoutNd(p=dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        if self.dropout_layer.p > 0:
            x = self.dropout_layer(x)
        return x
