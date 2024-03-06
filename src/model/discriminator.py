from torch import nn
from typing import Callable


class Discriminator(nn.Module):
    """Defines a PatchGAN critic (discriminator of Wasserstein GANs)
    The PatchGAN critic is a convolutional neural network that
    operates on small patches of the input image. It is used in Conditional GANs and
    WGANs to assign a realisticness score to each patch of the input image.
    Our implementation has a receptive field of 54x54 pixels."""

    def __init__(
            self,
            ndim: int = 2,
            input_nc: int = 2,  # number of channels in input images: source+target
            activation_fn: Callable = nn.LeakyReLU,
            activation_kwargs: dict = dict(negative_slope=0.05, inplace=True),
            ndf: int = 128,  # number of filters in the last conv layer
            norm_layer: Callable = nn.InstanceNorm3d,
    ):
        n_layers = 3  # number of conv layers in the discriminator
        if ndim == 2:
            ConvNd = nn.Conv2d
            norm_layer = nn.InstanceNorm2d
        elif ndim == 3:
            ConvNd = nn.Conv3d
            norm_layer = nn.InstanceNorm3d

        super().__init__()
        kw = 4  # kernel size
        padw = 1  # padding
        sequence = [ConvNd(input_nc, ndf, kernel_size=kw, stride=2,
                           padding=padw), activation_fn(*activation_kwargs)]
        nf_mult = 1  # number of filters
        nf_mult_prev = 1  # number of filters in previous layer
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # number of filters
            sequence += [
                ConvNd(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw,
                       stride=2, padding=padw),  # bias=use_bias),
                norm_layer(ndf * nf_mult, affine=True),
                activation_fn(*activation_kwargs)
            ]

        kw = 3
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)  # number of filters
        sequence += [
            ConvNd(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw,
                   stride=1, padding=padw),  # bias=use_bias),
            norm_layer(ndf * nf_mult, affine=True),
            activation_fn(*activation_kwargs)
        ]

        # output 1 channel prediction map
        sequence += [ConvNd(ndf * nf_mult, 1, kernel_size=kw,
                            stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
