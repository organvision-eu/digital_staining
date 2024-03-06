import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

from .metrics import corrcoef, PSNR


def display_batch(source, target, pred=None, target_channel_names=None,
                  show3d=None, limit_images=None, show_metrics=False, mask=None, batch=0):
    '''Display a batch of predictions with the source, target and prediction images.
    Args:
        source: tensor, shape (batch_size, 1, z, x, y)
        target: tensor, shape (batch_size, n_channels, z, x, y)
        pred: tensor, shape (batch_size, n_channels, z, x, y)
        target_channel_names: list, the names of the channels
        show3d: str, the method to display the 3D images
        limit_images: int, the maximum number of images to display
        show_metrics: bool, if True, display the metrics between the target and the prediction
        mask: tensor, shape (batch_size, n_channels, z, x, y), the mask to apply to the prediction
        batch: int, the batch number
    Returns:
        fig: matplotlib.figure.Figure, the figure with the images'''
    source = source.cpu()
    target = target.cpu()

    if batch == 0:
        first_batch = True
    else:
        first_batch = False

    if pred is not None:
        pred = pred.cpu().float()
        if mask is not None:
            pred = pred * mask.cpu()

    if limit_images is not None and source.shape[0] > limit_images:
        source = source[:limit_images, ...]
        target = target[:limit_images, ...]
        if pred is not None:
            pred = pred[:limit_images, ...]

    def remove_axis(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        plt.setp(ax.spines.values(), color=None)

    num_chs = target.shape[1]
    mult = 2 if pred is not None else 1

    if target_channel_names is not None:
        assert len(target_channel_names) == num_chs

    if source.ndim == 5:  # 3-D, we have a Z dim (+ batch, channel, *, Y, X)
        if show3d is None:
            show3d = 'max'

        if show3d in ('average', 'mean'):
            source = torch.mean(source, 2)
            target = torch.mean(target, 2)
            if pred is not None:
                pred = torch.mean(pred, 2)
        elif show3d == 'max':
            source = torch.max(source, 2).values
            target = torch.max(target, 2).values
            if pred is not None:
                pred = torch.max(pred, 2).values
        elif show3d == 'middle':
            mid = source.shape[2] // 2
            source = source[:, :, mid, ...]
            target = target[:, :, mid, ...]
            if pred is not None:
                pred = pred[:, :, mid, ...]

    num_axes = (1 + mult*num_chs, source.shape[0])

    figsize = np.roll(np.asarray(num_axes)*2, 1)

    fig, axs = plt.subplots(*num_axes, figsize=figsize)
    if axs.ndim == 1:
        axs = np.expand_dims(axs, 1)

    for i in range(source.shape[0]):
        ax = axs[:, i]
        # if there are NaNs, print their locations
        if torch.isnan(source[i, 0, ...]).any():
            print(f'NaNs in source[{i}]')
        scalebar = ScaleBar(0.108, 'um', frameon=False,
                            color='white', location='lower right')
        ax[0].imshow(source[i, 0, ...], cmap='gray')
        ax[0].add_artist(scalebar)
        if first_batch and i == 0:
            ax[0].set_ylabel('source')
        remove_axis(ax[0])

        for j in range(num_chs):
            if target_channel_names is not None:
                ch_name = f'\n({target_channel_names[j]})'
            else:
                ch_name = ''

            ax_i = 1 + j*mult
            ax[ax_i].imshow(target[i, j, ...], cmap='gray')
            if first_batch and i == 0:
                ax[ax_i].set_ylabel('target' + ch_name)
            remove_axis(ax[ax_i])

            if not torch.all(pred[i, j, ...] == 0):
                ax[ax_i + 1].imshow(pred[i, j, ...], cmap='gray')
                if first_batch and i == 0:
                    ax[ax_i + 1].set_ylabel('prediction' + ch_name)
            remove_axis(ax[ax_i + 1])

            # compute metrics between target and prediction
            if show_metrics and pred is not None:
                L1 = torch.nn.L1Loss()(pred[i, j, ...], target[i, j, ...])
                L2 = torch.nn.MSELoss()(pred[i, j, ...], target[i, j, ...])
                corr = corrcoef(pred[i, j, ...], target[i, j, ...])
                psnr = PSNR(pred[i, j, ...], target[i, j, ...])
                # add metrics to plot
                ax[ax_i + 1].set_xlabel(
                    f'L1: {L1:.3f}\nL2: {L2:.3f}\nCorr: {corr:.3f}\nPSNR: {psnr:.3f}')

    fig.tight_layout()

    return fig


def display_error_map(source, target, pred, target_channel_names=None, show3d='middle'):
    '''Display the error map between the target and the prediction.
    Args:
        source: tensor, shape (batch_size, 1, z, x, y)
        target: tensor, shape (batch_size, n_channels, z, x, y)
        pred: tensor, shape (batch_size, n_channels, z, x, y)
        target_channel_names: list, the names of the channels
        show3d: str, the method to display the 3D images
    Returns:
        fig: matplotlib.figure.Figure, the figure with the images'''

    if target.shape[0] > 1:
        raise ValueError('Only one image can be displayed at a time.')

    source = source.cpu()
    target = target.cpu()
    pred = pred.cpu().float()

    def remove_axis(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        plt.setp(ax.spines.values(), color=None)

    num_chs = target.shape[1]

    if target_channel_names is not None:
        assert len(target_channel_names) == num_chs

    if source.ndim == 5:  # 3-D, we have a Z dim (+ batch, channel, *, Y, X)

        if show3d == 'middle':
            mid = source.shape[2] // 2
            source = source[:, :, mid, ...]
            target = target[:, :, mid, ...]
            if pred is not None:
                pred = pred[:, :, mid, ...]
        elif show3d in ('average', 'mean'):
            source = torch.mean(source, 2)
            target = torch.mean(target, 2)
            if pred is not None:
                pred = torch.mean(pred, 2)
        elif show3d == 'max':
            source = torch.max(source, 2).values
            target = torch.max(target, 2).values
            if pred is not None:
                pred = torch.max(pred, 2).values

    num_axes = (3, 3)

    figsize = np.roll(np.asarray(num_axes)*2, 1)*2

    fig, axs = plt.subplots(*num_axes, figsize=figsize)
    if axs.ndim == 1:
        axs = np.expand_dims(axs, 1)

    channel_idx = 0
    for j in range(num_chs):
        if target_channel_names is not None:
            ch_name = f'\n({target_channel_names[j]})'
        else:
            ch_name = ''

        if not torch.isnan(target[0, j, ...]).any():
            axs[channel_idx, 0].imshow(
                target[0, j, ...], cmap='gray')  # target
            axs[channel_idx, 0].set_xlabel('target' + ch_name)
            axs[channel_idx, 1].imshow(
                pred[0, j, ...], cmap='gray')  # prediction
            axs[channel_idx, 1].set_xlabel('prediction' + ch_name)
            error_map = torch.abs(pred[0, j, ...] - target[0, j, ...])
            axs[channel_idx, 2].imshow(error_map, cmap='gray')  # error map
            total_error = torch.sum(error_map)
            axs[channel_idx, 2].set_xlabel(
                'error' + ch_name + f'\nTotal error: {total_error:.3f}')
            remove_axis(axs[channel_idx, 0])
            remove_axis(axs[channel_idx, 1])
            remove_axis(axs[channel_idx, 2])
            channel_idx += 1

    fig.tight_layout()

    return fig


def hstack_figures(images):
    '''Stack images horizontally.
    Args:
        images: list, the images to stack
    Returns:
        new_im: PIL.Image, the stacked images'''
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im
