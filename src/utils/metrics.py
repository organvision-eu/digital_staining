from sklearn.metrics import f1_score
import torch
import math
import numpy as np

from .utils import labeled_prediction


@torch.compile(mode="reduce-overhead")  # , dynamic=True)
def corrcoef(img, target, reduction_mode='mean'):
    """Compute the Pearson correlation coefficient between two images"""
    dim = list(range(1, img.ndim))
    img = img - img.mean(dim=dim, keepdim=True)
    img = img / img.norm(2, dim=dim, keepdim=True)
    target = target - target.mean(dim=dim, keepdim=True)
    target = target / target.norm(2, dim=dim, keepdim=True)
    res = (img * target).sum(dim=dim)
    if reduction_mode == 'mean':
        return res.mean()
    return res.sum()


@torch.compile(mode="reduce-overhead")  # , dynamic=True)
def corrcoef_channel(img, target, ch=0):
    """Compute the Pearson correlation coefficient between two images for a given channel"""

    img = img[:, ch, ...]
    target = target[:, ch, ...]

    img = img - img.mean()
    target = target - target.mean()
    img = img / img.norm()
    target = target / target.norm()
    return (img * target).sum()


@torch.compile(mode="reduce-overhead")  # , dynamic=True)
def PSNR(im1, im2):
    '''Compute the Peak Signal to Noise Ratio between two images'''
    im1 = im1.detach().cpu().float().numpy().astype(np.float64) / 255
    im2 = im2.detach().cpu().float().numpy().astype(float) / 255
    mse = np.mean((im1 - im2)**2)
    return 10*math.log10(1. / mse)


def compute_batch_metrics(target, pred):
    '''Compute the L1, L2, correlation and PSNR between the target and the prediction, given the prediction and the target
    Args:
        target: tensor, shape (batch_size, n_channels, z, x, y)
        pred: tensor, shape (batch_size, n_channels, z, x, y)
    Returns:
        L1: float, L1 loss
        L2: float, L2 loss
        corr: float, Pearson correlation coefficient
        psnr: float, Peak Signal to Noise Ratio'''
    target = target.cpu()
    target = torch.nan_to_num(target, nan=0.0)

    pred = pred.cpu().float()
    mask = ~(target.isnan())
    pred = pred * mask

    for i in range(target.shape[0]):
        for ch in range(target.shape[1]):
            if target[i, ch, ...].max() == 0:
                continue
            L1 = torch.nn.L1Loss()(pred[i, ch, ...], target[i, ch, ...])
            L2 = torch.nn.MSELoss()(pred[i, ch, ...], target[i, ch, ...])
            corr = corrcoef(pred[i, ch, ...], target[i, ch, ...])
            psnr = PSNR(pred[i, ch, ...], target[i, ch, ...])
    return L1.item(), L2.item(), corr.item(), psnr.item()


def compute_metrics(loader, model):
    '''Compute the L1, L2, correlation and PSNR between the target and the prediction of a given model
    Args:
        loader: torch.utils.data.DataLoader, the data loader
        model: torch.nn.Module, the model
    Returns:
        L1: float, L1 loss
        L2: float, L2 loss
        corr: float, Pearson correlation coefficient
        psnr: float, Peak Signal to Noise Ratio'''
    L1s = []
    L2s = []
    corrs = []
    psnrs = []

    for batch in loader:
        source, target = batch
        pred = model(source)
        L1, L2, corr, psnr = compute_batch_metrics(target, pred)
        L1s.append(L1)
        L2s.append(L2)
        corrs.append(corr)
        psnrs.append(psnr)

    L1 = torch.stack(L1s).mean()
    L2 = torch.stack(L2s).mean()
    corr = torch.stack(corrs).mean()
    psnr = torch.stack(psnrs).mean()

    return L1.item(), L2.item(), corr.item(), psnr.item()


@torch.compile
def classification_metrics(generator_model, classifier, batch, prediction=None, skip_channel=None):
    """Compute the F1 score for the classifier on the fake images generated by the generator.
    Args:
        generator_model: torch.nn.Module, the generator model
        classifier: torch.nn.Module, the classifier model
        batch: tuple, the batch of real images and labels
        prediction: torch.nn.Module, the prediction of the generator if we want to use a precomputed prediction
        skip_channel: int, the channel to skip
    """
    y_true_fake = []
    y_pred_fake = []
    with torch.no_grad():
        images, labels = labeled_prediction(
            generator_model, batch[0], batch[1], inference=True, prediction=prediction, skip_channel=skip_channel)
        outputs = classifier(images).argmax(dim=1)
        y_true_fake.extend(labels.tolist())
        y_pred_fake.extend(outputs.tolist())
    # f1_per_class =  f1_score(y_true_fake, y_pred_fake, average=None)
    f1_macro = f1_score(y_true_fake, y_pred_fake, average='macro')
    f1_micro = f1_score(y_true_fake, y_pred_fake, average='micro')
    f1_weighted = f1_score(y_true_fake, y_pred_fake, average='weighted')

    return f1_macro, f1_micro, f1_weighted,  # f1_per_class
