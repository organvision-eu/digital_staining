import numpy as np
import torch.nn.functional as F
import torch


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_device(gpuid):
    if not torch.cuda.is_available():
        raise NotImplementedError("We only support CUDA devices at the moment")
    torch.cuda.set_device(gpuid)
    return torch.device(f'cuda:{gpuid}')


def normalize(t: torch.Tensor, norm: str = 'Lp', p: float = 2, mu: np.ndarray | None = None, std: np.ndarray | None = None):
    # normalize reducing each spatial dimension (ie, normalize per channel: dim 0 is channels)
    # dim = list(range(1, t.ndim + 1))
    dim = list(range(1, t.ndim))

    if norm == 'standard':
        if mu is None or std is None:
            # normalize to mean 0, std 1
            mu = torch.mean(t, dim, keepdim=True)
            std = torch.std(t, dim, keepdim=True)
        else:
            if len(dim) > 1:
                # py3.11
                # extra_dims = [None] * len(dim)
                # mu = mu[:, *extra_dims]
                # std = std[:, *extra_dims]
                mu = np.expand_dims(mu, axis=dim)
                std = np.expand_dims(std, axis=dim)
            else:
                # squeeze
                mu = mu[0]
                std = std[0]
        return (t - mu) / std

    elif norm == 'minmax_global':
        # normalize to [0, 1]
        min = torch.min(t)
        max = torch.max(t)
        return (t - min) / (max - min)

    elif norm == 'minmax':
        # normalize to [0, 1] per channel
        min_d1 = torch.min(t, dim=1, keepdim=True)[0]
        min_d2 = torch.min(min_d1, dim=2, keepdim=True)[0]
        min_d3 = torch.min(min_d2, dim=3, keepdim=True)[0]

        max_d1 = torch.max(t, dim=1, keepdim=True)[0]
        max_d2 = torch.max(max_d1, dim=2, keepdim=True)[0]
        max_d3 = torch.max(max_d2, dim=3, keepdim=True)[0]
        # 100 is to avoid numerical issues
        return 100*(t - min_d3) / (max_d3 - min_d3)

    # normalize using Lp norm
    return F.normalize(t, p=p, dim=dim)


@torch.compile
def labeled_prediction(generator, source, target, all_channels=False, inference=False, prediction=None, skip_channel=None):
    '''Extracts the images and label predictions from the generated images.
    Args:
        generator: torch.nn.Module, generator model
        source: tensor, shape (batch_size, 1, z, x, y)
        target: tensor, shape (batch_size, n_channels, z, x, y)
        all_channels: bool, if True, uses all the channels in the target tensor without applying the NaN mask
        inference: bool, if True, returns the images and labels without shuffling
        prediction: tensor, shape (batch_size, n_channels, z, x, y), if None, uses the generator to predict the images
        skip_channel: int, if not None, skips channels in the target tensor that are after than the specified value
    Returns:
        images: tensor, shape (1, batch_size*n_channels, z, x, y)
        labels: tensor, shape (batch_size*n_channels)'''
    device = next(generator.parameters()).device
    images = []
    labels = []
    if prediction is None:
        generator.eval()
        prediction = generator(source.to(device))
    for i, fov in enumerate(prediction):
        for ch in range(fov.shape[0]):
            if all_channels:
                nan_filter = fov[ch]
            else:
                nan_filter = target[i][ch]
            if not nan_filter.isnan().any():
                images.append(fov[ch])
                if skip_channel is not None:  # Classifier is trained on 6 channels, but we are generating less than 6
                    if ch >= skip_channel:
                        labels.append(ch+1)
                    else:
                        labels.append(ch)
                else:  # skip_channel is None
                    labels.append(ch)

    if inference:
        return torch.unsqueeze(torch.stack(images), dim=1).to(device), torch.tensor(labels, device=device)

    indices = torch.randperm(len(images))

    return (torch.unsqueeze(torch.stack(images), dim=1)[indices]).to(device), torch.tensor(labels[indices], device=device)


@torch.compile
def labeled_from_target(target):
    '''Extracts the images and labels from the target tensor.
    Args:
        target: tensor, shape (batch_size, n_channels, z, x, y)
    Returns:
        images: tensor, shape (1, batch_size*n_channels, z, x, y)
        labels: tensor, shape (batch_size*n_channels)'''
    device = target.device
    images = []
    labels = []
    for fov in target:
        for ch in range(fov.shape[0]):
            if not fov[ch].isnan().any():
                images.append(fov[ch])
                labels.append(ch)
    indices = torch.randperm(len(images), device=device)  # shuffle the images

    return (torch.unsqueeze(torch.stack(images).to(device)[indices], dim=1),
            torch.tensor(labels, device=device)[indices])
