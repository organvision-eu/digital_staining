import torch
from torch import nn
import torch.nn.functional as F

# option for perception loss - not used


class GenLoss(nn.Module):
    def __init__(self, loss='l1'):
        super(GenLoss, self).__init__()
        if loss == 'l1' or loss == 'mae':
            self.pw_loss = nn.L1Loss(reduction='none')  # pixel wise loss
        elif loss == 'l2' or loss == 'mse':
            self.pw_loss = nn.MSELoss(reduction='none')  # pixel wise loss

        else:
            raise NotImplementedError('Loss [%s] is not implemented' % loss)

    def forward(self, out_scores, out_images, target_images, epoch=0, mask=None, adv_weight=0.01):
        # Adversarial Loss
        # pushes the generator to produce images with a more positive out_score as possible
        adversarial_loss = -torch.mean(out_scores)

        # Perception Loss
        # perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))

        # Image Loss
        image_loss = self.pw_loss(out_images, target_images)

        if mask is not None:
            image_loss = image_loss[mask]

        # /sqrt(epoch+1) #+ 0.006 * perception_loss
        combined_loss = image_loss.mean() + adv_weight * adversarial_loss
        return combined_loss, image_loss.mean(), adversarial_loss,


class UNetLoss(nn.Module):
    def __init__(self):
        super(UNetLoss, self).__init__()
        self.mae_loss = nn.L1Loss(reduce=False)

    def forward(self, out_images, target_images, epoch):
        image_loss = self.mae_loss(out_images, target_images)
        combined_loss = image_loss.mean()
        return combined_loss
