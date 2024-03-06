import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchmetrics

from ..utils.utils import labeled_from_target, get_device

# batch_size = 2
# num_available_channels = 3
# input_shape = (batch_size*num_available_channels, 1, 16, 512, 512)


class ResidualBlock(nn.Module):
    """Residual Block with 3D convolutions."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet3D(nn.Module):
    """ResNet3D with 3D convolutions."""

    def __init__(self, block, layers, num_classes=6):
        """Constructor for ResNet3D class.
        Args:
            block: Residual block to use.
            layers: List of number of residual blocks in each layer.
            num_classes: Number of classes for the classification task."""
        super(ResNet3D, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv3d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        strides = [stride] + [1]*(blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class Classifier(L.LightningModule):
    """Classifier class for ResNet3D model.
    Args:
        model: ResNet3D model.
        class_sizes: List of number of samples in each class.
        target_channels: List of target channels for the classification task.
        learning_rate: Learning rate for the optimizer. Default: 0.01."""

    def __init__(self, model, class_sizes, target_channels, learning_rate=0.01):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss(weight=1/class_sizes)
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=len(target_channels))
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _, y = batch
        images, labels = labeled_from_target(y)

        outputs = self.model(images)
        loss = self.loss(outputs, labels)
        accuracy = self.accuracy(outputs, labels)
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy, on_epoch=True,
                 on_step=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        _, y = batch
        images, labels = labeled_from_target(y)

        with torch.no_grad():
            outputs = self.model(images)
        loss = self.loss(outputs, labels)
        accuracy = self.accuracy(outputs, labels)
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def test_step(self, batch, batch_idx):
        _, y = batch
        images, labels = labeled_from_target(y)

        with torch.no_grad():
            outputs = self.model(images)
        loss = self.loss(outputs, labels)
        accuracy = self.accuracy(outputs, labels)
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy, on_epoch=True, prog_bar=True)

        return accuracy
