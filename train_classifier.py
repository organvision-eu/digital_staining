import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import pathlib
from sklearn.model_selection import train_test_split
import argparse

from src.data.aicszarr import AICSZarrPatchExpandedDataset, read_metadata_csv, TARGET_CHANNELS, src_channel, MAP_NAME_STRUCTURE
from src.model import ResNet3D, ResidualBlock, Classifier
from src.utils.utils import get_device
from src.utils.parsers import add_train_classifier_arguments


def main():

    parser = argparse.ArgumentParser(
        description="Train the classifier"
    )
    add_train_classifier_arguments(parser)

    args = parser.parse_args()

    DATASET_DIR = args.dataset
    structures_of_interest = args.structures_of_interest

    batch_size = args.batch_size  # 2
    patch_shape = args.patch_shape  # (16, 384, 384)
    patch_strides = args.patch_stride  # (16, 384, 384)
    ignore_incomplete_patches = True
    z_range = args.z_range  # in-focus-centre, (in-focus-hint,2)
    epochs = args.epochs  # 10
    learning_rate = args.learning_rate  # 0.01
    num_workers = args.num_workers
    prefetch_factor = args.prefetch_factor
    version = args.version
    seed = args.seed
    gpuid = args.gpuid

    # Device configuration
    device = get_device(gpuid)

    torch.set_float32_matmul_precision('medium')

    # for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = args.deterministic  # True
    # set True to speed up constant input size inference
    torch.backends.cudnn.benchmark = args.benchmark  # True

    persistent_workers = False
    compute_pooled_stats = False
    use_normalization = 'standard'
    dtype = np.float32

    DATASET_DIR = pathlib.Path(DATASET_DIR)

    specific_structures_names = [MAP_NAME_STRUCTURE[s]
                                 for s in structures_of_interest]

    df_metadata, channels_pooled_stats, target_channels = read_metadata_csv(
        DATASET_DIR = DATASET_DIR,
        src_types=src_channel, target_channels=TARGET_CHANNELS, magnifications=120,
        compute_pooled_stats=compute_pooled_stats, specific_structures=specific_structures_names,
        unify_channels=True
    )

    train_ds_meta, val_ds_meta = train_test_split(
        df_metadata, test_size=0.2, random_state=seed, stratify=df_metadata["specific_structure"])

    train_ds = AICSZarrPatchExpandedDataset(
        train_ds_meta.copy(),
        root_dir=DATASET_DIR,
        signal_channel='src',
        target_channels=target_channels,
        patch_shape=patch_shape,
        patch_strides=patch_strides,
        ignore_incomplete_patches=ignore_incomplete_patches,
        z_range=z_range,
        dtype=dtype,
        use_normalization=use_normalization,
        normalization_stats=channels_pooled_stats,
        random_seed=seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    val_ds = AICSZarrPatchExpandedDataset(
        val_ds_meta.copy(),
        root_dir=DATASET_DIR,
        signal_channel='src',
        target_channels=target_channels,
        patch_shape=patch_shape,
        patch_strides=patch_strides,
        ignore_incomplete_patches=ignore_incomplete_patches,
        z_range=z_range,
        dtype=dtype,
        use_normalization=use_normalization,
        normalization_stats=channels_pooled_stats,
        shuffle=False,
        random_seed=seed,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True,
    )

    num_classes = len(target_channels)
    # Model
    resnet3D = ResNet3D(ResidualBlock, [2, 2, 2, 2], num_classes=num_classes)
    resnet3D = resnet3D.to(device)

    # hyperparameters
    class_sizes_dict = dict(df_metadata["specific_structure"].value_counts())
    class_sizes_dict["DNA"] = len(df_metadata)
    class_sizes_dict["cell_membrane"] = len(df_metadata)
    class_sizes = []
    for ch in target_channels:
        class_sizes.append(class_sizes_dict[ch])
    class_sizes = torch.tensor(class_sizes)

    # Classifier
    resnet_model = Classifier(resnet3D, class_sizes,
                              target_channels, learning_rate)

    # training
    version = f'3Dresnet18_weighted_CELoss_unified_channels{version}'
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    loggers = L.pytorch.loggers.TensorBoardLogger(
        'classifier', version=version, )
    trainer = L.Trainer(max_epochs=epochs, logger=loggers, precision="bf16-mixed",
                        callbacks=[EarlyStopping(monitor="val_loss", patience=3), checkpoint_callback]) 
    #TODO: accelerator, gpu, devices, strategy

    trainer.fit(model=resnet_model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
