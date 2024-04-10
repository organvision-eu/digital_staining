import argparse
import torch
from torch.utils.data import DataLoader
import lightning as L
import numpy as np
from sklearn.model_selection import train_test_split
from lightning.pytorch.strategies import DDPStrategy
import os

from src.data.aicszarr import AICSZarrPatchExpandedDataset, read_metadata_csv, src_channel, MAP_NAME_STRUCTURE
from src.model import WGANGP, UNet, Discriminator, ResNet3D, ResidualBlock, Classifier
from src.utils.utils import get_device
from src.utils.parsers import add_training_parser_argument


def load_classifier(ckp_path, df_metadata, target_channels, device):
    try:
        checkpoint = torch.load(ckp_path, map_location=device)
    except:
        return ValueError(f"Model file {ckp_path} not found")
    resnet3D = ResNet3D(
        ResidualBlock, [2, 2, 2, 2], num_classes=len(target_channels))
    resnet3D = resnet3D.to(device)

    class_sizes_dict = dict(df_metadata["specific_structure"].value_counts())
    class_sizes_dict["DNA"] = len(df_metadata)
    class_sizes_dict["cell_membrane"] = len(df_metadata)

    class_sizes = []
    for ch in target_channels:
        class_sizes.append(class_sizes_dict[ch])
    class_sizes = torch.tensor(class_sizes)

    resnet_model = Classifier(resnet3D, class_sizes, target_channels)
    model_state_dict = checkpoint['state_dict']
    resnet_model.load_state_dict(model_state_dict)
    resnet_model = torch.compile(resnet_model)

    return resnet_model


def main():

    parser = argparse.ArgumentParser(description="Train the model")
    add_training_parser_argument(parser)
    args = parser.parse_args()

    # for reproducibility
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    gpuid = args.gpuid
    device = get_device(gpuid)
    accelerator = 'gpu' if device.type == 'cuda' else 'cpu'

    # set the device
    torch.cuda.set_device(device)

    # limit available vmemory
    torch.cuda.set_per_process_memory_fraction(
        args.per_process_memory_fraction)

    torch.set_float32_matmul_precision('medium')

    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = args.benchmark

    DATASET_DIR = args.dataset
    structures_of_interest = args.structures_of_interest
    target_channels = args.target_channels

    checkpoint_path = args.checkpoint

    train_batch_size = args.batch_size_training  # 16
    train_patch_shape = args.patch_shape_training  # (16, 128, 128)
    train_patch_strides = args.patch_stride_training  # (8, 64 , 64)
    batch_size = args.batch_size  # 2
    patch_shape = args.patch_shape  # (16, 384, 384)
    patch_strides = args.patch_stride  # (16, 384, 384)
    ignore_incomplete_patches = True

    z_range = args.z_range  # in-focus-centre, (in-focus-hint,2)

    use_classification_metric = args.classification_metric  # True
    classifier_path = args.classifier  # Path to the classifier
    adversarial_training = args.adversarial_training  # True
    epochs = args.epochs  # 10
    lr_g = args.learning_rate_generator  # 0.00005
    lr_d = args.learning_rate_critic  # 0.00005 #discriminator
    negative_slope_g = args.negative_slope_generator  # 0.05
    negative_slope_c = args.negative_slope_critic  # 0.05

    num_workers = args.num_workers
    if num_workers > 0:
        prefetch_factor = args.prefetch_factor
    else:
        prefetch_factor = None
        
    ddp = args.ddp

    print(args)

    persistent_workers = False
    compute_pooled_stats = False
    use_normalization = 'standard'
    dtype = np.float32

    specific_structures_names = [MAP_NAME_STRUCTURE[s]
                                 for s in structures_of_interest]

    df_metadata, channels_pooled_stats, target_channels = read_metadata_csv(
        DATASET_DIR=DATASET_DIR,
        src_types=src_channel, target_channels=target_channels, magnifications=120,
        compute_pooled_stats=compute_pooled_stats, specific_structures=specific_structures_names,
        unify_channels=True,
    )

    # stratified train/val/test split
    train_ds_meta, val_ds_meta = train_test_split(
        df_metadata, test_size=0.2, random_state=seed, stratify=df_metadata["specific_structure"])

    # Prepare datasets and dataloaders
    train_ds = AICSZarrPatchExpandedDataset(
        train_ds_meta.copy(),
        root_dir=DATASET_DIR,
        signal_channel='src',
        target_channels=target_channels,
        patch_shape=train_patch_shape,
        patch_strides=train_patch_strides,
        ignore_incomplete_patches=ignore_incomplete_patches,
        z_range=z_range,
        dtype=dtype,
        use_normalization=use_normalization,
        normalization_stats=channels_pooled_stats,  # normalize using global mu&sigma
        random_seed=seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    if use_classification_metric:
        # prepare dataset for classification metric
        classification_ds = AICSZarrPatchExpandedDataset(
            # dataset used for the classification metric
            val_ds_meta.copy().groupby('specific_structure', group_keys=False).apply(
                lambda x: x.sample(frac=0.5)),  # TODO: solve deprecation warning: Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
            root_dir=DATASET_DIR,
            signal_channel='src',
            target_channels=target_channels,
            patch_shape=patch_shape,
            patch_strides=patch_strides,
            ignore_incomplete_patches=ignore_incomplete_patches,
            z_range=z_range,
            dtype=dtype,
            use_normalization=use_normalization,
            normalization_stats=channels_pooled_stats,  # normalize using global mu&sigma
            random_seed=seed,
        )

        classification_loader = DataLoader(
            classification_ds,
            batch_size=batch_size,
            shuffle=False,
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
        z_range="in-focus-centre",
        dtype=dtype,
        use_normalization=use_normalization,
        normalization_stats=channels_pooled_stats,  # normalize using global mu&sigma
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

    if use_classification_metric:
        resnet_model = load_classifier(
            classifier_path, df_metadata, target_channels, device)

    wgan_config = {
        "use_classification_metric": use_classification_metric,
        "classifier": resnet_model if use_classification_metric else None,
        "target_channels": target_channels,
        "classification_loader": classification_loader if use_classification_metric else None,
        "len_val_loader": len(val_loader),
        "adversarial_training": adversarial_training,
        "lr_g": lr_g,
        "lr_d": lr_d
    }

    # define generator and critic
    activation_fn = torch.nn.LeakyReLU
    activation_kwargs_g = (negative_slope_g, True)
    activation_kwargs_c = (negative_slope_c, True)
    ndim = 3
    depth = 3
    mult_chan = 64

    Generator = UNet(
        ndim=ndim,
        activation_fn=activation_fn,
        activation_kwargs=activation_kwargs_g,
        depth=depth,
        # dropout=dropout, #
        n_in_channels=1,
        out_channels=len(target_channels),
        mult_chan=mult_chan,
    )

    Critic = Discriminator(
        ndim=ndim,
        input_nc=len(target_channels)+1,
        activation_fn=activation_fn,
        activation_kwargs=activation_kwargs_c,
        ndf=64,
    ).to(device)

    wmodel = WGANGP(Generator, Critic, wgan_config)

    version = f"{z_range}_{str(wmodel.__class__).split('.')[-1][:-2]}_advT_{adversarial_training}_{ndim}D_depth:{depth}_{lr_g}lrG_{lr_d}lrD_{structures_of_interest}"
    loggers = L.pytorch.loggers.TensorBoardLogger('.', version=version)
    trainer = L.Trainer(max_epochs=epochs, precision="bf16-mixed", logger=loggers, num_sanity_val_steps=0, accelerator=accelerator, 
                        devices= 'auto' if ddp else [gpuid],
                        strategy=DDPStrategy(find_unused_parameters=True) if ddp else "auto",
                        )

    if checkpoint_path:
        trainer.fit(model=wmodel, train_dataloaders=train_loader, val_dataloaders=val_loader,
                    ckpt_path=checkpoint_path
                    )
    else:
        trainer.fit(model=wmodel, train_dataloaders=train_loader, val_dataloaders=val_loader,
                    )

    if not os.path.exists("trained_models"):
        os.makedirs("trained_models")

    torch.save({"gen_state_dict": Generator.state_dict(),
                "target_channels": target_channels,
                "gen_hyperparams": {
                    "ndim": ndim,
                    "depth": depth,
                    "mult_chan": mult_chan,
                    "lr_g": lr_g, },
                "adversarial_training": adversarial_training,
                }, f"trained_models/Gen_{version}.tar")
    print(f"Generator model saved as Gen_{version}.tar")


if __name__ == "__main__":
    main()
