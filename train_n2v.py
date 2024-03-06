import numpy as np
import zarr
from skimage import img_as_float32
import glob
import argparse

from n2v.models import N2VConfig, N2V
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator

from .src.data.aicszarr import MAP_NAME_STRUCTURE
from .src.utils.parsers import add_train_n2v_arguments


def load_images(path, channel):
    image_list = []
    images = glob.glob(path)

    for index in range(3):  # len(images)):
        image = zarr.open(images[index])
        # Pick the appropriate channels
        image = image[channel:channel+1, :, :, :]
        # for channel 1 take only the slices where max intensity is > 0
        filter_z = np.where(np.max(image, axis=(0, 2, 3)) > 0)
        # print(filter_z)
        image = image[:, slice(filter_z[0][0], filter_z[0][-1]), :, :]
        image = np.moveaxis(image, 0, 3)  # Swap axes to the right format ZYXC
        image = img_as_float32(image)  # Convert to float32
        # Expand dims to get it ready for neural network
        image = np.expand_dims(image, axis=0)
        image_list.append(image)  # add images to our list

    return image_list


def train_n2v(struct_dir, structure_name, channel=1):

    imgs = load_images(str(struct_dir), channel=channel)

    datagen = N2V_DataGenerator()
    patch_size = 64
    patch_size_z = 8
    patch_shape = (patch_size_z, patch_size, patch_size)
    patches = datagen.generate_patches_from_list(imgs, shape=patch_shape)

    train_val_split = int(patches.shape[0] * 0.8)
    X = patches[:train_val_split]
    X_val = patches[train_val_split:]

    # train_steps_per_epoch is set to (number of training patches)/(batch size), 
    # like this each training patch is shown once per epoch.
    train_batch = 16
    config = N2VConfig(X, unet_kern_size=3,
                       unet_n_first=64, unet_n_depth=3,
                       train_steps_per_epoch=int(len(X)/train_batch), train_epochs=40, train_loss='mse',
                       batch_norm=True, train_batch_size=train_batch, n2v_perc_pix=0.198, n2v_patch_shape=(patch_size_z, patch_size, patch_size),
                       n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5, single_net_per_channel=False)

    model_name = f'n2v_3D_{structure_name}'
    # the base directory in which our model will be saved
    basedir = 'n2v_models'
    # We are now creating our network model.
    model = N2V(config, model_name, basedir=basedir)

    model.train(X, X_val)

    return None


def main():

    parser = argparse.ArgumentParser(description="Train N2V model")
    add_train_n2v_arguments(parser)
    args = parser.parse_args()

    DATASET_DIR = args.dataset
    structures_of_interest = args.structures_of_interest
    DNA = args.DNA
    cell_membrane = args.cm

    if DNA:
        struct_dir = DATASET_DIR / f"{structures_of_interest[0]}/*.zarr"
        train_n2v(struct_dir, "DNA", channel=2)

    if cell_membrane:
        struct_dir = DATASET_DIR / f"{structures_of_interest[0]}/*.zarr"
        train_n2v(struct_dir, "cell_membrane", channel=0)

    for structure in structures_of_interest:
        struct_dir = DATASET_DIR / f"{structure}/*.zarr"
        structure_name = MAP_NAME_STRUCTURE[structure]
        train_n2v(struct_dir, structure_name, channel=1)


if __name__ == "__main__":
    main()
