from pathlib import Path
import os
import zarr
from tqdm.auto import tqdm
import shutil
import pandas as pd
import numpy as np
from skimage import img_as_float32, img_as_uint
from n2v.models import N2V
import argparse

from src.data.aicszarr import MAP_NAME_STRUCTURE


def parse_args():
    parser = argparse.ArgumentParser(description='Denoise dataset')

    parser.add_argument(
        '--src_dir',
        type=Path,
        help='Path to the source directory',
    )
    parser.add_argument(
        '--target_dir',
        type=Path,
        help='Path to the target directory',
    )
    parser.add_argument(
        '-s',
        '--structures_of_interest',
        type=list,
        help='List of structures to denoise',
        default=["TOMM20", "ACTB", "MYH10", "ACTN1", "LMNB1", "FBL", "NPM1"]
    )

    return parser.parse_args()


def denoise_channel(image, channel, in_focus_slice, model):
    # # extract channel and in focus slice
    infocus_channel_image = image[channel:channel+1, in_focus_slice, ...]

    # prepare for n2v
    infocus_channel_image = np.moveaxis(infocus_channel_image, 0, 3)
    infocus_channel_image = img_as_float32(infocus_channel_image)

    # denoise
    pred = model.predict(infocus_channel_image,
                         axes='ZYXC', n_tiles=(1, 1, 2, 1))

    return pred


def normalize16(I):
    mn = I.min()
    mx = I.max()

    mx -= mn

    I = ((I - mn)/mx) * 65535.0
    return I.astype(np.uint16)


if __name__ == "__main__":
    args = parse_args()
    src_dir = args.src_dir
    target_dir = args.target_dir
    structures_of_interest = args.structures_of_interest

    if not os.path.exists(target_dir):
        # make target dir
        os.mkdir(target_dir)

    # txt file with processed images from previous runs
    if os.path.exists(target_dir / "processed_images.txt"):  # already processed images
        with open(target_dir / "processed_images.txt", "r") as f:
            processed_images = f.read().splitlines()
    else:  # no processed images
        with open(target_dir / "processed_images.txt", "w") as f:
            f.write("")
            processed_images = []
            metadata = pd.read_csv(src_dir / "metadata.csv")

    if len(processed_images) > 0:
        metadata = pd.read_csv(target_dir / "metadata.csv", index_col=0)
    else:  # no processed images
        metadata = pd.read_csv(src_dir / "metadata.csv", index_col=0)

    print("Already processed images (channels): ", len(processed_images))

    # channel: DNA
    model_name = 'n2v_3D_DNA'
    n2v_basedir = 'n2v_models'
    model = N2V(config=None, name=model_name, basedir=n2v_basedir)

    channel = 2
    channel_structure = "DNA"

    for structure in tqdm(structures_of_interest):
        # mkdir for structure
        if not os.path.exists(target_dir / structure):
            os.mkdir(target_dir / structure)

        print("Denoising structure: ", structure, "channel: ", channel)

        fovs = os.listdir(src_dir / structure)  # target_dir is still empty
        for fov in tqdm(fovs):

            img_name = str(Path(structure) / fov)

            if structure+"/"+fov+f"/channel{channel}" in processed_images:
                continue

            if os.path.exists(target_dir / img_name):
                # print("Already denoised: ", img_name)
                continue
            # else: copy to target dir
            shutil.copytree(src_dir / img_name, target_dir / img_name)

            img = zarr.open(str(target_dir / structure / fov))

            channel_zfocus_int = eval(
                metadata[metadata.index == img_name][f'zfocusint_{channel_structure}'].values[0])

            try:
                pred = denoise_channel(
                    img, channel, slice(*channel_zfocus_int), model)
            except Exception as e:
                print(f"Error {e}: {img_name}")
                if os.path.exists(target_dir / "error_images.txt"):
                    with open(target_dir / "error_images.txt", "a") as f:
                        # TODO: channel{channel}
                        f.write(img_name + f"{channel_structure} \n")
                else:
                    with open(target_dir / "error_images.txt", "w") as f:
                        f.write(img_name + f"/channel{channel}\n")
                continue
            pred = pred[..., 0]  # remove channel dimension
            # insert denoised slice back into image
            if pred.min() < -1 or pred.max() > 1:
                print(
                    "Warning: pred min/max not in [-1,1] for image: ", img_name)
                img[channel, slice(
                    *channel_zfocus_int), ...] = normalize16(pred)
            else:
                img[channel, slice(
                    *channel_zfocus_int), ...] = img_as_uint(pred)

            in_focus_mean = img[channel, slice(
                *channel_zfocus_int), ...].mean()
            in_focus_var = img[channel, slice(*channel_zfocus_int), ...].var()

            metadata.loc[
                str(img_name),
                [f"zfocus_mean_{channel_structure}",
                 f"zfocus_var_{channel_structure}",
                 ],
            ] = (
                in_focus_mean,
                in_focus_var,
            )

            metadata.to_csv(target_dir / "metadata.csv")

            with open(target_dir / "processed_images.txt", "a") as f:
                f.write(img_name + f"/channel{channel}\n")

    # channel: cell_membrane
    channel = 0
    channel_structure = "cell_membrane"

    model_name = 'n2v_3D_cell_membrane'
    model = N2V(config=None, name=model_name, basedir=n2v_basedir)

    for structure in tqdm(structures_of_interest):
        print("Denoising structure: ", structure, "channel: ", channel)
        fovs = os.listdir(target_dir / structure)
        for fov in tqdm(fovs):

            if structure+"/"+fov+f"/channel{channel}" in processed_images:
                # print("Already denoised: ", fov)
                continue

            # denoise fov
            img_name = str(Path(structure) / fov)

            # already copied to target dir in previous loop

            img = zarr.open(str(target_dir / structure / fov))

            channel_zfocus_int = eval(
                metadata[metadata.index == img_name][f'zfocusint_{channel_structure}'].values[0])

            try:
                pred = denoise_channel(
                    img, channel, slice(*channel_zfocus_int), model)
            except Exception as e:
                print(f"Error {e}: {img_name}")
                if os.path.exists(target_dir / "error_images.txt"):
                    with open(target_dir / "error_images.txt", "a") as f:
                        # TODO: channel{channel}
                        f.write(img_name + f"{channel_structure} \n")
                else:
                    with open(target_dir / "error_images.txt", "w") as f:
                        f.write(img_name + f"/channel{channel}\n")
                continue
            pred = pred[..., 0]  # remove channel dimension
            # insert denoised slice back into image
            if pred.min() < -1 or pred.max() > 1:
                print(
                    "Warning: pred min/max not in [-1,1] for image: ", img_name)
                img[channel, slice(
                    *channel_zfocus_int), ...] = normalize16(pred)
            else:
                img[channel, slice(
                    *channel_zfocus_int), ...] = img_as_uint(pred)

            in_focus_mean = img[channel, slice(
                *channel_zfocus_int), ...].mean()
            in_focus_var = img[channel, slice(*channel_zfocus_int), ...].var()

            metadata.loc[
                str(img_name),
                [f"zfocus_mean_{channel_structure}",
                 f"zfocus_var_{channel_structure}",
                 ],
            ] = (
                in_focus_mean,
                in_focus_var,
            )

            metadata.to_csv(target_dir / "metadata.csv")

            with open(target_dir / "processed_images.txt", "a") as f:
                f.write(img_name + f"/channel{channel}\n")

    # channel: specific structure
    channel = 1

    for structure in tqdm(structures_of_interest):
        channel_structure = MAP_NAME_STRUCTURE[structure]
        model_name = f'n2v_3D_{channel_structure}'
        try:
            model = N2V(config=None, name=model_name, basedir=n2v_basedir)
        except FileNotFoundError:
            print("Model not found: ", model_name)
            continue

        print("Denoising structure: ", structure, "channel: ", channel)

        fovs = os.listdir(target_dir / structure)

        for fov in tqdm(fovs):
            if structure+"/"+fov+f"/channel{channel}" in processed_images:
                # print("Already denoised: ", fov)
                continue
            # denoise fov
            img_name = str(Path(structure) / fov)
            img = zarr.open(str(target_dir / structure / fov))

            channel_zfocus_int = eval(
                metadata[metadata.index == img_name][f'zfocusint_{channel_structure}'].values[0])

            try:
                pred = denoise_channel(
                    img, channel, slice(*channel_zfocus_int), model)
            except Exception as e:
                print(f"Error {e}: {img_name}")
                if os.path.exists(target_dir / "error_images.txt"):
                    with open(target_dir / "error_images.txt", "a") as f:
                        # TODO: channel{channel}
                        f.write(img_name + f"{channel_structure} \n")
                else:
                    with open(target_dir / "error_images.txt", "w") as f:
                        # TODO: channel{channel}
                        f.write(img_name + f"{channel_structure} \n")
                continue
            pred = pred[..., 0]  # remove channel dimension
            # insert denoised slice back into image
            if pred.min() < -1 or pred.max() > 1:
                print(
                    "Warning: pred min/max not in [-1,1] for image: ", img_name)
                img[channel, slice(
                    *channel_zfocus_int), ...] = normalize16(pred)
            else:
                img[channel, slice(
                    *channel_zfocus_int), ...] = img_as_uint(pred)

            in_focus_mean = img[channel, slice(
                *channel_zfocus_int), ...].mean()
            in_focus_var = img[channel, slice(*channel_zfocus_int), ...].var()

            # metadata = pd.read_csv(target_dir / "metadata.csv")
            metadata.loc[
                str(img_name),
                [f"zfocus_mean_{channel_structure}",
                 f"zfocus_var_{channel_structure}",
                 ],
            ] = (
                in_focus_mean,
                in_focus_var,
            )

            metadata.to_csv(target_dir / "metadata.csv")

            with open(target_dir / "processed_images.txt", "a") as f:
                f.write(img_name + f"/channel{channel}\n")
