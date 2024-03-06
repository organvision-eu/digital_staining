from src.data.aicszarr import MAP_NAME_STRUCTURE, STRUCTURES
import scipy.stats as stats
import zarr
from numcodecs import Blosc
from aicsimageio import AICSImage
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import gc
from pathlib import Path
import glob
import argparse
import os
import quilt3
import numpy as np
os.environ["QUILT_MINIMIZE_STDOUT"] = "true"  # suppress quilt3 progress bars


def get_structure_name(structure_name, channel_name):
    if channel_name.startswith("H3342"):
        return "DNA"
    elif channel_name.startswith("CMDRP"):
        return "cell_membrane"
    elif channel_name.startswith("Bright") or channel_name.startswith("TL"):
        return "bright-field"
    elif structure_name in STRUCTURES:
        return MAP_NAME_STRUCTURE[structure_name]
    else:
        print("Structure_name NOT FOUND:", structure_name)


def plls(img, z):

    # take a square patch of the image 256x256 pixels from the center of the image
    image = img[z, img.shape[-2]//2-128:img.shape[-2] //
                2+128, img.shape[-1]//2-128:img.shape[-1]//2+128]

    npix = image.shape[0]

    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2

    kfreq = np.fft.fftfreq(npix) * npix  # return the FT sample frequencies
    kfreq2D = np.meshgrid(kfreq, kfreq)  # 2D grid of the frequency values
    # radial coordinate (norm) of the frequencies
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)  # array of bin edges
    kvals = 0.5 * (kbins[1:] + kbins[:-1])  # array of bin centers
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic="mean",
                                         bins=kbins)  # mean of the power spectrum in each bin
    # multiply by the area of the bin
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    logx = np.log(kvals)
    logy = np.log(Abins)

    coefficients = np.polyfit(logx, logy, 1)
    slope = coefficients[0]
    # intercept = coefficients[1]

    return -slope


def plls_channel(img):
    plls_channels = []
    for z in range(img.shape[0]):
        slope = plls(img, z)
        plls_channels.append(slope)
    return plls_channels  # return a list of slopes for each slice


def on_focus_slices(img):
    plls_channels = plls_channel(img)
    thr = np.mean(plls_channels)
    on_focus_filter = np.array(plls_channels) >= thr
    # return an array of booleans: True if the slice is on focus, False otherwise
    return on_focus_filter


def get_broken_z(img):  # img is 3D "ZYX"
    # find z slice with max value = 0
    broken_z = []
    for z in range(len(img)):
        if np.max(img[z, :, :]) == 0:
            broken_z.append(z)
    return broken_z


def fix_broken_z(img, broken_z):
    # copy a contiguous slice to the broken slice (usually the previous slice)
    for z in broken_z:
        if z == 0:
            img[z, :, :] = img[z+1, :, :]
            return img
        img[z, :, :] = img[z-1, :, :]
    return img


def find_connected_interval(bool_array, tolerance=2):
    # Find the indices where the True values change
    changes = np.flatnonzero(
        np.diff(np.hstack(([False], bool_array == True, [False])).astype(int)))
    regions = changes.reshape(-1, 2)

    # Adjust the regions to allow at most "tolerance" False values between consecutive True values
    adjusted_regions = []
    start_previous, end_previous = regions[0]
    for i in range(1, len(regions)):
        start = regions[i][0]
        if start - end_previous <= tolerance:  # we are still in the same connected region
            end_previous = regions[i][1]
            continue
        # we have found a new connected region
        adjusted_regions.append((start_previous, end_previous))
        start_previous, end_previous = regions[i]

    # add the last region
    adjusted_regions.append((start_previous, end_previous))

    i = np.diff(adjusted_regions).argmax()
    return list(adjusted_regions[i])


def update_target_infocus_range(infocus_z_interval, target_infocus_range):
    if infocus_z_interval[0] < target_infocus_range[0]:
        target_infocus_range[0] = infocus_z_interval[0]
    if infocus_z_interval[1] > target_infocus_range[1]:
        target_infocus_range[1] = infocus_z_interval[1]
    return target_infocus_range


compressor = Blosc(cname="zstd", clevel=4)  # , shuffle=Blosc.BITSHUFFLE)


def aics2zar(
        aics_img,
        out_file,
        channel_names,
        channel_structures,
        source_channels,
        target_channels,
        store_as_zip=False):

    if not str(out_file).endswith(".zarr"):
        out_file = f"{out_file}.zarr"
    if store_as_zip:
        store = zarr.ZipStore(out_file)
    else:
        store = zarr.DirectoryStore(out_file)

    z = zarr.zeros(
        (len(channel_names),
         *aics_img.shape[-3:]),
        chunks=(1, None, 256, 256),
        store=store,
        dtype=np.uint16,
        overwrite=True,
        compressor=compressor,
    )

    z.attrs["meta"] = {
        "channels": channel_names,
        "channel_structures": channel_structures,
        "dims": aics_img.shape[-4:],
        "in_focus_z_intervals": dict(),
        "in_focus_size": dict(),
        "in_focus_mean": dict(),
        "in_focus_var": dict(),
        "magnification": 120,
        "null_z_slices": dict(),
    }

    # infocus range on the target channels
    target_infocus_range = [100, 0]

    for i, ch_idx in enumerate(target_channels):
        cs = channel_names[i]
        ch_struct = channel_structures[i]
        # print(f"Processing channel {cs}")

        z_pre = aics_img.get_image_data("CZYX", C=ch_idx)[0]

        broken_z = get_broken_z(z_pre)
        if len(broken_z) > 0:
            z_pre = fix_broken_z(z_pre, broken_z)
            z.attrs["meta"]["null_z_slices"][ch_struct] = broken_z

        # on focus slices:
        on_focus_filter = on_focus_slices(z_pre)

        infocus_z_interval = find_connected_interval(on_focus_filter)

        block = z_pre[slice(*infocus_z_interval), ...]

        z[i, slice(*infocus_z_interval), ...] = block

        z.attrs["meta"]["in_focus_z_intervals"][ch_struct] = infocus_z_interval
        target_infocus_range = update_target_infocus_range(
            infocus_z_interval, target_infocus_range)

        z.attrs["meta"]["in_focus_size"][ch_struct] = block.size
        z.attrs["meta"]["in_focus_mean"][ch_struct] = block.mean()
        z.attrs["meta"]["in_focus_var"][ch_struct] = block.var()

    for i, ch_idx in enumerate(source_channels):
        cs = channel_names[i+len(target_channels)]
        ch_struct = channel_structures[i+len(target_channels)]

        z_pre = aics_img.get_image_data("CZYX", C=ch_idx)[0]

        broken_z = get_broken_z(z_pre)
        if len(broken_z) > 0:
            z_pre = fix_broken_z(z_pre, broken_z)
            z.attrs["meta"]["null_z_slices"][ch_struct] = broken_z
        else:  # need to add dummy empty list to avoid error when saving to Parquet
            z.attrs["meta"]["null_z_slices"][ch_struct] = []

        slice_infocus = slice(*target_infocus_range)

        z[i+len(target_channels), slice_infocus, ...] = z_pre[slice_infocus, ...]

        z.attrs["meta"]["in_focus_z_intervals"][ch_struct] = target_infocus_range
        z.attrs["meta"]["in_focus_size"][ch_struct] = z[i +
                                                        len(target_channels), slice_infocus, ...].size
        z.attrs["meta"]["in_focus_mean"][ch_struct] = z[i +
                                                        len(target_channels), slice_infocus, ...].mean()
        z.attrs["meta"]["in_focus_var"][ch_struct] = z[i +
                                                       len(target_channels), slice_infocus, ...].var()

    # print("Z FINAL CHECK, shape:", z.shape)
    return z


def process_name2zarr(
    fov_path,
    fov_id,
    structure_name,
    source_channels,
    target_channels,
    single_fov_metadata,
    # skip_existing=True,
    as_zip=False,
):
    output_dir = Path(str(structure_name))

    out_filename = output_dir / f"{fov_id}.zarr"

    # if skip_existing:
    #     if str(out_filename) in df_metadata.index:
    #         print(out_filename, "already extracted")
    #         return

    img = pkg[fov_path]
    # full_imgs_dir = DATASET_DIR/"full_img_dir"
    # if not os.path.isdir(full_imgs_dir):
    #     os.mkdir(full_imgs_dir)
    img_path = os.path.join(full_imgs_dir, fov_path.split("/")[-1])
    img.fetch(img_path)
    img_aics = AICSImage(img_path)
    all_channel_names = img_aics.channel_names
    channel_names = [all_channel_names[i]
                     for i in target_channels+source_channels]
    channel_structures = [get_structure_name(
        structure_name, cs) for cs in channel_names]
    # check if in channel_sructures there are duplicates
    if len(channel_structures) != len(set(channel_structures)):
        raise ValueError(
            f"Duplicate channel structures for fov {fov_id} in structure {structure_name}")

    try:
        z = aics2zar(
            img_aics,
            DATASET_DIR / out_filename,
            channel_names,
            channel_structures,
            source_channels,
            target_channels,
            store_as_zip=as_zip,
        )
    except Exception as e:
        print("ERROR:", e)

    for i, _ in enumerate(source_channels):
        # build single_fov_metadata with the correct dtype
        single_fov_metadata["specific_structure"] = pd.Series(dtype=str)
        single_fov_metadata["magnification"] = pd.Series(dtype=np.int32)
        single_fov_metadata["src_type"] = pd.Series(dtype=str)
        single_fov_metadata["shape"] = pd.Series(dtype=str)
        single_fov_metadata["src"] = pd.Series(dtype=np.int32)
        single_fov_metadata["null_z_slices"] = pd.Series(dtype=str)
        for cs in channel_structures:
            single_fov_metadata[f"zfocusint_{cs}"] = pd.Series(dtype=str)

        single_fov_metadata.loc[
            str(out_filename),  # row index
            [
                "specific_structure",
                "magnification",
                "src_type",
                "shape",
                "src",
                *channel_structures,
                *[f"zfocusint_{cs}" for cs in channel_structures],
                *[f"zfocus_size_{cs}" for cs in channel_structures],
                *[f"zfocus_mean_{cs}" for cs in channel_structures],
                *[f"zfocus_var_{cs}" for cs in channel_structures],
                "null_z_slices",
            ],
        ] = (
            str(MAP_NAME_STRUCTURE[structure_name]),  # cast to str
            120,  # magnification
            str(channel_structures[i + len(target_channels)]),  # cast to str
            str(z.shape),  # cast to str
            i + len(target_channels),
            *range(len(channel_names)),
            *[str(z.attrs["meta"]["in_focus_z_intervals"][cs])
              for cs in channel_structures],
            *[int(z.attrs["meta"]["in_focus_size"][cs])
              for cs in channel_structures],
            *[float(z.attrs["meta"]["in_focus_mean"][cs])
              for cs in channel_structures],
            *[float(z.attrs["meta"]["in_focus_var"][cs])
              for cs in channel_structures],
            str(z.attrs["meta"]["null_z_slices"]),
        )
    # remove file from disk
    os.remove(img_path)

    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process the files from aics/hipsc_single_cell_image_dataset and convert each sample to zarr."
    )

    parser.add_argument("--dest_dir", type=str,
                        help="dir to store processed zarr files")

    parser.add_argument(
        "-l",
        "--file_limit_per_structure",
        metavar="file_limit_per_structure",
        help="extract up to this number of images for each structure",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "-p",
        "--parallel",
        metavar="num_parallel_jobs",
        help="use multiple parallel jobs",
        type=int,
        default=4,
    )

    parser.add_argument(
        "-z",
        "--as-zip",
        help="save each zarr file inside a zip rather than a dir",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--overwrite", help="ignore existing files", action="store_true", default=False
    )

    parser.add_argument(
        "-s",
        "--structure",
        metavar="selected_structure",
        help=f"process only this structure, possible structures: {STRUCTURES}",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-i",
        "--fovid",
        metavar="fov_id",
        help=f"process only this fov_id",
        default=None,
        type=int,
    )

    args = parser.parse_args()
    zarr_as_zip = args.as_zip
    file_limit_per_structure = args.file_limit_per_structure
    num_parallel_jobs = args.parallel
    selected_structure = args.structure
    overwrite = args.overwrite
    specific_fov_id = args.fovid

    DATASET_DIR = Path(args.dest_dir)
    SINGLE_FOV_METADATA_DIR = DATASET_DIR / "single_fovs_metadata"

    if not os.path.isdir(DATASET_DIR):
        os.mkdir(DATASET_DIR)

    if not os.path.isdir(SINGLE_FOV_METADATA_DIR):
        os.mkdir(SINGLE_FOV_METADATA_DIR)

    full_imgs_dir = DATASET_DIR/"full_img_dir"
    if not os.path.isdir(full_imgs_dir):
        os.mkdir(full_imgs_dir)

    # if there are already images in full_imgs_dir, delete them (from a previous interrupted run of this script)
    if len(os.listdir(full_imgs_dir)) > 0:
        print("Found existing images in full_imgs_dir, deleted.")
        for f in glob.glob(str(full_imgs_dir / "*")):
            os.remove(f)

    if overwrite:
        try:
            os.remove(DATASET_DIR / "metadata.csv")
            print("Found existing metadata.csv, deleted.")
        except Exception:
            pass

    # unite all the metadata files in SINGLE_FOV_METADATA_DIR into a single metadata.csv file in DATASET_DIR
    # single metadata files are named metadata_{fov_id}.parquet
    # if SINGLE_FOV_METADATA_DIR is not empty, merge the metadata files

    if len(os.listdir(SINGLE_FOV_METADATA_DIR)) > 0:
        print("Merging metadata files...")
        # new_metadata = pd.read_parquet(SINGLE_FOV_METADATA_DIR)
        new_metadata = pd.DataFrame()
        for f in glob.glob(str(SINGLE_FOV_METADATA_DIR / "*.parquet")):
            single_fov_metadata = pd.read_parquet(f)
            new_metadata = pd.concat([new_metadata, single_fov_metadata])
        print("Found existing metadata.parquet files, merged.")
        print("Number of files found: ", len(new_metadata))

        # merge with existing metadata, if any
        try:
            df_metadata = pd.read_csv(
                DATASET_DIR / "metadata.csv", index_col="filename")
            df_metadata = pd.concat([df_metadata, new_metadata])
            print("Found existing metadata.csv, merged.")
        except FileNotFoundError:
            df_metadata = new_metadata

        df_metadata.to_csv(DATASET_DIR / "metadata.csv")

        # #remove metadata files: empty the dir
        for f in glob.glob(str(SINGLE_FOV_METADATA_DIR / "*")):
            os.remove(f)

    # load metadata.csv
    try:
        df_metadata = pd.read_csv(
            DATASET_DIR / "metadata.csv", index_col="filename")
    except Exception:
        df_metadata = pd.DataFrame()
        df_metadata.index.name = "filename"

    pkg = quilt3.Package.browse(
        "aics/hipsc_single_cell_image_dataset", registry="s3://allencell")  # source

    fovs_meta_path = DATASET_DIR / 'aics_fovs_meta.csv'
    if not os.path.isfile(fovs_meta_path):
        meta = pkg['metadata.csv']
        aics_metadata_path = DATASET_DIR / 'aics_metadata.csv'
        meta.fetch(aics_metadata_path)
        fovs_df = pd.read_csv(aics_metadata_path, dtype_backend="pyarrow", engine='pyarrow', usecols=[
            'fov_path', 'structure_name', 'scale_micron',
            'FOVId', 'PlateId', 'WellId', 'cell_stage',
            # 'InstrumentId', 'WorkflowId', 'ProtocolId', 'PiezoId',
            'ChannelNumberStruct', 'ChannelNumberBrightfield', 'ChannelNumber405', 'ChannelNumber638',
            'meta_fov_position', 'meta_imaging_mode'
        ]
            # , index_col='fov_path'
        ).drop_duplicates(subset='fov_path')
        fovs_df.to_csv(fovs_meta_path, index=False)
        print("length of fovs_df: ", len(fovs_df))
        # os.unlink(aics_metadata_path)
    else:
        fovs_df = pd.read_csv(fovs_meta_path, dtype_backend="pyarrow", engine='pyarrow'
                              #   , index_col='fov_path'
                              )

    if specific_fov_id is not None:
        fovs_df = fovs_df[fovs_df["FOVId"] == specific_fov_id]
        if len(fovs_df) == 0:
            raise ValueError(f"Invalid FOVId: {specific_fov_id}")

    if selected_structure is not None:
        if selected_structure not in STRUCTURES:
            raise ValueError(f"Invalid structure name: {selected_structure}")
        fovs_df = fovs_df[fovs_df["structure_name"] == selected_structure]

    # number of FOVId to sample for each cell line
    num_sample_structure = file_limit_per_structure
    if num_sample_structure > 0:
        data_lines_stratified = fovs_df.groupby(
            "structure_name", group_keys=False)
        data_lines_stratified = data_lines_stratified.apply(
            pd.DataFrame.sample, n=num_sample_structure)
        data_lines_stratified = data_lines_stratified.reset_index(drop=True)

    else:
        data_lines_stratified = fovs_df

    def process_fov(fov):
        single_fov_metadata = pd.DataFrame()
        single_fov_metadata.index.name = "filename"
        structure_name = fov["structure_name"]
        fov_path = fov["fov_path"]
        fov_id = fov["FOVId"]
        bf_channel = fov["ChannelNumberBrightfield"]
        struct_channel = fov["ChannelNumberStruct"]
        _405_channel = fov["ChannelNumber405"]
        _638_channel = fov["ChannelNumber638"]
        source_channels = [bf_channel]
        target_channels = [_638_channel, struct_channel, _405_channel]

        try:
            process_name2zarr(
                fov_path,
                fov_id,
                structure_name,
                source_channels,
                target_channels,
                single_fov_metadata,
                # skip_existing=not overwrite,
                as_zip=zarr_as_zip,
            )
        except Exception as e:
            print("ERROR:", e)
            print(f"Fov: {fov_path}, structure: {structure_name}")
            exit(1)
        finally:
            single_fov_metadata.to_parquet(
                SINGLE_FOV_METADATA_DIR / f"metadata_{fov_id}.parquet")

    # remove fovs that have already been processed
    # if df_metadata is empty, then all fovs will be processed
    if len(df_metadata) > 0:
        fov_id_to_filter_out = list(
            df_metadata.index.str[:-5].str.split("/").str[-1].astype(int))
        data_lines_stratified = data_lines_stratified[~data_lines_stratified["FOVId"].isin(
            fov_id_to_filter_out)]

    parallel = Parallel(n_jobs=num_parallel_jobs)
    parallel(delayed(process_fov)(fov) for _, fov in tqdm(
        data_lines_stratified.iterrows(), total=len(data_lines_stratified)))

    # merge all the metadata files in SINGLE_FOV_METADATA_DIR into a single metadata.csv file in DATASET_DIR
    # single metadata files are named metadata_{fov_id}.parquet
    if len(os.listdir(SINGLE_FOV_METADATA_DIR)) > 0:
        print("Finally merging metadata files...")
        # new_metadata = pd.read_parquet(SINGLE_FOV_METADATA_DIR)
        new_metadata = pd.DataFrame()
        for f in glob.glob(str(SINGLE_FOV_METADATA_DIR / "*.parquet")):
            single_fov_metadata = pd.read_parquet(f)
            new_metadata = pd.concat([new_metadata, single_fov_metadata])
        print(len(new_metadata))
        print("Found existing metadata.parquet files, merged.")

        # merge with existing metadata, if any
        try:
            df_metadata = pd.read_csv(
                DATASET_DIR / "metadata.csv", index_col="filename")
            df_metadata = pd.concat([df_metadata, new_metadata])
            print("Found existing metadata.csv, merged.")
        except FileNotFoundError:
            df_metadata = new_metadata

        df_metadata.to_csv(DATASET_DIR / "metadata.csv")

        # remove metadata files: empty the dir
        for f in glob.glob(str(SINGLE_FOV_METADATA_DIR / "*")):
            os.remove(f)
