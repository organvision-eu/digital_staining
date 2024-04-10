import numpy as np
import pathlib
import pandas as pd
import sys
import zarr
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from ..utils.utils import normalize

MAP_NAME_STRUCTURE = {
    "TOMM20": "mitochondria",
    "ACTB": "actin_filaments",
    "CETN2": "centrioles",
    "TUBA1B": "microtubules",
    "LMNB1": "nuclear_envelope",
    "DSP": "desmosomes",
    "SEC61B": "ER(Sec61_beta)",
    "ST6GAL1": "golgi_apparatus",
    "SON": "nuclear speckles",
    "GJA1": "gap_junctions",
    "AAVS1": "plasma_membrane",
    "MYH10": "actomyosin_bundles",
    "TJP1": "tight_junctions",
    "ACTN1": "actin_bundles",
    "LAMP1": "lysosomes",
    "FBL": "nucleoli(DFC)",
    "HIST1H2BJ": "histones",
    "PXN": "matrix_adhesions",
    "NPM1": "nucleoli(GC)",
    "NUP153": "nuclear_pores",
    "ATP2A2": "ER(SERCA2)",
    "CTNNB1": "adherens_junctions",
    "RAB5A": "endosomes",
    "SLC25A17": "peroxisomes",
    "SMC1A": "cohesins",
}

STRUCTURES = list(MAP_NAME_STRUCTURE.keys())
src_channel = "bright-field"
TARGET_CHANNELS = ["DNA", "cell_membrane",
                   "mitochondria", "actin_filaments", "actomyosin_bundles", "actin_bundles", "nuclear_envelope",
                   "nucleoli(DFC)", "nucleoli(GC)"]


def im2patch(im, pch_size, stride=1):
    '''
    Transform image to patches.
    Input:
        im: 3 x H x W or 1 X H x W image, numpy format
        pch_size: (int, int) tuple or integer
        stride: (int, int) tuple or integer
    '''
    if isinstance(pch_size, tuple):
        pch_H, pch_W = pch_size
    elif isinstance(pch_size, int):
        pch_H = pch_W = pch_size
    else:
        sys.exit('The input of pch_size must be a integer or a int tuple!')

    if isinstance(stride, tuple):
        stride_H, stride_W = stride
    elif isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        sys.exit('The input of stride must be a integer or a int tuple!')

    C, H, W = im.shape
    num_H = len(range(0, H-pch_H+1, stride_H))
    num_W = len(range(0, W-pch_W+1, stride_W))
    num_pch = num_H * num_W
    pch = np.zeros((C, pch_H*pch_W, num_pch), dtype=im.dtype)
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H-pch_H+ii+1:stride_H, jj:W-pch_W+jj+1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.reshape((C, pch_H, pch_W, num_pch))


def parse_list(x):
    """Parse a string of a list/tuple of integers to a numpy array"""
    return np.asarray([int(i) for i in x.strip("[]()").split(",")])


class AICSDataset:
    def __init__(
        self,
        root_dir: str = ".",
        signal_channel: str = "bright-field",
        # , 'specific_structure'],
        target_channels: list[str] = ["DNA", "cell_membrane"],
        # augment: bool = False,
        patch_shape: int | tuple = (256, 256),
        patch_strides: int | tuple | None = None,
        ndim: int | None = None,
        ignore_incomplete_patches: bool | str = True,  # or 'xy'
        # tuple of ratios (w/ ref to z length) for relative start, end and an int for absolute step; or tuple('in-focus-hint', step:int) which uses the 'in_focus_z_intervals' col from metadata # TODO: use namedtuple
        z_range: tuple[float | int] | None = None,
        dtype: np.dtype = np.float16,
        # dtype: torch.dtype = torch.float16,
        use_normalization: bool | str = True,  # 'Lp', 'standard', 'normal_global'
        normalization_p: float | None = 2,  # only used if normalize=='Lp',
        # provide dataset-wide mean/std (if None, will normalize patch-wise)
        normalization_stats: dict | None = None,
    ):
        self.root_dir = pathlib.Path(root_dir)

        self.dtype = dtype

        self.signal_channel = signal_channel
        self.target_channels = target_channels

        self.use_normalization = use_normalization
        self.normalization_p = normalization_p

        self.signal_mu_std = (None, None)
        self.target_mu_std = (None, None)
        if normalization_stats:
            # FIXME: make it general and adjust it for different src types, e.g., bf, dic, glim, etc
            self.signal_mu_std = (
                [normalization_stats['bright-field']['mean'].astype(dtype)],
                [normalization_stats['bright-field']['std'].astype(dtype)]
            )

            self.target_mu_std = (
                [normalization_stats[ch]['mean'].astype(
                    dtype) for ch in target_channels],
                [normalization_stats[ch]['std'].astype(
                    dtype) for ch in target_channels]
            )

        if isinstance(patch_shape, int):
            if ndim is None:
                raise ValueError(
                    "You need to explicitly provide either the patch ndim or the shape")
            self.patch_shape = np.repeat(patch_shape, ndim)
        else:
            self.patch_shape = np.asarray(patch_shape)

        self.ndim = len(self.patch_shape)

        if isinstance(patch_strides, str) and patch_strides == 'random':
            self.patch_strides = patch_strides
        elif patch_strides is None:
            # use non-overlapping patches
            self.patch_strides = self.patch_shape
        elif isinstance(patch_strides, int):
            self.patch_strides = np.repeat(patch_strides, self.ndim)
        else:
            self.patch_strides = np.asarray(patch_strides)

        # exclude patches that would need padding
        self.ignore_incomplete_patches = ignore_incomplete_patches

        self.z_range = z_range

    def _patch_indices(self, meta, available_channels: list | pd.Index | None = None):
        data_shape = parse_list(meta['shape'])

        z_interval = None
        if self.z_range is not None:

            # 'in-focus-hint'
            if isinstance(self.z_range[0], str) or self.z_range in ('in-focus-hint', 'in-focus-centre'):
                # print('patch', available_channels)
                if not available_channels:
                    # available_channels = list(meta[self.target_channels].dropna().index)
                    available_channels = meta[self.target_channels].dropna(
                    ).index
                # print('patch', available_channels)
                # get interpatch_shape across all the target channels we are using (source channel is not important here -> we want nice predictions irrespective of the source quality)
                z_intervals = [parse_list(
                    meta[f'zfocusint_{c}']) for c in available_channels]
                if len(self.target_channels) > 1:
                    z_intervals = np.vstack(z_intervals)
                    z_interval = np.max(z_intervals[:, 0]), np.min(
                        z_intervals[:, 1])
                    if z_interval[1] < z_interval[0]:
                        return None
                        raise ValueError(
                            "Empty overlap of in-focus z intervals.")
                else:
                    z_interval = z_intervals[0]

                if self.z_range == 'in-focus-centre':
                    iter_z = int(np.mean(z_interval))
                else:
                    if self.ndim == 2:
                        # self.z_range[1] is the step when using 'in-focus-hint'
                        iter_z = np.arange(*z_interval, self.z_range[1])
            else:
                # self.z_range is (start_ratio, end_ratio, step: int)
                if self.ndim == 2:
                    iter_z = np.arange(int(
                        data_shape[1] * self.z_range[0]), int(data_shape[1] * self.z_range[1]), self.z_range[2])
        elif self.ndim == 2:
            iter_z = np.arange(data_shape[1])

        if z_interval is not None and self.ndim == 3:
            data_shape[1] = np.diff(z_interval)[0]
        # else:
        #     z_interval = np.array([0, data_shape[1]])  # use whole range

        if isinstance(self.patch_strides, str) and self.patch_strides == 'random':
            ranges = data_shape[-self.ndim:] - np.array(self.patch_shape)
            if iter_z is not None:
                slice_z = iter_z
            else:
                slice_z = np.random.randint(z_interval[0], z_interval[1])
                slice_z = slice(slice_z, slice_z + self.patch_shape[0])
            slice_y = np.random.randint(0, ranges[-2])
            slice_y = slice(slice_y, slice_y + self.patch_shape[-2])
            slice_x = np.random.randint(0, ranges[-1])
            slice_x = slice(slice_x, slice_x + self.patch_shape[-1])
            return [(slice_z, slice_y, slice_x)]

        # TODO: if x or y patch size is None, use whole image (ie, slice(None))
        # if self.

        n_patches = (data_shape[-self.ndim:] - np.array(self.patch_shape) +
                     self.patch_strides) / self.patch_strides

        offset_x, offset_y = 0, 0
        if self.ignore_incomplete_patches == 'xy':
            # z's (will be overwritten by the following if just 2-D)
            n_patches[0] = np.max([1, np.ceil(n_patches[0])])
            n_patches[-2:] = np.floor(n_patches[-2:])
            n_patches = n_patches.astype(int)

            offset_y = (data_shape[-2] - ((n_patches[-2]-1) *
                        self.patch_strides[-2] + self.patch_shape[-2]))//2
            offset_x = (data_shape[-1] - ((n_patches[-1]-1) *
                        self.patch_strides[-1] + self.patch_shape[-1]))//2
        elif self.ignore_incomplete_patches is True:
            n_patches = np.floor(n_patches).astype(int)
            offset_y = (data_shape[-2] - ((n_patches[-2]-1) *
                        self.patch_strides[-2] + self.patch_shape[-2]))//2
            offset_x = (data_shape[-1] - ((n_patches[-1]-1) *
                        self.patch_strides[-1] + self.patch_shape[-1]))//2
        else:
            n_patches = np.ceil(n_patches).astype(int)

        if not np.all(n_patches):
            return None
            # raise ValueError("Selected stride and/or patch size result in empty patches.")

        if self.ndim == 3:
            if self.z_range == 'in-focus-centre':
                # ignore z stride and centre the image
                z_start = int(np.mean(z_interval)) - self.patch_shape[0] // 2
                slices_z = slice(z_start, z_start + self.patch_shape[0], )
            else:
                slices_z = [slice(self.patch_strides[0] * z + z_interval[0], self.patch_strides[0]
                                  * z + z_interval[0] + self.patch_shape[0], ) for z in range(n_patches[0])]
        else:
            slices_z = iter_z

        slices_y = [slice(self.patch_strides[-2] * y + offset_y, self.patch_strides[-2]
                          * y + self.patch_shape[-2] + offset_y, ) for y in range(n_patches[-2])]
        slices_x = [slice(self.patch_strides[-1] * x + offset_x, self.patch_strides[-1]
                          * x + self.patch_shape[-1] + offset_x, ) for x in range(n_patches[-2])]

        return np.stack(np.meshgrid(slices_z, slices_y, slices_x), -1).reshape(-1, 3)

    def _pad(self, patch):
        pad_size = self.patch_shape - patch.shape[-self.ndim:]
        if np.any(pad_size > 0):
            pad = np.zeros(self.ndim * 2, dtype=np.uint16)
            pad[1::2] = pad_size[::-1]
            return F.pad(patch, pad.tolist(), "constant", 0)
        return patch


class AICSZarrPatchExpandedDataset(Dataset, AICSDataset):
    """Read zarr file (data is 'CZYX')"""

    def __init__(
        self,
        file_list_meta: pd.DataFrame,
        root_dir: str = ".",
        signal_channel: str = "src",
        # , 'specific_structure'],
        target_channels: list[str] = ["DNA", "cell_membrane"],
        # augment: bool = False,
        patch_shape: int | tuple = (256, 256),
        patch_strides: int | tuple | None = None,
        ndim: int | None = None,
        ignore_incomplete_patches: bool | str = True,  # or 'xy'
        # tuple of ratios (w/ ref to z length) for relative start, end and an int for absolute step; or tuple('in-focus-hint', step:int) which uses the 'in_focus_z_intervals' col from metadata # TODO: use namedtuple
        z_range: tuple[float | int] | None = None,
        dtype: np.dtype = np.float16,
        # dtype: torch.dtype = torch.float16,
        # 'Lp', 'standard', 'normal_global', 'minmax', 'minmax_global
        use_normalization: bool | str = True,
        normalization_p: float | None = 2,  # only used if normalize=='Lp',
        # provide dataset-wide mean/std (if None, will normalize patch-wise)
        normalization_stats: dict | None = None,
        shuffle: bool = True,
        random_seed: int = 0,
    ):
        """Args:
            file_list_meta (pd.DataFrame): metadata dataframe 
            root_dir (str): root directory with subdirs representing the different structures, containing the zarr files.
            signal_channel (str): Name of the signal channel. (source)
            target_channels (list): List of target channel names.
            patch_shape (tuple): Size of the patch. 
            patch_strides (tuple): Stride of the patch. If None, non-overlapping patches are used.
            ndim (int): Number of dimensions of the patch.
            ignore_incomplete_patches (bool): Ignore incomplete patches.
            z_range (tuple): Range of z values to use.
            dtype (np.dtype): Data type of the patch.
            use_normalization (str): Type of normalization to use.
            normalization_p (float): p value for Lp normalization.
            normalization_stats (dict): Dictionary with the mean and std of the data.
            shuffle (bool): Shuffle the dataset.
            random_seed (int): Random seed for shuffling."""
        
        AICSDataset.__init__(self, root_dir, signal_channel, target_channels, patch_shape, patch_strides, ndim,
                             ignore_incomplete_patches, z_range, dtype, use_normalization, normalization_p, normalization_stats)

        # create examples from each row
        self.file_list_meta = file_list_meta[[
            self.signal_channel] + self.target_channels].copy().astype(np.float16)
        self.file_list_meta['slice_indices'] = file_list_meta.apply(
            self._patch_indices, axis=1)
        self.file_list_meta = self.file_list_meta[~self.file_list_meta['slice_indices'].isna(
        )]

        self.file_list_meta = self.file_list_meta.explode('slice_indices')

        if self.use_normalization == 'standard_per_fov':
            scale_factor = np.iinfo(np.uint16).max
            # TODO: use generic src type
            file_list_meta[[f"zfocus_std_{ch}" for ch in self.target_channels + ['bright-field']]] = (np.sqrt(
                file_list_meta[[f"zfocus_var_{ch}" for ch in self.target_channels + ['bright-field']]]) / scale_factor).astype(dtype)
            file_list_meta[[f"zfocus_mean_{ch}" for ch in self.target_channels + ['bright-field']]] = (
                file_list_meta[[f"zfocus_mean_{ch}" for ch in self.target_channels + ['bright-field']]] / scale_factor).astype(dtype)
            file_list_meta = file_list_meta[[f"zfocus_mean_{ch}" for ch in self.target_channels + ['bright-field']] + [
                f"zfocus_std_{ch}" for ch in self.target_channels + ['bright-field']] + ['src_type']]

            self.file_list_meta = self.file_list_meta.join(file_list_meta)

        if shuffle:
            self.file_list_meta = self.file_list_meta.sample(
                frac=1, random_state=random_seed)

    def __len__(self):
        return len(self.file_list_meta)

    def __getitem__(self, idx):
        meta = self.file_list_meta.iloc[idx]
        filename = meta.name

        chs = meta[self.target_channels].astype(np.float16).to_numpy()
        av_channels = ~np.isnan(chs)  # the channels we have in this example
        # each target_zarr_idx will land in pos target_idx
        target_zarr_ch_idx = chs[av_channels].astype(np.int8)
        target_ch_idx = np.argwhere(av_channels).flatten()

        # print(self.target_channels, target_zarr_ch_idx, target_ch_idx)
        if len(self.target_channels) > len(target_ch_idx):
            av_target_channels = [self.target_channels[i]
                                  for i in target_ch_idx]
            # TODO: explore sparse or masked (https://pytorch.org/maskedtensor/main/notebooks/overview.html) tensors
            # raise NotImplementedError('Sparse tensors for missing channels are not implemented yet')
        else:
            av_target_channels = self.target_channels

        src_ch_idx = int(meta[self.signal_channel])

        data = zarr.open(self.root_dir / filename, "r")

        try:
            slice_z, slice_y, slice_x = meta['slice_indices']
        except:
            # print(cur['slice_indices'])
            raise Exception(str(meta['slice_indices']))

        # preserve channel axis (list vs int)
        src_data = torch.from_numpy(
            data.oindex[[src_ch_idx], slice_z, slice_y, slice_x].astype(self.dtype))
        target_data = torch.from_numpy(
            data.oindex[
                target_zarr_ch_idx, slice_z, slice_y, slice_x
            ].astype(self.dtype)
        )

        # scale values to [0, 1] (assuming src is uint)
        src_data /= np.iinfo(data.dtype).max
        target_data /= np.iinfo(data.dtype).max

        if self.use_normalization:
            if self.use_normalization == 'standard_per_fov':
                src_type = meta['src_type']
                signal_mu_std = ([meta[f"zfocus_mean_{src_type}"]], [
                                 meta[f"zfocus_std_{src_type}"]])
                target_mu_std = (
                    [meta[f"zfocus_mean_{ch}"] for ch in av_target_channels],
                    [meta[f"zfocus_mean_{ch}"] for ch in av_target_channels]
                )
            else:
                signal_mu_std = self.signal_mu_std
                target_mu_std = self.target_mu_std

            # # normalize each spatial dimension (dim 0 is channels)
            src_data = normalize(src_data, self.use_normalization.replace(
                '_per_fov', ''), self.normalization_p, mu=signal_mu_std[0], std=signal_mu_std[1])
            target_data = normalize(target_data, self.use_normalization.replace(
                '_per_fov', ''), self.normalization_p, mu=target_mu_std[0], std=target_mu_std[1])

        src_data = self._pad(src_data)
        target_data = self._pad(target_data)

        # create torch tensor with all target channels
        if self.ndim == 2:
            # build a tensor with all NaNs
            target_data_multich = torch.full(
                (len(self.target_channels), target_data.shape[1], target_data.shape[2]), float('nan'))
            target_data_multich[target_ch_idx, :, :] = target_data

        if self.ndim == 3:
            # build a tensor with all NaNs
            target_data_multich = torch.full((len(self.target_channels), target_data.shape[1],
                                              target_data.shape[2], target_data.shape[3]), float('nan'))
            target_data_multich[target_ch_idx, :, :, :] = target_data

        return src_data, target_data_multich


def read_metadata_csv(
    DATASET_DIR: pathlib.Path,
    src_types: str | list[str] | None,
    target_channels: list[str],
    magnifications: int | list[int] = [100],
    compute_pooled_stats: bool | None = False,
    specific_structures: str | list[str] | None = None,
    unify_channels: bool = False,
):
    """Read metadata csv file and filter on src_types, target_channels, magnifications and specific_structures.
    Args:
        DATASET_DIR (pathlib.Path): Path to the dataset directory.
        src_types (str, list): Source types to filter on.
        target_channels (list): Target channels to filter on.
        magnifications (int, list): Magnifications to filter on.
        compute_pooled_stats (bool): Compute pooled stats.
        specific_structures (str, list): Specific structures to filter on.
        unify_channels (bool): Unify channels to myofibrils (actin_filaments, actin_bundles, actomyosin_bundles) and nucleoli (nucleoli(DFC), nucleoli(GC))."""
    
    df_metadata = pd.read_csv(
        DATASET_DIR / "metadata.csv",
        index_col="filename",
        engine="pyarrow",
        dtype_backend="pyarrow",
    )

    # filter on target channels
    filtered = df_metadata[target_channels].dropna(thresh=1).index
    df_metadata = df_metadata.loc[filtered]

    # filter on magnification
    if magnifications is not None:
        if isinstance(magnifications, int):
            magnifications = [magnifications]
        df_metadata = df_metadata[df_metadata["magnification"].isin(
            magnifications)]

    # filter on src channel types
    if src_types is not None:
        if isinstance(src_types, str):
            src_types = [src_types]
        df_metadata = df_metadata[df_metadata["src_type"].isin(src_types)]
    else:
        src_types = []

    # filter specific structures
    if specific_structures is not None:
        if isinstance(specific_structures, str):
            specific_structures = [specific_structures]
        df_metadata = df_metadata[df_metadata["specific_structure"].isin(
            specific_structures)]

    # downcast integer columns
    icols = df_metadata.select_dtypes("int64[pyarrow]").columns
    df_metadata[icols] = df_metadata[icols].apply(
        pd.to_numeric, downcast="unsigned")

    # df_metadata["shape"] = df_metadata["shape"].astype(tuple)

    if compute_pooled_stats:
        channels_pooled_stats = dict()

        # compute pooled stats
        # https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups/2971563#2971563
        # https://en.wikipedia.org/wiki/Pooled_variance

        # TODO: if FoV has multiple (selected) src types, each channel will be counted multiple times... (shouldn't be a problem, but it's useless)
        for ch in set(target_channels) | set(src_types):
            sizes = df_metadata[f"zfocus_size_{ch}"]
            vars = df_metadata[f"zfocus_var_{ch}"]
            means = df_metadata[f"zfocus_mean_{ch}"]

            tot_size = sizes.sum()
            scale_factor = np.iinfo(np.uint16).max  # scale values to [0, 1]

            mean_pooled = np.dot(sizes, means) / tot_size

            # var = np.dot(sizes - 1, vars) / (sizes - 1).sum()
            var = (
                np.dot(sizes - 1, vars)
                + np.dot(sizes, means**2)
                - mean_pooled * tot_size
            ) / (tot_size - 1)
            var /= scale_factor**2

            channels_pooled_stats[ch] = {
                "mean": mean_pooled / scale_factor,
                "var": var,
                "std": np.sqrt(var),
            }
    else:
        channels_pooled_stats = None

    df_metadata = df_metadata.drop(
        df_metadata.columns[df_metadata.columns.str.startswith(
            "zfocus_")], axis=1
    )

    if unify_channels:
        # copy metadata
        df_metadata = df_metadata.copy()

        # replace all act* specific_structure with "myofibrils"
        df_metadata.loc[df_metadata["specific_structure"].str.contains(
            "act"), "specific_structure"] = "myofibrils"
        df_metadata.loc[df_metadata["specific_structure"].str.contains(
            "nucleoli"), "specific_structure"] = "nucleoli"

        # create myofibrils column: has value 1 if specific_structure is myofibrils, 0 otherwise
        df_metadata["myofibrils"] = df_metadata["specific_structure"].apply(
            lambda x: 1.0 if x == "myofibrils" else np.nan)

        # create nucleli column: has value 1 if specific_structure is nucleoli, 0 otherwise
        df_metadata["nucleoli"] = df_metadata["specific_structure"].apply(
            lambda x: 1.0 if x == "nucleoli" else np.nan)

        # Define a custom function to get the first non '' value
        def non_empty(row):
            for x in row:
                if not pd.isna(x):
                    return x

        # create zfocusint_myofibrils column with the value from zfocusint_act* column
        df_metadata["zfocusint_myofibrils"] = df_metadata[["zfocusint_actin_bundles",
                                                           "zfocusint_actin_filaments", "zfocusint_actomyosin_bundles"]].apply(non_empty, axis=1)
        # create zfocusint_nucleoli column with the value from zfocusint_nucleoli(DFC) or zfocusint_nucleoli(GC) column
        df_metadata["zfocusint_nucleoli"] = df_metadata[[
            "zfocusint_nucleoli(DFC)", "zfocusint_nucleoli(GC)"]].apply(non_empty, axis=1)

        new_target_channels = target_channels.copy()
        myofibrils_channels = False
        if "actin_bundles" in new_target_channels:
            new_target_channels.remove("actin_bundles")
            myofibrils_channels = True
        if "actin_filaments" in new_target_channels:
            new_target_channels.remove("actin_filaments")
            myofibrils_channels = True
        if "actomyosin_bundles" in new_target_channels:
            new_target_channels.remove("actomyosin_bundles")
            myofibrils_channels = True
        if myofibrils_channels:
            new_target_channels.append("myofibrils")
        
        nucleoli_channels = False
        if "nucleoli(DFC)" in new_target_channels:
            new_target_channels.remove("nucleoli(DFC)")
            nucleoli_channels = True
        if "nucleoli(GC)" in new_target_channels:
            new_target_channels.remove("nucleoli(GC)")
            nucleoli_channels = True
        if nucleoli_channels:
            new_target_channels.append("nucleoli")

        return df_metadata, channels_pooled_stats, new_target_channels
    
    return df_metadata, channels_pooled_stats, target_channels


def pad_volume_to_multiple_of_8(volume):
    """Pad a 3D volume to have dimensions that are multiples of 8."""
    depth, height, width = volume.size()[-3:]

    # Calculate the amount of padding needed for each dimension
    pad_depth = (8 - depth % 8) % 8
    pad_height = (8 - height % 8) % 8
    pad_width = (8 - width % 8) % 8

    # Apply padding
    padding = (pad_width // 2, pad_width - pad_width // 2,  # left, right
               pad_height // 2, pad_height - pad_height // 2,  # top, bottom
               pad_depth // 2, pad_depth - pad_depth // 2)  # front, back
    padded_volume = torch.nn.functional.pad(volume, padding)

    return padded_volume


def zarr_to_input(img_path, src_channel=3, section=None, device='cuda:0', pad=True):
    """section is a tuple of 2 tuples, each containing the start and end of the section in the z,x,y axis"""
    img = zarr.open(str(img_path), mode='r')
    source = torch.from_numpy(img[src_channel].astype(np.float32))
    src_data = normalize(source.unsqueeze(0), 'standard')
    src_data = src_data.unsqueeze(0)
    if section:
        src_data = src_data[:, :, section[0][0]:section[1][0],
                            section[0][1]:section[1][1], section[0][2]:section[1][2]]
    if pad:
        src_data = pad_volume_to_multiple_of_8(src_data)
    src_data = src_data.to(device)
    return src_data
