import pathlib
import argparse
from src.data.aicszarr import TARGET_CHANNELS


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_training_parser_argument(parser):

    parser.add_argument(
        "-adv",
        "--adversarial_training",
        metavar="adv",
        help="use adversarial training",
        type=str2bool,
        default=True,
    )

    parser.add_argument(
        "-d",
        "--dataset",
        metavar="dataset",
        help="dataset directory",
        type=pathlib.Path,
        required=True
    )
    parser.add_argument(
        "-s",
        "--structures_of_interest",
        metavar="structures_of_interest",
        help="structures of interest",
        nargs = '+',
        default=["TOMM20", "ACTB", "MYH10", "ACTN1", "LMNB1", "FBL", "NPM1"],
    )
    
    parser.add_argument(
        "-tc",
        "--target_channels",
        nargs='+',
        default = TARGET_CHANNELS,
        help="target channels",
    )
    
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="epochs",
        help="number of epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-lr_g",
        "--learning_rate_generator",
        metavar="learning_rate_generator",
        help="learning rate generator",
        type=float,
        default=0.00005,
    )
    parser.add_argument(
        "-lr_c",
        "--learning_rate_critic",
        metavar="learning_rate_critic",
        help="learning rate critic",
        type=float,
        default=0.00005,
    )
    # negative slope of leaky relu generator
    parser.add_argument(
        "-ns_g",
        "--negative_slope_generator",
        metavar="negative_slope_generator",
        help="negative slope generator",
        type=float,
        default=0.05,
    )
    # negative slope of leaky relu critic
    parser.add_argument(
        "-ns_c",
        "--negative_slope_critic",
        metavar="negative_slope_critic",
        help="negative slope critic",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "-bs_t",
        "--batch_size_training",
        metavar="batch_size_training",
        help="batch size training",
        type=int,
        default=16,
    )

    parser.add_argument(
        "-ps_t",
        "--patch_shape_training",
        metavar="patch_shape_training",
        help="patch shape training",
        type=tuple,
        default=(16, 128, 128),
    )

    parser.add_argument(
        "-pst_t",
        "--patch_stride_training",
        metavar="patch_stride_training",
        help="patch stride training",
        type=tuple,
        default=(8, 64, 64),
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        metavar="batch_size",
        help="batch size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "-ps",
        "--patch_shape",
        metavar="patch_shape",
        help="patch shape",
        type=tuple,
        default=(16, 384, 384),
    )

    parser.add_argument(
        "-pst",
        "--patch_stride",
        metavar="patch_stride",
        help="patch stride",
        type=tuple,
        default=(16, 384, 384),
    )

    parser.add_argument(
        "-z",
        "--z_range",
        metavar="z_range",
        help="z_range",
        type=str,
        default="in-focus-centre",  # in-focus-centre, (in-focus-hint,2)
    )

    parser.add_argument(
        "-c",
        "--classification_metric",
        metavar="classification_metric",
        help="use classification metric",
        type=str2bool,
        default=True,
    )

    parser.add_argument(
        '--classifier',
        type=str,
        help='Path to the classifier model',
        required=False,
    )

    parser.add_argument(
        "-nw",
        "--num_workers",
        metavar="num_workers",
        help="number of workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "-pf",
        "--prefetch_factor",
        metavar="prefetch_factor",
        help="prefetch factor",
        type=int,
        default=4,
    )

    parser.add_argument(
        '-ckpt',
        '--checkpoint',
        type=str,
        help='Path to the checkpoint file',
        required=False
    )

    # seed
    parser.add_argument(
        "--seed",
        metavar="seed",
        help="seed",
        type=int,
        default=13,
    )

    # gpu id
    parser.add_argument(
        "--gpuid",
        metavar="gpuid",
        help="gpu id",
        type=int,
        default=0,
    )

    # deterministic
    parser.add_argument(
        "--deterministic",
        metavar="deterministic",
        help="deterministic",
        type=str2bool,
        default=False,
    )
    # benchmark
    parser.add_argument(
        "--benchmark",
        metavar="benchmark",
        help="benchmark",
        type=str2bool,
        default=True,
    )

    # per process memory fraction
    parser.add_argument(
        "--per_process_memory_fraction",
        metavar="per_process_memory_fraction",
        help="per process memory fraction",
        type=float,
        default=1.0,
    )


def add_train_classifier_arguments(parser):
    parser.add_argument(
        "-d",
        "--dataset",
        metavar="dataset",
        help="dataset directory",
        type=pathlib.Path,
        required=True
    )
    parser.add_argument(
        "-s",
        "--structures_of_interest",
        metavar="structures_of_interest",
        help="structures of interest",
        nargs = '+',
        default=["TOMM20", "ACTB", "MYH10", "ACTN1", "LMNB1", "FBL", "NPM1"],
    )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="epochs",
        help="number of epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        metavar="learning_rate",
        help="learning rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "-ps",
        "--patch_shape",
        metavar="patch_shape",
        help="patch shape",
        type=tuple,
        default=(16, 384, 384),
    )
    # patch stride training
    parser.add_argument(
        "-pst",
        "--patch_stride",
        metavar="patch_stride",
        help="patch stride",
        type=tuple,
        default=(16, 384, 384),
    )
    # batch size
    parser.add_argument(
        "-bs",
        "--batch_size",
        metavar="batch_size",
        help="batch size",
        type=int,
        default=2,
    )
    # z_range
    parser.add_argument(
        "-z",
        "--z_range",
        metavar="z_range",
        help="z_range",
        type=str,
        default="in-focus-centre",
    )
    # number of workers
    parser.add_argument(
        "-nw",
        "--num_workers",
        metavar="num_workers",
        help="number of workers",
        type=int,
        default=8,
    )

    # prefetch factor
    parser.add_argument(
        "-pf",
        "--prefetch_factor",
        metavar="prefetch_factor",
        help="prefetch factor",
        type=int,
        default=4,
    )
    # version_number
    parser.add_argument(
        "-v",
        "--version",
        metavar="version",
        help="version",
        type=str,
        default="",
    )

    parser.add_argument(
        '--seed',   # for reproducibility
        type=int,
        default=13,
    )

    parser.add_argument(
        '--gpuid',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--deterministic',
        type=str2bool,
        default=True,
    )

    parser.add_argument(
        '--benchmark',
        type=str2bool,
        default=True,
    )


def add_train_n2v_arguments(parser):
    parser.add_argument(
        "-d",
        "--dataset",
        metavar="dataset",
        help="dataset directory",
        type=pathlib.Path,
        required=True
    )
    parser.add_argument(
        '--DNA',
        metavar='DNA',
        help='train model to denoise DNA channel',
        type=str2bool,
        default=True,
    )

    parser.add_argument(
        '--cm',
        metavar='cm',
        help='train model to denoise cell membrane channel',
        type=str2bool,
        default=True,
    )

    parser.add_argument(
        "-s",
        "--structures_of_interest",
        metavar="structures_of_interest",
        help="structures of interest",
        nargs = '+',
        default=["TOMM20", "ACTB", "MYH10", "ACTN1", "LMNB1", "FBL", "NPM1"],
    )


def add_inference_parser_arguments(parser):
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Path to the model file",
        required=True
    )

    parser.add_argument(
        "-i",
        "--img_path",
        type=str,
        help="Path to the image file",
        required=True
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to the output file",
        required=True
    )

    parser.add_argument(
        "-of",
        "--output_format",
        type=str,
        help="Output format",
        default='tiff'
    )

    parser.add_argument(
        '-s',
        '--section',
        type=int,
        help='Section to select from the input file',
        # default=((24, 0, 0), (40, 512, 512))
    )

    parser.add_argument(
        '-src',
        '--source',
        type=int,
        help='Specify the source channel',
        default=3
    )
