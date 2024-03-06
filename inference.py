import argparse
import torch
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from aicsimageio.writers.ome_zarr_writer import OmeZarrWriter
from aicsimageio import types

from src.data.aicszarr import zarr_to_input
from src.model import UNet
from src.utils.utils import get_device
from src.utils.parsers import add_inference_parser_arguments


def load_generator(ckp_path):
    try:
        gen_ckp = torch.load(ckp_path)
    except:
        return ValueError(f"Model file {ckp_path} not found")

    gen_hp = gen_ckp['gen_hyperparams']

    ndim = gen_hp['ndim']
    depth = gen_hp['depth']
    mult_chan = gen_hp['mult_chan']
    lr_g = gen_hp['lr_g']
    target_channels = gen_ckp['target_channels']

    # Load the model
    Gen = UNet(
        ndim=ndim,
        activation_fn=torch.nn.LeakyReLU,
        activation_kwargs=(lr_g, True),
        depth=depth,
        n_in_channels=1,
        out_channels=len(target_channels),
        mult_chan=mult_chan,
    )
    # checkpoint = torch.load(ckp_path, map_location=device)
    # model_state_dict = checkpoint['state_dict']
    # model_state_dict = OrderedDict([(k.replace('generator.', ''), v) for k, v in model_state_dict.items() if 'generator' in k])
    # Gen.load_state_dict(model_state_dict)
    Gen.load_state_dict(gen_ckp['gen_state_dict'])
    return Gen, target_channels


def save_output(output, output_format, output_path, target_channels):

    output = output.cpu().numpy()
    if output_format == 'tiff':
        OmeTiffWriter.save(output,
                           f"{output_path}.ome.tiff",
                           dim_order='CZYX',
                           channel_names=target_channels,
                           )
    elif output_format == 'zarr':
        writer = OmeZarrWriter(f"{output_path}.ome.zarr")
        writer.write_image(output,
                           dimension_order='CZYX',
                           image_name=f'{output_path}',
                           channel_names=target_channels,
                           physical_pixel_sizes=types.PhysicalPixelSizes(
                               X=0.108, Y=0.108, Z=0.108),
                           channel_colors=[
                               0, 255, 255*256, 255*256**2, 255*256+255*256**2, 255*256+255, 256**2*255+128*256]
                           )
    else:
        raise ValueError(f"Output format {output_format} not supported")


def inference(img_path, ckp_path, output_path, output_format, section, source, device):

    Gen, target_channels = load_generator(ckp_path)
    Gen = Gen.to(device)

    source = zarr_to_input(img_path, section=section,
                           src_channel=source, device=device)

    Gen.eval()
    with torch.inference_mode():
        output = Gen(source).squeeze(0)

    save_output(output, output_format, output_path, target_channels)

    return output


def main() -> None:

    device = get_device(0)

    parser = argparse.ArgumentParser(
        description="Inference script for the model")
    add_inference_parser_arguments(parser)
    args = parser.parse_args()

    img_path = args.img_path
    ckp_path = args.model
    output_path = args.output
    output_format = args.output_format
    section = args.section
    source = args.source

    inference(img_path, ckp_path, output_path,
              output_format, section, source, device)


if __name__ == "__main__":
    main()
