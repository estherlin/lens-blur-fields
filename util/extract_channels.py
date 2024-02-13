# @author: estherlin
# @date: 2023-03-28
# @usage: python extract_channels.py --input_dir path/to/dngs --output_dir path/to/channels
# @requirements: rawpy, imageio, glob, argparse, numpy
# @license: MIT


import os
import glob
import rawpy
import imageio
import argparse
import numpy as np
from printarr import printarr

def extract_color_channels(raw_mosaic):
    raw_mosaic /= raw_mosaic.max()
    channels = [
        raw_mosaic[0::2, 0::2],  # Red
        raw_mosaic[0::2, 1::2],  # Green 1
        raw_mosaic[1::2, 0::2],  # Green 2
        raw_mosaic[1::2, 1::2],  # Blue

    ]
    return channels

def process_dng_files(input_dir, output_dirs):
    dng_files = glob.glob(os.path.join(input_dir, '**', '*.[dD][nN][gG]'), recursive=True)

    for dng_file in dng_files:
        with rawpy.imread(dng_file) as raw:
            raw_mosaic = np.float32(raw.raw_image.copy())
            channels = extract_color_channels(raw_mosaic)
            printarr(raw_mosaic, channels[0], channels[1], channels[2], channels[3])

        for i, channel in enumerate(channels):
            output_dir = output_dirs[i]
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(dng_file))[0]}_channel_{i}.png")
            imageio.imwrite(output_filename, np.uint8(255*channel))
            print(f"Saved channel {i} to {output_filename}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DNG to PNG Converter')
    parser.add_argument("--input_dir", required=True, help="Input directory containing DNG files")
    parser.add_argument("--output_dir", required=True, help="Output directory for converted PNG files")
    args = parser.parse_args()

    input_dir = args.input_dir  # Replace with your input directory containing DNG files
    output_dirs = [
        f"{args.output_dir}/r", 
        f"{args.output_dir}/g1", 
        f"{args.output_dir}/g2", 
        f"{args.output_dir}/b"
    ]

    process_dng_files(input_dir, output_dirs)
