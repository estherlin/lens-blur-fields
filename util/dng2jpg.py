# @brief: Converts a directory of DNG files to JPG files using rawpy and imageio
# @author: @estherlin
# @date: 2020-05-05
# @usage: python dng2jpg.py --input_dir path/to/dngs --output_dir path/to/jpgs
# @requirements: rawpy, imageio, glob, argparse
# @license: MIT


import os
import glob
import rawpy
import imageio
import argparse

def convert_dng_to_jpg(input_dng, output_jpg):
    """Converts a DNG file to JPG using rawpy and imageio"""
    with rawpy.imread(input_dng) as raw:
        rgb = raw.postprocess()
    imageio.imwrite(output_jpg, rgb)

def process_directory(input_dir, output_dir):
    dng_files = glob.glob(os.path.join(input_dir, '**', '*.[dD][nN][gG]'), recursive=True)
    
    for input_dng in dng_files:
        output_jpg = os.path.join(output_dir, os.path.splitext(os.path.relpath(input_dng, input_dir))[0] + '.jpg')
        os.makedirs(os.path.dirname(output_jpg), exist_ok=True)
        convert_dng_to_jpg(input_dng, output_jpg)
        print(f"Converted {input_dng} to {output_jpg}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DNG to JPG Converter')
    parser.add_argument("--input_dir", required=True, help="Input directory containing DNG files")
    parser.add_argument("--output_dir", required=True, help="Output directory for converted JPG files")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)
    print("Done!")