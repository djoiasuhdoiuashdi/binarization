#!/usr/bin/env python3
"""
scale_images.py

This script reads a log file with entries like:
    2025-02-27 16:32:13 - Abraamios_1.jpg, (1001, 513), (1005, 419), 94.09

It uses the last number (the distance value) in each log entry to compute a scaling factor
so that this distance becomes the same number of pixels (specified by the user) in the scaled image.
Each image file (by filename) is read from the input directory and then saved in the output directory
after scaling.
"""

import argparse
import os
import sys
from PIL import Image

def parse_log_file(logfile_path):
    """
    Parses the log file to create a dictionary mapping filenames to distance values.

    Returns:
        dict: key is the filename (str) and value is the distance (float).
    """
    mapping = {}
    with open(logfile_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Expecting a line format:
            # 2025-02-27 16:32:13 - Abraamios_1.jpg, (1001, 513), (1005, 419), 94.09
            try:
                # Remove the timestamp part
                timestamp_sep = " - "
                if timestamp_sep not in line:
                    continue
                parts = line.split(timestamp_sep, 1)
                data = parts[1].strip()
                segments = data.split(',')
                # The first segment is the filename; the last segment is the distance
                if len(segments) < 4:
                    continue
                filename = segments[0].strip()
                distance_str = segments[-1].strip()
                distance = float(distance_str)
                mapping[filename] = distance
            except Exception as e:
                print(f"Error parsing line: {line}\n{e}", file=sys.stderr)
                continue
    return mapping

def scale_image(image, scale_factor):
    """
    Scales the image by the given scale_factor.

    Args:
        image (PIL.Image.Image): The image to scale.
        scale_factor (float): The factor by which to scale the image dimensions.

    Returns:
        PIL.Image.Image: The scaled image.
    """
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    return image.resize((new_width, new_height), Image.LANCZOS)

def main():
    parser = argparse.ArgumentParser(
        description="Scale images so that the distance in the log file equals a target pixel distance."
    )
    parser.add_argument("--logfile", required=True, help="Path to the input log file")
    parser.add_argument("--inputdir", required=True, help="Directory containing input images")
    parser.add_argument("--outputdir", required=True, help="Directory to save scaled images")
    parser.add_argument("--target", type=float, required=True, help="Target pixel distance for the log file distance")

    args = parser.parse_args()

    # Validate arguments
    if not os.path.isfile(args.logfile):
        print(f"Error: Log file '{args.logfile}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.inputdir):
        print(f"Error: Input directory '{args.inputdir}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.outputdir):
        os.makedirs(args.outputdir, exist_ok=True)

    # Parse log file to get mapping from filename to distance value.
    log_mapping = parse_log_file(args.logfile)

    # Process each image file in the input directory using the mapping.
    for fname in os.listdir(args.inputdir):
        fpath = os.path.join(args.inputdir, fname)
        if not os.path.isfile(fpath):
            continue

        # Check if we have log data for this file.
        if fname not in log_mapping:
            print(f"Warning: File '{fname}' not found in log mapping. Skipping...", file=sys.stderr)
            continue

        try:
            # Open image
            with Image.open(fpath) as img:
                recorded_distance = log_mapping[fname]
                # Compute scale factor so that recorded_distance becomes args.target pixels.
                # scale_factor = target_pixels / recorded_distance.
                scale_factor = args.target / recorded_distance

                # Scale image
                scaled_img = scale_image(img, scale_factor)

                # Save the resulting image in the output directory with the same name.
                out_path = os.path.join(args.outputdir, fname)
                scaled_img.save(out_path)
                print(f"Processed '{fname}': scale factor = {scale_factor:.4f} -> Saved to '{out_path}'")
        except Exception as e:
            print(f"Error processing '{fname}': {e}", file=sys.stderr)
            continue

if __name__ == "__main__":
    main()
