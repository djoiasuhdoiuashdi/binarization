import argparse
import os
import shutil

def rename_gt_files(base_path):
    images_gt_path = os.path.join(base_path, 'images_gt')
    images_path = os.path.join(base_path, 'images')
    if not os.path.isdir(images_gt_path):
        raise FileNotFoundError(f"Directory '{images_gt_path}' does not exist.")

    for filename in os.listdir(images_gt_path):
        if filename.endswith('_GT.tiff'):
            new_filename = filename.replace('_GT.tiff', '_gt.png')
            old_file = os.path.join(images_gt_path, filename)
            new_file = os.path.join(base_path, new_filename)
            shutil.move(old_file, new_file)

    for filename in os.listdir(images_path):
        if filename.endswith('.tiff'):
            new_filename = filename.replace('.tiff', '_in.png')
            old_file = os.path.join(images_path, filename)
            new_file = os.path.join(base_path, new_filename)
            shutil.move(old_file, new_file)

    os.rmdir(images_gt_path)
    os.rmdir(images_path)


def main():
    parser = argparse.ArgumentParser(description="Rename _GT.tiff files to .tiff in images_gt folder.")
    parser.add_argument('path', type=str, help='Path containing images and images_gt folders')
    args = parser.parse_args()

    rename_gt_files(args.path)


if __name__ == "__main__":
    main()