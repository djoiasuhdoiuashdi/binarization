import argparse
import os


def rename_gt_files(base_path):
    images_gt_path = os.path.join(base_path, 'images_gt')
    if not os.path.isdir(images_gt_path):
        raise FileNotFoundError(f"Directory '{images_gt_path}' does not exist.")

    for filename in os.listdir(images_gt_path):
        if filename.endswith('_GT.tiff'):
            new_filename = filename.replace('_GT.tiff', '.tiff')
            old_file = os.path.join(images_gt_path, filename)
            new_file = os.path.join(images_gt_path, new_filename)
            os.rename(old_file, new_file)

    os.rename(os.path.join(base_path, 'images_gt'), os.path.join(base_path, 'gt'))
    os.rename(os.path.join(base_path, 'images'), os.path.join(base_path, 'img'))


def main():
    parser = argparse.ArgumentParser(description="Rename _GT.tiff files to .tiff in images_gt folder.")
    parser.add_argument('path', type=str, help='Path containing images and images_gt folders')
    args = parser.parse_args()

    rename_gt_files(args.path)


if __name__ == "__main__":
    main()