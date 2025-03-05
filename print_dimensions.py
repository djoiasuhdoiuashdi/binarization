import os
import argparse
from PIL import Image


def is_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify that it is, in fact, an image
        return True
    except Exception:
        return False


def print_image_dimensions(folder1, folder2):
    if not os.path.isdir(folder1) or not os.path.isdir(folder2):
        print(f"One or both specified paths are not valid directories.")
        return

    # Get a list of files in both directories
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    # Find common files in both directories
    common_files = files1.intersection(files2)

    # Print dimensions for common files
    print(f"{'Filename':<30} {'Folder 1 Dimensions':<20} {'Folder 2 Dimensions':<20}")
    print("=" * 70)

    for filename in common_files:
        file1_path = os.path.join(folder1, filename)
        file2_path = os.path.join(folder2, filename)

        if os.path.isfile(file1_path) and is_image(file1_path):
            try:
                with Image.open(file1_path) as img1:
                    width1, height1 = img1.size
                    dimensions1 = f"{width1}x{height1}"
            except Exception as e:
                dimensions1 = f"Error: {e}"

        if os.path.isfile(file2_path) and is_image(file2_path):
            try:
                with Image.open(file2_path) as img2:
                    width2, height2 = img2.size
                    dimensions2 = f"{width2}x{height2}"
            except Exception as e:
                dimensions2 = f"Error: {e}"

        print(f"{filename:<30} {dimensions1:<20} {dimensions2:<20}")


def main():
    parser = argparse.ArgumentParser(description="Print dimensions of images in two given folders.")
    parser.add_argument("input_folder1", help="Path to the first folder containing images")
    parser.add_argument("input_folder2", help="Path to the second folder containing images")
    args = parser.parse_args()

    print_image_dimensions(args.input_folder1, args.input_folder2)


if __name__ == "__main__":
    main()
