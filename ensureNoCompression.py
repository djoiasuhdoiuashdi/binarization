import os
from PIL import Image

def is_tiff_uncompressed(tiff_file):
    with Image.open(tiff_file) as img:
        return img.info.get('compression', 'none') == 'none'

def convert_to_uncompressed(tiff_file):
    with Image.open(tiff_file) as img:
        img.save(tiff_file, format='TIFF', compression='none')
        print(f"Converted {tiff_file} to uncompressed format: {tiff_file}")

def process_tiff_files(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.tif', '.tiff')):
            tiff_file = os.path.join(directory, filename)
            if not is_tiff_uncompressed(tiff_file):
                print(f"{tiff_file} is compressed. Converting to uncompressed...")
                convert_to_uncompressed(tiff_file)
            else:
                print(f"{tiff_file} is already uncompressed.")

if __name__ == "__main__":
    directory_path = input("Enter the directory path containing TIFF files: ")
    process_tiff_files(directory_path)
