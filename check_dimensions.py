import os
import argparse
import cv2
import math
import logging

# Global container to hold clicked points
clicked_points = []


def mouse_callback(event, x, y, flags, param):
    global clicked_points, img_for_display
    # When left mouse button is clicked, record the point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        # Draw a small circle at the click location for visual feedback
        cv2.circle(img_for_display, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", img_for_display)


def process_image(image_path):
    global clicked_points, img_for_display
    clicked_points = []  # Reset points for current image

    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return None

    # Clone the image for display
    img_for_display = image.copy()

    # Create a window and set the mouse callback using the clone image
    cv2.imshow("Image", img_for_display)
    cv2.setMouseCallback("Image", mouse_callback)

    print(f"\nProcessing image: {os.path.basename(image_path)}")
    print("Please click two points on the image. Press 'q' to quit early.")

    # Wait for two valid clicks
    while True:
        key = cv2.waitKey(1) & 0xFF

        # Allow the user to quit early if needed
        if key == ord("q"):
            print("Exiting early.")
            return "exit"

        if len(clicked_points) == 2:
            # Calculate Euclidean distance between the two clicked points
            pt1, pt2 = clicked_points
            distance = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
            print(f"Distance between points {pt1} and {pt2}: {distance:.2f} pixels")
            return {
                "filename": os.path.basename(image_path),
                "pt1": pt1,
                "pt2": pt2,
                "distance": distance
            }


def setup_logging(logfile):
    # Configure logging to output to a file with INFO level
    logging.basicConfig(filename=logfile,
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    # Also log a header line for clarity
    logging.info("filename, pt1, pt2, distance")


def main():
    parser = argparse.ArgumentParser(
        description="Click two points on each image to compute and log the distance between them."
    )
    parser.add_argument("input_folder", help="Path to the folder containing images")
    parser.add_argument("--logfile", default="distance_log1.txt",
                        help="Log file to save the computed distances (default: distance_log.txt)")
    args = parser.parse_args()

    # Check if the input folder exists
    if not os.path.isdir(args.input_folder):
        print(f"Error: The folder {args.input_folder} does not exist.")
        return

    # Set up logging
    setup_logging(args.logfile)

    # Define valid image extensions
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
    image_files = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder)
                   if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(args.input_folder, f))]

    if not image_files:
        print("No image files found in the given folder.")
        return

    print(f"Found {len(image_files)} images in the folder.")

    for image_path in image_files:
        result = process_image(image_path)
        if result == "exit":
            break
        elif result is None:
            continue
        else:
            # Log the information in CSV style: filename,pt1,pt2,distance
            logging.info(f"{result['filename']}, {result['pt1']}, {result['pt2']}, {result['distance']:.2f}")
        # Close the current window to move on to the next image
        cv2.destroyWindow("Image")

    cv2.destroyAllWindows()
    print("Finished processing images. Log saved in:", args.logfile)


if __name__ == "__main__":
    main()
