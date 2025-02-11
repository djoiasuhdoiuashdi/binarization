import os
import cv2
import numpy as np
import doxapy
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.StreamHandler()  # Also log to console
    ]
)

ALGORITHMS = [
    "OTSU",
    "BERNSEN",
    "NIBLACK",
    "SAUVOLA",
    "WOLF",
    "NICK",
    "SU",
    "TRSINGH",
    "BATAINEH",
    "ISAUVOLA",
    "WAN",
    "GATOS"
]

WINDOW_SIZE_MIN = 37
WINDOW_SIZE_MAX =  150
WINDOW_SIZE_STEP = 10

K_MIN = 0.1
K_MAX = 0.4
K_STEP = 0.05

THRESHOLD_MIN = 50
THRESHOLD_MAX = 200
THRESHOLD_STEP = 10

CONTRAST_LIMIT_MIN = 12
CONTRAST_LIMIT_MAX = 50
CONTRAST_LIMIT_STEP = 5

MIN_N_MIN = 37
MIN_N_MAX = 150
MIN_N_STEP = 10

GLYPH_MIN = 30
GLYPH_MAX = 120
GLYPH_STEP = 10


def process_single_image_task(input_image, output_folder, algorithm, param_dict, param_readable):
    if(input_image[1] != "16.bmp"):
        return 
    try:
        logging.info(f"Processing: {input_image[1]}")
        algo = getattr(doxapy.Binarization.Algorithms, algorithm)
        algo = doxapy.Binarization(algo)
        binary_image = np.empty(input_image[0].shape, dtype=np.uint8)
        algo.initialize(input_image[0])
        algo.to_binary(binary_image, param_dict)

        output_filename = f"{input_image[1]}"
        combination_output_path = f"{output_folder}/{algorithm}_{param_readable}"
        os.makedirs(combination_output_path, exist_ok=True)
        output_file_path = os.path.join(combination_output_path, output_filename)
        
        if cv2.imwrite(output_file_path, binary_image):
            logging.info(f"Successfully saved: {output_file_path}")
        else:
            logging.error(f"Failed to save image: {output_file_path}")

    except Exception as e:
        logging.error(f"Error processing {input_image[1]} with {algorithm} ({param_readable}): {e}")
        return f"Error processing {input_image[1]} with {algorithm} ({param_readable}): {e}"

    logging.info(f"Processed {input_image[1]} with {algorithm} ({param_readable}) and saved to {output_file_path}")
    return f"Processed {input_image[1]} with {algorithm} ({param_readable}) and saved to {output_file_path}"

def dict_to_string(input_dict):
    return '_'.join(f"{key}{value}" for key, value in input_dict.items())

def get_param_combinations(algo):
    if algo == "OTSU" or algo == "BATAINEH":
        return [{}]
    
    elif algo == "BERNSEN":
        return [
            {"window": w, "threshold": t, "contrast-limit": cl}
            for w in np.arange(WINDOW_SIZE_MIN, WINDOW_SIZE_MAX + 1, WINDOW_SIZE_STEP)
            for t in np.arange(THRESHOLD_MIN, THRESHOLD_MAX + 1, THRESHOLD_STEP)
            for cl in np.arange(CONTRAST_LIMIT_MIN, CONTRAST_LIMIT_MAX + 1, CONTRAST_LIMIT_STEP)
        ]

    elif algo in ["NIBLACK", "SAUVOLA", "WOLF", "TRSINGH", "ISAUVOLA", "WAN"]:
        return [
            {"window": w, "k": k}
            for w in np.arange(WINDOW_SIZE_MIN, WINDOW_SIZE_MAX + 1, WINDOW_SIZE_STEP)
            for k in [round(k_val, 1) for k_val in np.arange(K_MIN, K_MAX + K_STEP, K_STEP)]
        ]

    elif algo == "NICK":
        return [
            {"window": w, "k": k}
            for w in np.arange(WINDOW_SIZE_MIN, WINDOW_SIZE_MAX + 1, WINDOW_SIZE_STEP)
            for k in [round(k_val, 1) for k_val in np.arange(-K_MAX, -K_MIN + K_STEP, K_STEP)]
        ]

    elif algo == "SU":
        return [
            {"window": w, "minN": n}
            for w in np.arange(WINDOW_SIZE_MIN, WINDOW_SIZE_MAX + 1, WINDOW_SIZE_STEP)
            for n in np.arange(MIN_N_MIN, MIN_N_MAX + 1, MIN_N_STEP)
        ]
    elif algo == "GATOS":
        return [
            {"window": w, "k": k, "glyph": g}
            for w in np.arange(WINDOW_SIZE_MIN, WINDOW_SIZE_MAX + 1, WINDOW_SIZE_STEP)
            for k in [round(k_val, 1) for k_val in np.arange(K_MIN, K_MAX + K_STEP, K_STEP)]
            for g in np.arange(GLYPH_MIN, GLYPH_MAX + 1, GLYPH_STEP)
            
        ]

    return []


def process_images_parallel(input_folder, output_folder, max_workers=1):
 

    reference_images = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    grayscale_images = []
    for filename in reference_images:
        file_path = os.path.join(input_folder, filename)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        grayscale_images.append([img, filename])

    print(f"Loaded {len(grayscale_images)} input images.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        
        
        for algo in ALGORITHMS:
            if algo != "GATOS":
                continue
            i = 0
            param_combinations = get_param_combinations(algo)

            print(f"{algo}: {len(param_combinations)} parameter combinations.")
    
            for param in param_combinations:
                futures = []
                
                param_suffix = dict_to_string(param) if param else "default"
                for image_data in grayscale_images:
                    future = executor.submit(
                        process_single_image_task,
                        image_data,
                        output_folder,
                        algo,
                        param,
                        param_suffix
                    )
                    futures.append(future)

                i+=1
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {i}/{len(param_combinations)}"):
                    try:
                        result = future.result()
    
                    except Exception as e:
                        print(f"Exception: {e}")
                        


    print("All tasks completed.")



if __name__ == "__main__":
    input_folder = "./reference_input"
    output_folder = "./benchmark_input"

    max_workers = 1#os.cpu_count()
    process_images_parallel(input_folder, output_folder, max_workers)
