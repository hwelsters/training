import os

from PIL import Image
import numpy as np

from segment_anything_utils import SegmentAnythingSession

# ==============================
# COLORS
# ==============================
FERRIS_WHEEL_COLORS = [(45,  157, 177), (219, 103, 127), (112, 158, 139), (117, 116, 121)]
TREE_COLORS = [(95, 224, 39)]
ROLLER_COASTER_COLORS = [(108, 116, 224)]
CAROUSEL_COLORS = [(117, 93, 91)]

MAX_TRAINING_SET_INDEX = 12
MIN_TEST_SET_INDEX = 17

MIN_INDEX = 0
MAX_INDEX = 9
OUTPUT_DIR = "dataset"

TRAINING_SET_SIZE = 30
TEST_SET_SIZE = 20

def create_directories(target_object_name, weather):
    os.makedirs(f"{OUTPUT_DIR}/{target_object_name}/{weather}/test/masks",    exist_ok=True) 
    os.makedirs(f"{OUTPUT_DIR}/{target_object_name}/{weather}/test/images",   exist_ok=True) 
    os.makedirs(f"{OUTPUT_DIR}/{target_object_name}/{weather}/train/masks",   exist_ok=True) 
    os.makedirs(f"{OUTPUT_DIR}/{target_object_name}/{weather}/train/images",  exist_ok=True) 
    os.makedirs(f"{OUTPUT_DIR}/{target_object_name}/train/masks",   exist_ok=True) 
    os.makedirs(f"{OUTPUT_DIR}/{target_object_name}/train/images",  exist_ok=True) 
    os.makedirs(f"{OUTPUT_DIR}/{target_object_name}/test/masks",   exist_ok=True) 
    os.makedirs(f"{OUTPUT_DIR}/{target_object_name}/test/images",  exist_ok=True) 
    os.makedirs(f"{OUTPUT_DIR}/test/images",  exist_ok=True) 
    os.makedirs(f"{OUTPUT_DIR}/test/masks",  exist_ok=True) 
    os.makedirs(f"{OUTPUT_DIR}/train/images",  exist_ok=True) 
    os.makedirs(f"{OUTPUT_DIR}/train/masks",  exist_ok=True) 

for target_object_name, target_object_colors in [
    ("ferris_wheel", FERRIS_WHEEL_COLORS),
    ("tree", TREE_COLORS),
    ("roller_coaster", ROLLER_COASTER_COLORS),
    ("carousel", CAROUSEL_COLORS),
]:
    for weather in [
        "dust-0.4",
        "mapleleaf-0.4",
        "normal",
    ]:
        create_directories(target_object_name, weather)
        count = 0

        for sample_index in range(10, 19):
            for rotation in [0, 90, 180, 270]:
                for index in range(MIN_INDEX, MAX_INDEX):
                    input_image_path = f"images/{weather}/sample_{sample_index}/rotation_{rotation}/{index}/Scene.png"
                    ground_truth_segmentation_path = f"images/{weather}/sample_{sample_index}/rotation_{rotation}/{index}/Segmentation.png"

                    filtered_colors = SegmentAnythingSession.filter_colors(ground_truth_segmentation_path, target_object_colors)


                    is_in_training_set = count < TRAINING_SET_SIZE
                    is_in_test_set = count >= TRAINING_SET_SIZE

                    if filtered_colors.sum() < 10: continue
                    if filtered_colors.sum() < 10 and is_in_training_set: 
                        is_in_training_set = False
                        is_in_test_set = True

                    if count >= TRAINING_SET_SIZE + TEST_SET_SIZE:
                        continue

                    image_path = ""
                    mask_path = ""
                    big_image_path = ""
                    big_mask_path = ""
                    bigus_imageus_pathus = ""
                    bigus_maskus_pathus = ""

                    if is_in_training_set:
                        image_path = f"{OUTPUT_DIR}/{target_object_name}/{weather}/train/images"
                        mask_path = f"{OUTPUT_DIR}/{target_object_name}/{weather}/train/masks"
                        big_image_path = f"{OUTPUT_DIR}/{target_object_name}/train/images/{weather}_{sample_index}_{rotation}_{index}.jpg"
                        big_mask_path = f"{OUTPUT_DIR}/{target_object_name}/train/masks/{weather}_{sample_index}_{rotation}_{index}.jpg"
                        bigus_imageus_pathus = f"{OUTPUT_DIR}/train/images/{weather}_{sample_index}_{rotation}_{index}.jpg"
                        bigus_maskus_pathus = f"{OUTPUT_DIR}/train/masks/{weather}_{sample_index}_{rotation}_{index}.jpg"
                    elif is_in_test_set:
                        image_path = f"{OUTPUT_DIR}/{target_object_name}/{weather}/test/images"
                        mask_path = f"{OUTPUT_DIR}/{target_object_name}/{weather}/test/masks"
                        big_image_path = f"{OUTPUT_DIR}/{target_object_name}/test/images/{weather}_{sample_index}_{rotation}_{index}.jpg"
                        big_mask_path = f"{OUTPUT_DIR}/{target_object_name}/test/masks/{weather}_{sample_index}_{rotation}_{index}.jpg"
                        bigus_imageus_pathus = f"{OUTPUT_DIR}/test/images/{weather}_{sample_index}_{rotation}_{index}.jpg"
                        bigus_maskus_pathus = f"{OUTPUT_DIR}/test/masks/{weather}_{sample_index}_{rotation}_{index}.jpg"

                    image = Image.open(input_image_path)
                    mask = Image.fromarray(np.uint8(filtered_colors) * 255)

                    image_rgb = image.convert("RGB")
                    mask_rgb = mask.convert("RGB")

                    image_rgb = image_rgb.resize((256, 144))
                    mask_rgb = mask_rgb.resize((256, 144))

                    image_rgb.save(f"{image_path}/{sample_index}_{rotation}_{index}.jpg")
                    mask_rgb.save(f"{mask_path}/{sample_index}_{rotation}_{index}.jpg")
                    image_rgb.save(big_image_path)
                    mask_rgb.save(big_mask_path)
                    image_rgb.save(bigus_imageus_pathus)
                    mask_rgb.save(bigus_maskus_pathus)

                    count += 1
