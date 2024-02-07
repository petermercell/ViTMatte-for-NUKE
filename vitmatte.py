#https://github.com/hustvl/ViTMatte
#https://github.com/huggingface/transformers

from transformers import VitMatteImageProcessor, VitMatteForImageMatting
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import os
import cv2
import numpy as np
import time

# Initialize the processor and model
processor = VitMatteImageProcessor.from_pretrained("C:/Users/WORKSTATION/miniconda3/envs/HFTF/Lib/site-packages/transformers/models/vitmatte/hustvl/vitmatte-small-composition-1k")
model = VitMatteForImageMatting.from_pretrained("C:/Users/WORKSTATION/miniconda3/envs/HFTF/Lib/site-packages/transformers/models/vitmatte/hustvl/vitmatte-small-composition-1k")

# Specify the directory where the images and trimaps are located
image_directory = "C:/Users/WORKSTATION/Desktop/IMAGE/"
trimap_directory = "C:/Users/WORKSTATION/Desktop/TRIMAP/"

# Specify the directory where you want to save the alpha mattes
save_directory = "C:/Users/WORKSTATION/Desktop/OUTPUT/"

# Create the output directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Get the number of images to process from the user
num_images = int(input("Enter the number of images to process: "))

# Initialize total time
total_time = 0.0

# Loop through the specified number of images
for i in range(1, num_images + 1):
    # Format the image and trimap filenames with leading zeros
    image_filename = f"{i:04d}.png"
    trimap_filename = f"{i:04d}.png"

    # Load the image and trimap from local paths
    image_path = image_directory + image_filename
    trimap_path = trimap_directory + trimap_filename

    image = Image.open(image_path).convert("RGB")
    trimap = Image.open(trimap_path).convert("L")

    # Prepare image + trimap for the model
    inputs = processor(images=image, trimaps=trimap, return_tensors="pt")

    # Run the model to get the alpha matte and measure the time
    start_time = time.time()
    with torch.no_grad():
        alphas = model(**inputs).alphas
    end_time = time.time()
    elapsed_time = end_time - start_time

    total_time += elapsed_time

    print(f"Image {i:04d} rendered in {elapsed_time:.4f} seconds.")
    print(f"Total time so far: {total_time:.4f} seconds")

    # Convert the alphas tensor to a NumPy array
    alphas_numpy = alphas.squeeze().cpu().numpy()

    # Declare alpha_save_path outside the loop
    alpha_save_path = save_directory + f"alphas_{i:04d}.png"

    # Measure the time for saving the alpha matte
    save_start_time = time.time()
    cv2.imwrite(alpha_save_path, (alphas_numpy * 255).astype(np.uint8))
    save_end_time = time.time()
    save_elapsed_time = save_end_time - save_start_time

    total_time += save_elapsed_time

    print(f"Time for saving alpha matte: {save_elapsed_time:.4f} seconds")
    print(f"Alpha matte saved: {alpha_save_path}")

    # Print or save the total time in minutes
    total_time_minutes = total_time / 60
    print(f"Total time for processing {num_images} images: {total_time_minutes:.4f} minutes")
