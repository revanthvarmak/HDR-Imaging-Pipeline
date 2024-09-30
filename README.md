# HDR Image Processing
This project includes multiple Python scripts for processing and rendering High Dynamic Range (HDR) images. The tasks include linearizing rendered images, merging exposure stacks, color correction, and tone mapping.

## Requirements
Python 3.x
NumPy
OpenCV (cv2)
Matplotlib
Scikit-Image (skimage)
src.cp_exr module for reading and writing EXR files
src.cp_hw2 for reading color checker data

### Install the required Python packages:
pip install numpy opencv-python matplotlib scikit-image

## Scripts Overview
### 1. Linearize_Rendered_Images.py
This script reads a sequence of rendered images, computes a response curve using a least squares method, and then linearizes the images.

Key Functions:
read_images(folder_path, image_type): Reads all images of a specified type from the folder.
get_exposure_times(num_images): Generates exposure times for each image.
weight_function(weight_type, I, tk): Calculates weight values for pixels based on different types such as uniform, tent, gaussian, or photon.
matrix_definition(images, exposure_time, weight_type, lambda_value): Constructs the system of linear equations needed for computing the response curve.
least_sqaures(A, b): Solves the system of equations using least squares to obtain the response curve.
linearize(g, folder_path, image_type): Linearizes the image intensities based on the computed response curve.
plot(g): Plots the response curve.

### 2. Merge_Exposure_Stack.py
This script merges a stack of images taken at different exposures to produce a single HDR image.

Key Functions:
initialize_hdr_image(shape, dtype): Initializes numerator and denominator arrays for HDR computation.
save_hdr_image(hdr_image, file_name): Saves the HDR image using the writeEXR function.
gamma(image): Applies gamma correction to the image.
main(): Reads the linearized images, merges them using either linear or logarithmic weighting, and saves the resulting HDR image.

### 3. Color_Correction.py
This script corrects the colors of the HDR image based on a color checker reference.

Key Functions:
select_patches(image): Prompts the user to select color patches from the displayed image.
compute_mean(image, patch_data): Computes the mean RGB value of each color patch.
homegeneous(patch_mean): Converts patch RGB values to homogeneous coordinates.
affine_transformation(measured_rgb, ground_truth_rgb): Computes the affine transformation matrix between measured RGB values and ground truth RGB values.
color_correction(image, error_data): Applies color correction to the HDR image.
save_hdr_image(white_balanced_image, filename): Saves the corrected image as an HDR file.

### 4. Tone_Mapping.py
This script applies tone mapping to the HDR image to make it suitable for display on standard monitors.

Key Functions:
photographic_tonemapping_rgb(image, K, B): Applies photographic tone mapping to the RGB channels of the image.
photographic_tonemapping_luminance(image, K, B): Applies photographic tone mapping to the luminance channel of the image, preserving color relationships.

Usage
Modify the folder paths and parameters inside the scripts (main() function) to point to your image data.
Run each script in sequence to perform HDR processing from linearization to tone mapping.
Notes
Ensure that the folder paths specified in each script are correct for your file structure.
The provided code uses .jpg and .tiff images for linearization and merging.
The src directory must contain the cp_exr.py and cp_hw2.py modules for reading and writing EXR files and color checker data.

## Author
Revanth Varma

