import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.cp_exr import readEXR, writeEXR
from src.cp_hw2 import read_colorchecker_gm

def select_patches(image):
    plt.imshow(np.clip(image * 38, 0, 1))
    plt.title("select each color patch by clicking on two points in the image")
    patch_data = []
    for i in range(24):
        patch = plt.ginput(2)
        (x1, y1), (x2, y2) = patch
        patch_data.append(((x1, y1), (x2, y2)))
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1)
        plt.gca().add_patch(rect)
        plt.draw()
    plt.close()
    return patch_data

def compute_mean(image, patch_data):
    patch_mean = []
    for idx, ((x1, y1), (x2, y2)) in enumerate(patch_data):
        x1, x2, y1, y2 = map(int, [x1, x2, y1, y2])
        patch = image[y1: y2, x1 : x2, : ]
        rgb_mean = np.mean(patch.reshape(-1, 3), axis=0)
        patch_mean.append(rgb_mean)
    return np.array(patch_mean)

def homegeneous(patch_mean):
    add_col = np.ones((patch_mean.shape[0], 1))
    homogeneous_compute = np.hstack((patch_mean, add_col))
    return homogeneous_compute

def get_ground_truth_rgb():
    ground_truth_rgb = np.array(read_colorchecker_gm())
    ground_truth_rgb = ground_truth_rgb.transpose(1, 2, 0).reshape(-1, 3)
    return ground_truth_rgb

def affine_transformation(measured_rgb, ground_truth_rgb):
    
    print(measured_rgb.shape)
    print(ground_truth_rgb.shape)
    v, _, _, _ = np.linalg.lstsq(measured_rgb, ground_truth_rgb, rcond=None)
    return v

def color_correction(image, error_data):
    height, width, channels = image.shape
    flat_image = image.reshape(-1, 3)
    image_hom = np.hstack((flat_image, np.ones((flat_image.shape[0], 1))))
    color_corrected_image = image_hom @ error_data
    color_corrected_image = color_corrected_image.reshape(height, width, channels)
    return color_corrected_image 

def white_balance_image(corrected_image, patch_coords, white_patch_index):
    x1, y1 = patch_coords[white_patch_index][0]
    x2, y2 = patch_coords[white_patch_index][1]
    x1, x2, y1, y2 = map(int, [x1, x2, y1, y2])
    white_patch = corrected_image[y1:y2, x1:x2, :]
    avg_white = np.mean(white_patch.reshape(-1, 3), axis=0)
    avg_intensity = np.mean(avg_white)
    scale = avg_intensity / avg_white
    wb_image = corrected_image * scale
    wb_image = np.clip(wb_image, 0, None)
    return wb_image

def save_hdr_image(white_balanced_image, filename):
    writeEXR(filename, white_balanced_image, 'FLOAT')

def main():
    image = readEXR("mine/merged_gamma_tiff_update.hdr")
    print(f"image.dtype is {image.dtype}")
    patch_data = select_patches(image)
    patch_mean = compute_mean(image, patch_data)
    measured_rgb = homegeneous(patch_mean)
    ground_truth_rgb = get_ground_truth_rgb()
    error_data = affine_transformation(measured_rgb, ground_truth_rgb)
    color_corrected_image = color_correction(image, error_data)
    # white_balanced_image = white_balance_image(color_corrected_image, patch_data, 18)
    save_hdr_image(0.09*color_corrected_image, "mine/color_corrected_tiff.hdr")

if __name__ == "__main__":
    main()

