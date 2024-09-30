import numpy as np
import cv2
import os
import Linearize_Rendered_Images
from src.cp_exr import writeEXR

def initialize_hdr_image(shape, dtype=np.float32):
    numerator = np.zeros(shape, dtype)
    denominator = np.zeros(shape, dtype)
    return numerator, denominator

def save_hdr_image(hdr_image, file_name):
    writeEXR(file_name, hdr_image, pixeltype='FLOAT')

def gamma(image):
    gamma_corrected_image = np.where(image <= 0.0031308, 2.92 * image, (1 + 0.055) * (image**(1 / 1.4)) - 0.055)
    gamma_corrected_image = np.clip(gamma_corrected_image, 0, 1)
    return gamma_corrected_image

def main():
    folder_path = "mine"
    image_type = 'tiff' 
    weight_type = 'tent' 
    merge_method = 'linear'
    g_file = f"g_response_{weight_type}.npy"
    files = [file for file in os.listdir(folder_path) if file.lower().endswith(image_type)]
    files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    num_images = len(files)
    exposure_times = Linearize_Rendered_Images.get_exposure_times(num_images)
    numerator = None
    denominator = None
    over_exposed_mask = None  
    under_exposed_mask = None  

    for idx, file in enumerate(files):
        print(f"Processing image {idx+1}/{num_images}: {file}")
        path = os.path.join(folder_path, file)
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image_type == 'tiff':
            I_LDR = image.astype(np.float32) / (2**16 - 1)
            I_lin = I_LDR  
        elif image_type == 'jpg':
            I_LDR = image.astype(np.float32) / 255.0
            g = np.load(g_file) 
            I_lin = np.exp(g[image.astype(np.uint8)])

        tk = exposure_times[idx]

        weight = Linearize_Rendered_Images.weight_function(weight_type, I_LDR, tk=tk if weight_type == 'photon' else None)

        Imax = 0.9
        Imin = 0.1

        over_exposed = (I_LDR >= Imax)  
        under_exposed = (I_LDR <= Imin)

        if over_exposed_mask is None:
            over_exposed_mask = over_exposed
            under_exposed_mask = under_exposed
        else:
            over_exposed_mask |= over_exposed  
            under_exposed_mask |= under_exposed

        if numerator is None and denominator is None:
          numerator, denominator = initialize_hdr_image(I_LDR.shape)

        if merge_method == 'linear':
            numerator += weight * I_lin / tk
            denominator += weight
        elif merge_method == 'logarithmic':
            epsilon = 1e-8
            numerator += weight * (np.log(I_lin + epsilon) - np.log(tk))
            denominator += weight

    print("Finalizing HDR image...")
    zero_weight_mask = (denominator == 0)
    hdr_image = np.zeros_like(numerator)

    non_zero_mask = ~zero_weight_mask
    if merge_method == 'linear':
        hdr_image[non_zero_mask] = numerator[non_zero_mask] / denominator[non_zero_mask]
    elif merge_method == 'logarithmic':
        ln_hdr_image = np.zeros_like(numerator)
        ln_hdr_image[non_zero_mask] = numerator[non_zero_mask] / denominator[non_zero_mask]
        hdr_image[non_zero_mask] = np.exp(ln_hdr_image[non_zero_mask])

    hdr_min = hdr_image[non_zero_mask].min() if non_zero_mask.any() else 0
    hdr_max = hdr_image[non_zero_mask].max() if non_zero_mask.any() else 1

    hdr_image[zero_weight_mask & over_exposed_mask] = hdr_max

    hdr_image[zero_weight_mask & under_exposed_mask] = hdr_min

    gamma_hdr_image = gamma(hdr_image)

    save_hdr_image(hdr_image, f"mine/merged_gamma_tiff_update.hdr")
    print("HDR image saved.")


if __name__ == "__main__":
    main()



