from ast import main
import numpy as np
import matplotlib.pyplot as plt
from src.cp_exr import readEXR, writeEXR
from src.cp_hw2 import XYZ2lRGB, lRGB2XYZ
from skimage import color

def photographic_tonemapping_rgb(image, K, B):
    height, width, channels = image.shape
    image = image.reshape(-1, 3)
    N = height * width
    Im_hdr = np.exp((1 / N) * np.sum(np.log(image+10e-8), axis= 0))
    Imean_hdr = K * image / Im_hdr
    I_white = B*np.max(Imean_hdr)
    I_tonemap = (Imean_hdr * (1 + Imean_hdr / I_white**2)) / (1 + Imean_hdr)
    I_tonemap = np.clip(I_tonemap.reshape(height, width, channels), 0, 1)
    writeEXR(f"mine/tonemapped_rgb_new_tiff.hdr", I_tonemap, 'FLOAT')
    return I_tonemap

def photographic_tonemapping_luminance(image, K, B):
    image_XYZ = lRGB2XYZ(image)
    Y = image_XYZ[:, :, 1]
    height, width = Y.shape
    Y = Y.reshape(height*width)
    N = height*width
    Im_hdr = np.exp((1 / N) * np.sum(np.log(Y+10e-8), axis= 0))
    Imean_hdr = K * Y / Im_hdr
    I_white = B*np.max(Imean_hdr)
    Y_tonemap = (Imean_hdr * (1 + Imean_hdr / I_white**2)) / (1 + Imean_hdr)
    Y_tonemap = np.clip(Y_tonemap.reshape(height, width), 0, 1)
    scale = Y_tonemap / (Y.reshape(height, width) + 1e-8)  
    image_XYZ[:, :, 0] *= scale
    image_XYZ[:, :, 2] *= scale
    image_XYZ[:, :, 1] = Y_tonemap
    image_rgb = XYZ2lRGB(image_XYZ)
    writeEXR(f"mine/tonemapped_luminance_dcraw.hdr", image_rgb, 'FLOAT')
    return image_rgb

def main():
    image = readEXR("mine/merged_gamma_tiff_update.hdr")
    K = 0.09
    B = 0.7
    photographic_tonemapping_rgb(image, K, B)

    

if __name__ == "__main__":
    main()



