import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from src.cp_exr import writeEXR

def read_images(folder_path, image_type):
    images = []
    files = [file for file in os.listdir(folder_path) if file.lower().endswith(image_type)]
    files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    for file in files:
        path = os.path.join(folder_path, file)
        image = cv2.imread(path)[::200, ::200]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return np.stack(images, axis=-1)

def get_exposure_times(num_images):
    return [ (1 * 2 ** (k - 1)) / 2048 for k in range(1, num_images + 1) ]

def weight_function(weight_type, I, tk, Imin = 0.1, Imax = 0.9):
    if weight_type == "uniform":
        return np.where((I <= Imax) & (I >= Imin), 1, 0)
    elif weight_type == "tent":
        return np.where((I <= Imax) & (I >= Imin), np.minimum(I, 1 - I), 0)
    elif weight_type == "gaussian":
        return np.where((I <= Imax) & (I >= Imin), np.exp(-16 * (I - 0.5)**2), 0)    
    elif weight_type == "photon":
        return np.where((I <= Imax) & (I >= Imin), tk, 0)

def matrix_definition(images, exposure_time, weight_type, lambda_value):
    num_pixels = images.shape[0] * images.shape[1] * images.shape[2]
    num_images = images.shape[3]
    num_intensities = 256
    flat_images = images.reshape((-1, num_images))
    normalized_intensities = flat_images / 255.0
    image_intensities = flat_images.astype(np.uint8)
    num_rows = (num_pixels * num_images) + num_intensities - 2
    num_cols = num_pixels + num_intensities
    A = np.zeros((num_rows, num_cols))
    b = np.zeros(num_rows)
    for i in range(num_images):
        tk = exposure_time[i]
        row_indices = np.arange(num_pixels * i, num_pixels * (i + 1))
        weights = weight_function(weight_type, normalized_intensities[:, i], tk)
        A[row_indices, image_intensities[:, i]] = weights
        A[row_indices, num_intensities + np.arange(num_pixels)] = -weights
        b[row_indices] = np.log(exposure_time[i]) * weights

    for i in range(1, num_intensities - 1):
        A[num_pixels * num_images + i - 1, i - 1] = lambda_value
        A[num_pixels * num_images + i - 1, i] = -2 * lambda_value
        A[num_pixels * num_images + i - 1, i + 1] = lambda_value
    
    return A, b

def least_sqaures(A, b):
    v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    g = v[:256]
    logL = v[256:]
    return g, logL

def linearize(g, folder_path, image_type):
    files = [file for file in os.listdir(folder_path) if file.lower().endswith(image_type)]
    files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    for idx, file in enumerate(files):
        path = os.path.join(folder_path, file)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_int = np.clip(image, 0, 255).astype(int)
        I_lin = np.exp(g[image_int])
        writeEXR(f"mine/linearized_image{idx}.hdr", I_lin, 'FLOAT')

        

def plot(g):
    x = np.arange(256)
    plt.plot(x, g)
    plt.title("Response Curve")
    plt.xlabel("pixel intensities")
    plt.ylabel("g(I) Log Radiance")
    # plt.savefig("/content/drive/MyDrive/assgn2/response.png")
    plt.savefig("mine/response.png")
    plt.show()

def main():
    # folder_path = "/content/drive/MyDrive/assgn2/data/door_stack"
    folder_path = "/Users/revanthvarma/Desktop/15663_CP/assgn2/mine/"
    images = read_images(folder_path, image_type = 'jpg')
    num_images = images.shape[-1]
    exposure_times = get_exposure_times(num_images)
    print(exposure_times)
    weight_type = "tent"
    A, b = matrix_definition(images, exposure_times, weight_type, lambda_value=150)
    g, logL = least_sqaures(A, b)
    np.save(f"mine/g_response_{weight_type}.npy", g)
    plot(g)  
    linear_exposure = linearize(g, folder_path, image_type='jpg')


if __name__ == "__main__":
    main()