import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def get_gaussian(s: tuple, sigma=1.0 / 8) -> np.ndarray:
    temp = np.zeros(s)
    coords = [i // 2 for i in s]  # 寻找中心坐标
    sigmas = [i * sigma for i in s]  # 输出每个维度的sigma值
    temp[tuple(coords)] = 1  # 将中心坐标的值赋为1,然后用高斯滤波使图像变模糊
    gaussian_map = gaussian_filter(temp, sigmas, 0, mode='constant', cval=0)
    gaussian_map /= np.max(gaussian_map)
    return gaussian_map


def patchify(ID: tuple, img: np.ndarray, mask: np.ndarray, patchSize: tuple, patchStride: tuple,
             SavingPath='0') -> list:
    patch_list = []
    for i in range(0, img.shape[0] - patchSize[0] + 1, patchStride[0]):
        for j in range(0, img.shape[1] - patchSize[1] + 1, patchStride[1]):
            for k in range(0, img.shape[2] - patchSize[2] + 1, patchStride[2]):
                img_patch = img[i:i + patchSize[0], j:j + patchSize[1], k:k + patchSize[2]].astype(np.float32)
                mask_patch = mask[i:i + patchSize[0], j:j + patchSize[1], k:k + patchSize[2]].astype(np.float32)
                direction = (i, j, k)
                patch_list.append([ID, direction, img_patch, mask_patch])
                if SavingPath != '0':
                    img_patch = sitk.GetImageFromArray(img_patch)
                    mask_patch = sitk.GetImageFromArray(mask_patch)
                    # todo:should be check
                    sitk.WriteImage(img_patch, SavingPath + str(ID[0]) + str(direction) + '/nii.gz')
                    sitk.WriteImage(mask_patch, SavingPath + str(ID[0]) + str(direction) + '/nii.gz')
    return patch_list



def unpatchify(ID: tuple, output: np.ndarray, ground_truth: np.ndarray, patchSize: tuple, patchStride: tuple,
               IsSave: False) -> np.ndarray:

    return np.zeros((4, 4))



if __name__ == '__main__':
    img = sitk.ReadImage('data/image/nii/sub-verse596_dir-ax_ct.nii.gz')
    img = sitk.GetArrayFromImage(img)
    patchSize = np.array((64, 64, 64))
    patchStride = patchSize // 2

    result = np.zeros(img.shape)
    normalization = np.zeros(img.shape)
    gaussian_map = get_gaussian(patchSize)
    for i in range(0, img.shape[0] - patchSize[0] + 1, patchStride[0]):
        for j in range(0, img.shape[1] - patchSize[1] + 1, patchStride[1]):
            for k in range(0, img.shape[2] - patchSize[2] + 1, patchStride[2]):
                patch = img[i:i + patchSize[0], j:j + patchSize[1], k:k + patchSize[2]].astype(np.float32)
                patch *= gaussian_map
                normalization[i:i + patchSize[0], j:j + patchSize[1], k:k + patchSize[2]] += gaussian_map
                result[i:i + patchSize[0], j:j + patchSize[1], k:k + patchSize[2]] += patch
    result /= normalization
    result = sitk.GetImageFromArray(result)
    sitk.WriteImage(result, 'data/image/preprocess/AFTER_PATCH.nii.gz')

    aaa = get_gaussian((64, 64, 64), 0.25)  # (C,H,W)
    fig = plt.figure()
    plt.imshow(aaa)
