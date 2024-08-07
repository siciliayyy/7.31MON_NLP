import re

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

nii_path = 'result/11.3featuremap/11.3_picture/'
featuremap_path = 'result/11.3featuremap/211/'


def show_slices(img_ID, direction, img: np.ndarray, mask: np.ndarray, DC, AUC, Show_nii: bool):
    img_ID = re.sub('[\"\'\(\),]', '', str(img_ID))
    img_size = img.shape
    mask_size = img.shape
    if Show_nii:
        img_ = sitk.GetImageFromArray(img)
        sitk.WriteImage(img_, nii_path + '{}.nii.gz'.format(str(img_ID)))
        mask_ = sitk.GetImageFromArray(mask)
        sitk.WriteImage(mask_, nii_path + '{}_mask.nii.gz'.format(str(img_ID)))

    slice_0 = np.flip(img[img_size[0] // 2, :, :])
    slice_3 = np.flip(mask[mask_size[0] // 2, :, :])

    slice_1 = np.flip(img[:, img_size[1] // 2, :])
    slice_4 = np.flip(mask[:, mask_size[1] // 2, :])

    slice_2 = np.flip(img[:, :, img_size[2] // 2])
    slice_5 = np.flip(mask[:, :, mask_size[2] // 2])
    fig = plt.figure()
    ax_0 = fig.add_subplot(2, 3, 1)
    ax_0.set_title('')
    plt.imshow(slice_0, cmap="gray", origin="lower")
    ax_1 = fig.add_subplot(2, 3, 2)
    ax_1.set_title('')
    plt.imshow(slice_1, cmap="gray", origin="lower")
    ax_2 = fig.add_subplot(2, 3, 3)
    ax_2.set_title('')
    plt.imshow(slice_2, cmap="gray", origin="lower")
    ax_3 = fig.add_subplot(2, 3, 4)
    ax_3.set_title('')
    plt.imshow(slice_3, cmap="gray", origin="lower")
    ax_4 = fig.add_subplot(2, 3, 5)
    ax_4.set_title('')
    plt.imshow(slice_4, cmap="gray", origin="lower")
    ax_5 = fig.add_subplot(2, 3, 6)
    ax_5.set_title('')
    plt.imshow(slice_5, cmap="gray", origin="lower")

    plt.suptitle(
        "ID={}__{:.4f}\n{}_{}_{}\nDC={:.4f}__AUC={:.4f}".format(str(img_ID), np.sum(mask > 0) / (64 * 64 * 64),
                                                                str(direction[0]),
                                                                str(direction[1]), str(direction[2]), DC, AUC))
    plt.savefig(featuremap_path + 'ID{}__{}_{}_{}.png'.format(str(img_ID), str(direction[0]), str(direction[1]),
                                                              str(direction[2])), dpi=600)


def show_slice(img_ID, direction, img: np.ndarray, Show_nii: bool):
    img_ID = re.sub('[\"\'\(\),]', '', str(img_ID))
    img_size = img.shape

    if Show_nii:
        img_ = sitk.GetImageFromArray(img)
        sitk.WriteImage(img_, nii_path + '{}.nii.gz'.format(str(img_ID)))


    slice_0 = np.flip(img[img_size[0] // 2, :, :])

    slice_1 = np.flip(img[:, img_size[1] // 2, :])

    slice_2 = np.flip(img[:, :, img_size[2] // 2])

    fig = plt.figure()
    ax_0 = fig.add_subplot(1, 3, 1)
    ax_0.set_title('')
    plt.imshow(slice_0, cmap="gray", origin="lower")
    ax_1 = fig.add_subplot(1, 3, 2)
    ax_1.set_title('')
    plt.imshow(slice_1, cmap="gray", origin="lower")
    ax_2 = fig.add_subplot(1, 3, 3)
    ax_2.set_title('')
    plt.imshow(slice_2, cmap="gray", origin="lower")

    plt.suptitle(
        "ID={}\n{}_{}_{}\n".format(str(img_ID), str(direction[0]), str(direction[1]), str(direction[2])))
    plt.savefig(featuremap_path + 'ID{}__{}_{}_{}.png'.format(str(img_ID), str(direction[0]), str(direction[1]),
                                                              str(direction[2])), dpi=600)
