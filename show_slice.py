import SimpleITK as sitk
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def show_slices(img_ID: int, x, y, z, img: np.ndarray):
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, 'data/image/nii/REORIENT.nii.gz')
    img_size = img.GetSize()
    img = nib.load('data/image/nii/REORIENT.nii.gz')
    img = img.get_fdata()
    slice_0 = np.flip(img[img_size[0] // 2, :, :])
    slice_1 = np.flip(img[:, img_size[1] // 2, :])
    slice_2 = np.flip(img[:, :, img_size[2] // 2])
    slices = [slice_0, slice_1, slice_2]
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    plt.suptitle("Center slices for spine after preprocess")
    plt.savefig('data/image/preprocess/48stride/ID{}_{}_{}_{}.png'.format(str(img_ID), str(x), str(y), str(z)), dpi=600)
