import SimpleITK as sitk
import numpy as np

from ShowingOutput import show_slice

patch_size = [64, 64, 64]
patch_stride = [64, 64, 64]

# itk_img = sitk.ReadImage('data/image/nii/sub-verse584_dir-ax_ct.nii.gz')
itk_img = sitk.ReadImage('data/image/nii/sub-gl003_dir-ax_seg-vert_msk.nii.gz')
itk_img = sitk.GetArrayFromImage(itk_img)
show_slice(108,[1,1,1],itk_img,False)
for i in range(0, itk_img.shape[0] - patch_size[0] + 1, patch_stride[0]):
    for j in range(0, itk_img.shape[1] - patch_size[1] + 1, patch_stride[1]):
        for k in range(0, itk_img.shape[2] - patch_size[2] + 1, patch_stride[2]):
            patch = itk_img[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]].astype(np.float32)
            show_slice(108, [i, j, k], patch, False)
            # for i in range(0, itk_img.shape[0] - patch_size[0], patch_stride[0]):
            #     for j in range(0, itk_img.shape[1] - patch_size[1], patch_stride[1]):
            #         for k in range(0, itk_img.shape[2] - patch_size[2], patch_stride[2]):
            #             img_patch = itk_img[i:i + patch_size[0], j:j + patch_size[1],
            #                         k:k + patch_size[2]]

