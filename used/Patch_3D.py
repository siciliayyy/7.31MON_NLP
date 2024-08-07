import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import numpy as np
from patchify import patchify, unpatchify
patch_size = 64
patch_step = 32


img = sitk.ReadImage("data/image/nii/sub-verse584_dir-ax_ct.nii.gz")
img = sitk.GetArrayFromImage(img)
patches = patchify(img, (64, 64, 64), step=32)   # split image into 2*3 small 2*2 patches.
H_num = patches.shape[0]
W_num = patches.shape[1]
C_num = patches.shape[2]
img_patch = []
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        for k in range(patches.shape[2]):
            patch = patches[i, j, k, :]
            patch = patch.tolist()
            img_patch.append(patch)
            # patchimg_num = len(os.listdir('data/image/preprocess/4_patches'))
            # patch_nii = sitk.GetImageFromArray(patch)
            # sitk.WriteImage(patch_nii, 'data/image/preprocess/4_patches/' + str(patchimg_num) + '.nii.gz')

for i in range(H_num):
    for j in range(W_num):
        for k in range(C_num):
            patches[i][j][k].append(img_patch[i*H_num+j*W_num+k*C_num])
patches = np.array(patches, dtype=object)
#assert patches.shape == (2, 3, 2, 2)
reconstructed_image = unpatchify(patches, (H_num*patch_size, W_num*patch_size, C_num*patch_size))
reconstructed_image = np.transpose(reconstructed_image, [1, 2, 0])
patchimg_num = len(os.listdir('result/featuremap'))
plt.imsave('result/featuremap/' + str(patchimg_num) + '.png', reconstructed_image)
