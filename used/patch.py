# import os
# from patchify import patchify, unpatchify
# import matplotlib.pyplot as plt
#
# num = len(os.listdir('result/featuremap'))
#
# img = plt.imread("data/image/0.png", format=None)  # format读取文件格式，None为自动读取,读下来是numpy形式
# print(img.shape)  # H,W,C
#
# #This will split the image into small images of shape [3,3]
# patches = patchify(img, (3, 3), step=1)
# plt.imsave('result/featuremap/' + str(num) + '.png', img)
# plt.show()


import os
import matplotlib.pyplot as plt
import numpy as np
# from patchify import patchify, unpatchify
from GetPatched import patchify, unpatchify
patch_size = 64
patch_step = 32


img = plt.imread("data/image/0.png", format=None)
img = np.transpose(img, [2, 0, 1])
patches = patchify(img, (3, 64, 64), step=32)  # split image into 2*3 small 2*2 patches.
# for i in range(patches.shape[1]):
#     for j in range(patches.shape[2]):
#         patch = patches[:, i, j, :, :]
#         patch = np.squeeze(patch)
#         patch = np.transpose(patch, [1, 2, 0])
#         patchimg_num = len(os.listdir('result/featuremap'))
#         plt.imsave('result/featuremap/' + str(patchimg_num) + '.png', patch)

patch_num = int((img.shape[1]-patch_size)/patch_step) + 1
new_img_size = (patch_num-1)*patch_step+patch_size

#assert patches.shape == (2, 3, 2, 2)
reconstructed_image = unpatchify(patches, (3, new_img_size, new_img_size))
reconstructed_image = np.transpose(reconstructed_image, [1, 2, 0])
patchimg_num = len(os.listdir('result/featuremap'))
plt.imsave('data/image/preprocess/zz2D_Patch' + str(patchimg_num) + '.png', reconstructed_image)

