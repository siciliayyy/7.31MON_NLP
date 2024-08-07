import SimpleITK as sitk
import numpy as np
import torch

from ShowingOutput import show_slice, show_slices
from dataset.resample import resampleImg
from evaluation.dice_score import dice_score
from model.vnet import VNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = VNet()
net.to(device=device)
s = torch.load('result/models/913/Epoch9_LS-0.984_DC-0.060.pth')
net.load_state_dict(s)

# H,W,C
itk_img = sitk.ReadImage('data/image/nii/sub-gl108_dir-ax_ct.nii.gz')
itk_mask = sitk.ReadImage('data/image/nii/sub-gl108_dir-ax_seg-vert_msk.nii.gz')

aa1 = sitk.GetArrayFromImage(itk_img)

itk_img = resampleImg(itk_img, (1, 1, 1), resamplemethod=sitk.sitkLinear)
itk_mask = resampleImg(itk_mask, (1, 1, 1), resamplemethod=sitk.sitkNearestNeighbor)

itk_img = sitk.GetArrayFromImage(itk_img)  # C,H,W
itk_mask = sitk.GetArrayFromImage(itk_mask)

img_shape = itk_img.shape
itk_img = np.clip(itk_img, 0, 3000, out=None)
itk_mask = np.clip(itk_mask, 0, 1, out=None)


patch_size = [64, 64, 64]
patch_stride = [64, 64, 64]
preprocessed = []

reconstruction_img = reconstruction_mask = torch.zeros([img_shape[0], img_shape[1], img_shape[2]])

dice_scores = []
for i in range(0, itk_img.shape[0] - patch_size[0], patch_stride[0]):
    for j in range(0, itk_img.shape[1] - patch_size[1], patch_stride[1]):
        for k in range(0, itk_img.shape[2] - patch_size[2], patch_stride[2]):
            img_patch = itk_img[i:i + patch_size[0], j:j + patch_size[1],
                        k:k + patch_size[2]]
            mask_patch = itk_mask[i:i + patch_size[0], j:j + patch_size[1],
                         k:k + patch_size[2]]

            # show_slices('010', [i, j, k], img_patch, mask_patch,0,0, False)

            net.eval()
            with torch.no_grad():
                mask_patch = torch.tensor((mask_patch.astype(float)))

                img_patch = torch.tensor((img_patch.astype(float)))

                img_patch = img_patch.unsqueeze(0)
                img_patch = img_patch.unsqueeze(0)

                img_patch = img_patch.to(device, dtype=torch.float)

                img_patch = net(img_patch)

                img_patch = (torch.sigmoid(img_patch) > 0.5).float().cpu().detach().numpy()


                img_patch = np.squeeze(img_patch)

                bbb = mask_patch.float().cpu().detach().numpy()
                show_slices('005', [i, j, k], img_patch, bbb, 0, 0, False)

                direction = [i, j, k]
                reconstruction_img[i:i + 64, j:j + 64, k:k + 64] = torch.tensor(img_patch)

                ccc = reconstruction_img.float().cpu().detach().numpy()
                show_slice('9980', [i, j, k], ccc, False)

                reconstruction_mask[i:i + 64, j:j + 64, k:k + 64] = mask_patch
dice_scores.append(dice_score((torch.sigmoid(reconstruction_img) > 0.5).float(), reconstruction_mask.float()))
rec_img = reconstruction_img.float().cpu().detach().numpy()
rec_mask = reconstruction_mask.float().cpu().detach().numpy()
show_slices('13', [1, 1, 1], rec_img, rec_mask, dice_scores[-1], dice_scores[-1], False)
rec_img = sitk.GetImageFromArray(rec_img)
rec_mask = sitk.GetImageFromArray(rec_mask)
sitk.WriteImage(rec_img, 'result/11.3featuremap/11.3_picture/img.nii.gz')
sitk.WriteImage(rec_mask, 'result/11.3featuremap/11.3_picture/mask.nii.gz')

#
# itk_img = torch.tensor(itk_img)
# itk_img = itk_img.detach().numpy()
# itk_img = itk_img# .transpose([1, 0, 2])
# aaa = sitk.GetImageFromArray(itk_img)
# sitk.WriteImage(aaa, 'result/nii/trytry.nii.gz')
