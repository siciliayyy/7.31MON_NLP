import csv
import os
import re
import nibabel as nib  # 注意：nib.oritation可以返回LAS信息

import SimpleITK as sitk
from tqdm import tqdm

# mask_file = 'E:\savefryomxftp\\5.22MON\data\dataset-verse20training\derivatives'
# img_file = 'E:\savefryomxftp\\5.22MON\data\dataset-verse20training\\rawdata'

mask_file = 'data/image/preprocess/1_resample'
img_file = 'data/image/preprocess/1_resample'


mask_file = re.sub(r'(\\){1,}', '/', mask_file)
img_file = re.sub(r'(\\){1,}', '/', img_file)
img_direction = os.listdir(img_file)
mask_direction = os.listdir(mask_file)
img_dirs = []
mask_dirs = []

def write_csv(ID, ori, size, spacing):
    with open(r'resampling.csv', 'a+', newline='', encoding='utf8',) as f:
        csv_write = csv.writer(f)
        data_row = [ID, ori, size, spacing]
        csv_write.writerow(data_row)

# todo:读取522MON的原图
# for i in img_direction:
#     inside = os.listdir(img_file + '/' + i)
#     for j in inside:
#         if re.search('.nii.gz', j):
#             img_dirs.append(img_file + '/' + i + '/' + j)

# todo:读取resample完以后的数据
for i in img_direction:
    img_dirs.append(img_file + '/' + i)



# for i in mask_direction:
#     inside = os.listdir(mask_file + '/' + i)
#     for j in inside:
#         if re.search('.nii.gz', j):
#             mask_dirs.append(mask_file + '/' + i + '/' + j)

pbar = tqdm(enumerate(img_dirs), unit='preprocessing', total=len(img_dirs))
for NO, i in pbar:
    itk_img = sitk.ReadImage(i)
    size = itk_img.GetSize()
    spacing = itk_img.GetSpacing()
    spacing = ('{:.2f}'.format(spacing[0]), '{:.2f}'.format(spacing[1]), '{:.2f}'.format(spacing[2]))
    nib_img = nib.load(i)
    ori = nib.aff2axcodes(nib_img.affine)
    ID = re.sub('[a-zA-Z/:_.]', '', i)

    ori_dict = {'L': 0, 'R': 0, 'A': 1, 'P': 1, 'S': 2, 'I': 2}
    output = [ori_dict[ori[0]], ori_dict[ori[1]], ori_dict[ori[2]]]


    write_csv(ID, ori, size, spacing)
