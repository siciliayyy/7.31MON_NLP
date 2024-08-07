import SimpleITK as sitk
import numpy as np


def resampleImage(image, targetSpacing, resamplemethod=sitk.sitkLinear):
    """
    将体数据重采样的指定的spacing大小
    paras：
    image：sitk读取的image信息，这里是体数据
    targetSpacing：指定的spacing，例如[1,1,1]
    resamplemethod:插值类型
    return：重采样后的数据
    """
    target_size = [0, 0, 0]
    # 读取原始数据的size和spacing信息
    ori_size = image.GetSize()
    ori_spacing = image.GetSpacing()
    transform = sitk.Transform()
    transform.SetIdentity()
    # 计算改变spacing后的size，用物理尺寸/体素的大小
    target_size[0] = round(ori_size[0] * ori_spacing[0] / targetSpacing[0])
    target_size[1] = round(ori_size[1] * ori_spacing[1] / targetSpacing[1])
    target_size[2] = round(ori_size[2] * ori_spacing[2] / targetSpacing[2])
    # 设定重采样的一些参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetSize(target_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputSpacing(targetSpacing)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(resamplemethod)
    if resamplemethod == sitk.sitkNearestNeighbor:
        # mask用最近邻插值，保存为uint8
        resampler.SetOutputPixelType(sitk.sitkUInt8)
    else:
        # 体数据用线性插值，保存为float32
        resampler.SetOutputPixelType(sitk.sitkFloat32)
    newImage = resampler.Execute(image)
    return newImage


itkimage = sitk.ReadImage('data/image/nii/sub-verse596_dir-ax_ct.nii.gz')
aaa = itkimage.GetSize()
bbb = resampleImage(itkimage, [1, 10, 100], resamplemethod=sitk.sitkNearestNeighbor)
sitk.WriteImage(bbb, r'data/image/nii/010.nii.gz')
img_array = sitk.GetArrayFromImage(itkimage)  # indexes are z,y,x (notice the ordering)
# center = np.array([node_x, node_y, node_z])  # nodule center
origin = np.array(itkimage.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
spacing = np.array(itkimage.GetSpacing())  # spacing of voxels in world coor. (mm)
# v_center = np.rint((center - origin) / spacing)  # nodule center in voxel space (still x,y,z ordering)
print()
