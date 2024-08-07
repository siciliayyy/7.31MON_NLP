import SimpleITK as sitk

# 最大连通域提取
def GetLargestConnectedCompont(binarysitk_image):
    cc = sitk.ConnectedComponent(binarysitk_image)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, binarysitk_image)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage == maxlabel] = 255
    outmask[labelmaskimage != maxlabel] = 0
    outmask_sitk = sitk.GetImageFromArray(outmask)
    outmask_sitk.SetDirection(binarysitk_image.GetDirection())
    outmask_sitk.SetSpacing(binarysitk_image.GetSpacing())
    outmask_sitk.SetOrigin(binarysitk_image.GetOrigin())
    return outmask_sitk

# 逻辑与操作
def GetMaskImage(sitk_src, sitk_mask, replacevalue=0):
    array_src = sitk.GetArrayFromImage(sitk_src)
    array_mask = sitk.GetArrayFromImage(sitk_mask)
    array_out = array_src.copy()
    array_out[array_mask == 0] = replacevalue
    outmask_sitk = sitk.GetImageFromArray(array_out)
    outmask_sitk.SetDirection(sitk_src.GetDirection())
    outmask_sitk.SetSpacing(sitk_src.GetSpacing())
    outmask_sitk.SetOrigin(sitk_src.GetOrigin())
    return outmask_sitk


# 读取nii
sitk_src = sitk.ReadImage('data/image/nii/sub-verse584_dir-ax_ct.nii.gz')

# step1.设置固定阈值为100，把骨骼和心脏及主动脉都分割出来
sitk_seg = sitk.BinaryThreshold(sitk_src, lowerThreshold=100, upperThreshold=3000, insideValue=255, outsideValue=0)
sitk.WriteImage(sitk_seg, 'data/image/nii/001.nii.gz')

# step2.形态学开运算+最大连通域提取,粗略的心脏和主动脉图像
sitk_open = sitk.BinaryMorphologicalOpening(sitk_seg != 0)  # todo 本来有两个参数的，但是第二个参数写上来要报错
sitk_open = GetLargestConnectedCompont(sitk_open)
sitk.WriteImage(sitk_open, 'data/image/nii/002.nii.gz')     # todo:提取出了骨骼，但是是二值化，感觉用不上

# step3.再将step1的结果与step2的结果相减,得到骨骼部分
array_open = sitk.GetArrayFromImage(sitk_open)
array_src = sitk.GetArrayFromImage(sitk_src)
array_bone = array_open > 0
array_mask = array_src*array_bone
sitk_mask = sitk.GetImageFromArray(array_mask)
sitk_mask.SetDirection(sitk_seg.GetDirection())
sitk_mask.SetSpacing(sitk_seg.GetSpacing())
sitk_mask.SetOrigin(sitk_seg.GetOrigin())
sitk.WriteImage(sitk_mask, 'data/image/nii/003.nii.gz')



