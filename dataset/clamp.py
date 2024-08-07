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
    return outmask




def clampImg(sitk_src):
    # 设置固定阈值为100，把骨骼和心脏及主动脉都分割出来
    sitk_seg = sitk.BinaryThreshold(sitk_src, lowerThreshold=100, upperThreshold=3000, insideValue=255, outsideValue=0)
    # 提取骨骼位置
    sitk_open = sitk.BinaryMorphologicalOpening(sitk_seg != 0)  # todo 本来有两个参数的，但是第二个参数写上来要报错
    array_open = GetLargestConnectedCompont(sitk_open)
    # 保留原图骨骼位置部分
    array_src = sitk.GetArrayFromImage(sitk_src)
    array_bone = array_open > 0
    array_mask = array_src * array_bone
    sitk_mask = sitk.GetImageFromArray(array_mask)
    sitk_mask.SetDirection(sitk_seg.GetDirection())
    sitk_mask.SetSpacing(sitk_seg.GetSpacing())
    sitk_mask.SetOrigin(sitk_seg.GetOrigin())
    return sitk_mask