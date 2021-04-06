import cv2 
import random
import numpy as np
import os
from keras.preprocessing.image import img_to_array
import sys 

classes = [0., 1.]
n_label = 2 
smooth = 1e-12


def Pii(gt_array, pd_array, classes):
    """ 
    找classes类中的Pii，即判断正确的结果，将其总数返回
    """
    m = gt_array.shape[0]
    n = gt_array.shape[1]

    sum = 0 
    for i in range(m):
        for j in range(n):
            if gt_array[i][j] == classes and pd_array[i][j] == classes:
                sum += 1
    return sum 

def get_union(array1, array2, classes):
    """ 
    遍历两个数组，求两个数组中具有classes值的元素的数量，重合的部分只统计一次�?    parameter
    ---------
    array1 : np.array object
      数组1
    array2 : np.array object
      数组2
    classes: int
    
    return
    ---------
    out : int
      数量
    """
    m = array1.shape[0]
    n = array1.shape[1]

    sum = 0
    for i in range(m):
        for j in range(n):
            if array1[i][j] == classes or array2[i][j] == classes:
                sum += 1
    return sum      

def pixel_accuracy(gt, pd):
    """
    求得ground_truth和prediction的pixel_accuarcy
    parameter
    ---------
    ground_truth : str or Image object
        标注图像,且应为灰度图�?        prediction ：str or Image object
        预测图像，且应为灰度图像
    
    return
    ---------
    pa : float or None
         pixel_accuarcy�?    """
    #将图像转为数�?    gt_a = np.array(gt)
    pd_a = np.array(pd)

    #求Pii
    sum_Pii = 0
    for i in range(len(classes)):
        sum_Pii += Pii(gt_a, pd_a, classes[i])

    #按公式求PA
    PA = sum_Pii/(gt.shape[0]*gt.shape[1])
    return PA

def mean_pixel_accuracy(gt, pd):
    """
    求得ground_truth和prediction的mean_pixel_accuarcy
    parameter
    ---------
    ground_truth : str or Image object
        标注图像,且应为灰度图�?        prediction ：str or Image object
        预测图像，且应为灰度图像
    
    return
    ---------
    pa : float
         mean_pixel_accuarcy�?    """
    gt_a = np.array(gt)
    pd_a = np.array(pd)

    #按公式求PA
    sum = 0
    for i in range(len(classes)):
        single_Pii = Pii(gt_a, pd_a, classes[i])
        gt_a_mid = gt_a.flatten()
        mask = (gt_a_mid == classes[i])
        total = gt_a_mid[mask]
        if(total.size!=0):
          sum += single_Pii/total.size
        else:
          sum += 1
    MPA = sum/len(classes)
    return MPA

def mean_IU(gt, pd):
    """
    求得ground_truth和prediction的pixel_accuarcy
    parameter
    ---------
    ground_truth : str or Image object
        标注图像,且应为灰度图�?        prediction ：str or Image object
        预测图像，且应为灰度图像
    
    return
    ---------
    pa : float
         pixel_accuarcy�?    """

    gt_a = np.array(gt)
    pd_a = np.array(pd)

    iou = np.zeros((len(classes)))

    #按公式计算MIoU
    sum = 0
    for i in range(len(classes)):
        single_Pii = Pii(gt_a, pd_a, classes[i])
        num = get_union(gt_a, pd_a, classes[i])
        if(num != 0):
          iou[i] = single_Pii/num
          sum += single_Pii/num
        else:
          iou[i] = 1
          sum += 1
    # MIoU = (iou[1]+iou[2]+iou[3])/(len(classes)-1)
    return iou

def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img,dtype="float") / 255.0
    return img


if __name__ == '__main__':
  label = load_img("/data/zlx/xian/experiment/test/test_label_groundtruth.tif", grayscale=True)
  label = img_to_array(label)
  print(label)

  pre = load_img("/data/zlx/xian/experiment/test/result_184_184.tif", grayscale=True)
  pre = img_to_array(pre)
  print(pre)

  acc = pixel_accuracy(label,pre)
  mean_acc = mean_pixel_accuracy(label,pre)
  iou = mean_IU(label,pre)
  print(acc,mean_acc)
  print(iou)

