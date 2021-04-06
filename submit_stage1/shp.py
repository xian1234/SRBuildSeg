import cv2
from skimage.io import *
import os
import gdal

data1="/home/zlx/xian/train/0_3m/18OCT07031915-S2AS_R1C3-I00000280048_01_P001.TIF"
output = "/home/zlx/xian/train/result_0_3m_r1c3.tif"
msk1=gdal.Open(data1)
msk2=imread(data1)
msk2[msk2<25]=0
msk2[msk2>0]=1
datatype = gdal.GDT_UInt16
driver = gdal.GetDriverByName("GTiff")
im_width = msk1.RasterXSize
im_height = msk1.RasterYSize
im_geotrans = msk1.GetGeoTransform()
im_proj = msk1.GetProjection()
out_mask = driver.Create(output, im_width, im_height, 1, datatype)
out_mask.SetGeoTransform(im_geotrans)
out_mask.SetProjection(im_proj)
out_mask.GetRasterBand(1).WriteArray(msk2)
