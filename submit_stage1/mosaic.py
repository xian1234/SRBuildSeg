import os, sys, gdal
from gdalconst import *
import glob
 
 
def get_extent(fn):
    ds = gdal.Open(fn)
    rows = ds.RasterYSize
    cols = ds.RasterXSize

    gt = ds.GetGeoTransform()
    minx = gt[0]
    maxy = gt[3]
    maxx = gt[0] + gt[1] * rows
    miny = gt[3] + gt[5] * cols
    return (minx, maxy, maxx, miny)
os.chdir('/home/zlx/xian/房山0.3米HD/Fangshan_HD_Sample/10500100127B2000/I00000280048_01_P001_PSH/')
in_files = glob.glob('*.TIF')

minX, maxY, maxX, minY = get_extent(in_files[0])
for fn in in_files[1:]:
    minx, maxy, maxx, miny = get_extent(fn)
    minX = min(minX, minx)
    maxY = max(maxY, maxy)
    maxX = max(maxX, maxx)
    minY = min(minY, miny)
 
in_ds = gdal.Open(in_files[0])
gt = in_ds.GetGeoTransform()
rows = int(maxX - minX) / abs(gt[5])
cols = int(maxY - maxy) / gt[1]
 
driver = gdal.GetDriverByName('GTiff')
out_ds = driver.Create('/home/zlx/xian/房山0.3米HD/mosaic.tif', cols, rows, 3, gdal.GDT_UInt16)
out_ds.SetProjection(in_ds.GetProjection())
out_band1 = out_ds.GetRasterBand(1)
out_band2 = out_ds.GetRasterBand(2)
out_band3 = out_ds.GetRasterBand(3)
 
gt = list(in_ds.GetGeoTransform())
gt[0], gt[3] = minX, maxY
out_ds.SetGeoTransform(gt)
 
for fn in in_files:
    in_ds = gdal.Open(fn)
    trans = gdal.Transformer(in_ds, out_ds, [])
    success, xyz = trans.TransformPoint(False, 0, 0)
    x, y, z = map(int, xyz)
    data1 = in_ds.GetRasterBand(1).ReadAsArray()
    out_band1.WriteArray(data1, x, y)
    data2 = in_ds.GetRasterBand(2).ReadAsArray()
    out_band2.WriteArray(data2, x, y)
    data3 = in_ds.GetRasterBand(3).ReadAsArray()
    out_band3.WriteArray(data3, x, y)
 
del in_ds, out_band1, out_band2, out_band3, out_ds
