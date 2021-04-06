import os
import sys
import cv2
import numpy as np

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
except ImportError:
    pass


def generate_mod_LR_bic():
    # set parameters
    up_scale = 4
    mod_scale = 4
    # set data dir
    #sourcedir = '/mnt/lustrenew/dongrunmin/dongrunmin/SR/dataset/RefSR_data/jinan_L18/train_data/img_HR_ms'
    #savedir = '/mnt/lustrenew/dongrunmin/dongrunmin/SR/dataset/RefSR_data/jinan_L18/train_data/use_data'
    sourcedir = '/mnt/lustrenew/dongrunmin/dongrunmin/SR/dataset/RefSR_data/jinan_L18/train_data/img_HR_ms'
    savedir = '/mnt/lustrenew/dongrunmin/dongrunmin/SR/dataset/RefSR_data/jinan_L18/train_data/use_data'

    saveHRpath = os.path.join(savedir, 'ms_HR')
    saveLRpath = os.path.join(savedir, 'ms_LR')
    saveBicpath = os.path.join(savedir, 'ms_LR_UX4')

    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if not os.path.isdir(saveHRpath):
        os.mkdir(saveHRpath)
    else:
        print('It will cover ' + str(saveHRpath))

    if not os.path.isdir(saveLRpath):
        os.mkdir(saveLRpath)
    else:
        print('It will cover ' + str(saveLRpath))

    if not os.path.isdir(saveBicpath):
        os.mkdir(saveBicpath)
    else:
        print('It will cover ' + str(saveBicpath))

    filepaths = [f for f in os.listdir(sourcedir) if f.endswith('.jpg')]
    num_files = len(filepaths)

    # prepare data with augementation
    for i in range(num_files):
        filename = filepaths[i]
        print('No.{} -- Processing {}'.format(i, filename))
        # read image
        image = cv2.imread(os.path.join(sourcedir, filename))

        width = int(np.floor(image.shape[1] / mod_scale))
        height = int(np.floor(image.shape[0] / mod_scale))
        # modcrop
        if len(image.shape) == 3:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width, :]
        else:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width]
        # LR
        image_LR = imresize_np(image_HR, 1 / up_scale, True)
        # bic
        image_Bic = imresize_np(image_LR, up_scale, True)

        cv2.imwrite(os.path.join(saveHRpath, filename), image_HR)
        cv2.imwrite(os.path.join(saveLRpath, filename), image_LR)
        cv2.imwrite(os.path.join(saveBicpath, filename), image_Bic)


if __name__ == "__main__":
    generate_mod_LR_bic()
