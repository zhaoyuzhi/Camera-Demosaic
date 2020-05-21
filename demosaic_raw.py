import os
import cv2
import numpy as np
import rawpy
import colour_demosaicing
import imageio

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

def get_NEFs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if 'NEF' == filespath[-3:]:
                ret.append(filespath)
    return ret

def adjust_gamma(image, gamma = 2.2):
    image = np.clip(image.astype(np.float64) / 255.0, 0, 1)
    invGamma = 1.0 / gamma
    return np.power(image, invGamma)
    
def demosaic_NEF(imgpath, imgshape = [3000, 4000], white_level = 1023, black_level = 0, awb = [1, 1, 1], gamma = 2.2):
    # read image
    raw = np.fromfile(imgpath, dtype = np.uint16)
    bayer_raw = raw.reshape(imgshape)
    # normalization
    bayer_raw = (bayer_raw.astype(np.float64) - black_level) / (white_level - black_level)
    # awb
    bayer_raw[0::2,0::2] *= awb[0]
    bayer_raw[1::2,1::2] *= awb[2]
    bayer_raw = (np.clip(bayer_raw, 0, 1) * 255.0).astype(np.uint8)
    # demosaic
    demosaicked_rgb = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(bayer_raw, 'BGGR')
    # gamma correction
    demosaicked_rgb = adjust_gamma(demosaicked_rgb, gamma)
    demosaicked_rgb = (demosaicked_rgb * 255.0).astype(np.uint8)
    return demosaicked_rgb

imgpath = 'E:\\Deblur\\Short-Long RGB to RGB Mapping\\data\\slrgb2rgb_long10_short1_mobile_phone\\2020_05_16_13_22_19_382\\2020_05_16_13_22_19_780.raw'
demosaicked_rgb = demosaic_NEF(imgpath, imgshape = [3000, 4000], \
    white_level = 1023, black_level = 64, awb = [1.687, 1, 1.398], gamma = 2.2)

# show
demosaicked_rgb = cv2.resize(demosaicked_rgb, (demosaicked_rgb.shape[1] // 4, demosaicked_rgb.shape[0] // 4))
demosaicked_rgb = cv2.cvtColor(demosaicked_rgb, cv2.COLOR_RGB2BGR)
cv2.imshow('demo', demosaicked_rgb)
cv2.waitKey(0)
