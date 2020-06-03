import os
import cv2
import numpy as np
import rawpy
import colour_demosaicing
import imageio

def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i]) - 1]
    file.close()
    return content

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

def get_files_NEFs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if 'raw' == filespath[-3:]:
                ret.append(os.path.join(root, filespath))
    return ret

def get_NEFs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if 'raw' == filespath[-3:]:
                ret.append(filespath)
    return ret

def get_files_txts(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if 'txt' == filespath[-3:]:
                ret.append(os.path.join(root, filespath))
    return ret

def adjust_gamma(image, gamma = 2.2):
    image = np.clip(image.astype(np.float64) / 255.0, 0, 1)
    invGamma = 1.0 / gamma
    return np.power(image, invGamma)

def get_camera_info(detail_info):
    print(detail_info)
    ratio = int(float(detail_info[0].split('=')[-1]))
    print('ratio', ratio)
    iso_range_min, iso_range_max = int(detail_info[1].split('=')[-1].split(',')[0].split('[')[-1]), int(detail_info[1].split('=')[-1].split(',')[1].split(']')[0])
    print('iso_range_min, iso_range_max', iso_range_min, iso_range_max)
    raw_size_h, raw_size_w = int(detail_info[2].split('=')[-1].split('x')[0]), int(detail_info[2].split('=')[-1].split('x')[1])
    print('raw_size_h, raw_size_w', raw_size_h, raw_size_w)
    auto_exposure_time = int(float(detail_info[3].split('=')[-1].split('ms')[0]))
    print('auto_exposure_time', auto_exposure_time)
    auto_iso = int(detail_info[4].split('=')[-1])
    print('auto_iso', auto_iso)
    exposure_time = int(float(detail_info[5].split('=')[-1].split('ms')[0]))
    print('exposure_time', exposure_time)
    iso = int(detail_info[6].split('=')[-1])
    print('iso', iso)
    white_level = int(detail_info[7].split('=')[-1])
    print('white_level', white_level)
    black_level = int(detail_info[8].split(',')[2])
    print('black_level', black_level)
    awb_b, awb_g, awb_r = float(detail_info[9].split('=')[-1].split(',')[0].split('[')[-1]), float(detail_info[9].split('=')[-1].split(',')[1]), float(detail_info[9].split('=')[-1].split(',')[3].split(']')[0])
    print('awb_b, awb_g, awb_r', awb_b, awb_g, awb_r)
    return raw_size_h, raw_size_w, white_level, black_level, awb_b, awb_g, awb_r

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

imglist = get_files_NEFs('E:\\Deblur\\Short-Long RGB to RGB Mapping\\data\\mobile_phone\\logn8_short1_20200601')
neflist = get_NEFs('E:\\Deblur\\Short-Long RGB to RGB Mapping\\data\\mobile_phone\\logn8_short1_20200601')
txtlist = get_files_txts('E:\\Deblur\\Short-Long RGB to RGB Mapping\\data\\mobile_phone\\logn8_short1_20200601')

for i in range(len(imglist)):
    print(i, imglist[i])
    # get camera information
    detail_info = text_readlines(txtlist[i])
    raw_size_h, raw_size_w, white_level, black_level, awb_b, awb_g, awb_r = get_camera_info(detail_info)
    # demosaic
    demosaicked_rgb = demosaic_NEF(imglist[i], imgshape = [raw_size_w, raw_size_h], \
        white_level = white_level, black_level = black_level, awb = [awb_b, awb_g , awb_r], gamma = 2.5)
    # save
    last_len = len(imglist[i].split('\\')[-1])
    savepath = os.path.join(imglist[i][:-last_len], neflist[i][:-3] + 'png')
    print(savepath)
    demosaicked_rgb = cv2.cvtColor(demosaicked_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(savepath, demosaicked_rgb)
