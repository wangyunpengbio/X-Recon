# 我自己新加的
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.utils.transform_3d import *
import torch
import numpy as np


def tensor_backto_unnormalization_image(input_image, mean, std):
    '''
    1. image = (image + 1) / 2.0
    2. image = image
    :param input_image: tensor whose size is (c,h,w) and channels is RGB
    :param imtype: tensor type
    :return:
       numpy (c,h,w)
    '''
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image = image_tensor.data.cpu().float().numpy()
    image = image * std + mean
    return image


class CT_XRAY_Data_Augmentation_Multi(object):
    def __init__(self, opt=None):
        self.augment = List_Compose([
            (None, None, None),

            (Resize_image(size=(opt.ct_channel, opt.fine_size, opt.fine_size)),
             Resize_image(size=(opt.xray_channel, opt.fine_size, opt.fine_size)),
             Resize_image(size=(opt.xray_channel, opt.fine_size, opt.fine_size)),),

            (Limit_Min_Max_Threshold(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]), None, None),

            (Normalization(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]),
             Normalization(opt.XRAY1_MIN_MAX[0], opt.XRAY1_MIN_MAX[1]),
             Normalization(opt.XRAY2_MIN_MAX[0], opt.XRAY2_MIN_MAX[1])),

            (Normalization_gaussian(opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1]),
             Normalization_gaussian(opt.XRAY1_MEAN_STD[0], opt.XRAY1_MEAN_STD[1]),
             Normalization_gaussian(opt.XRAY2_MEAN_STD[0], opt.XRAY2_MEAN_STD[1])),

            # (Get_Key_slice(opt.select_slice_num), None, None),

            (ToTensor(), ToTensor(), ToTensor())

        ])

    def __call__(self, img_list):
        '''
        :param img: PIL image
        :param boxes: numpy.ndarray
        :param labels: numpy.ndarray
        :return:
        '''
        return self.augment(img_list)


class CT_XRAY_Data_Test_Multi(object):
    def __init__(self, opt=None):
        self.augment = List_Compose([
            (None, None, None),

            (Resize_image(size=(opt.ct_channel, opt.fine_size, opt.fine_size)),
             Resize_image(size=(opt.xray_channel, opt.fine_size, opt.fine_size)),
             Resize_image(size=(opt.xray_channel, opt.fine_size, opt.fine_size)),),

            (Limit_Min_Max_Threshold(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]), None, None),

            (Normalization(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]),
             Normalization(opt.XRAY1_MIN_MAX[0], opt.XRAY1_MIN_MAX[1]),
             Normalization(opt.XRAY2_MIN_MAX[0], opt.XRAY2_MIN_MAX[1])),

            (Normalization_gaussian(opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1]),
             Normalization_gaussian(opt.XRAY1_MEAN_STD[0], opt.XRAY1_MEAN_STD[1]),
             Normalization_gaussian(opt.XRAY2_MEAN_STD[0], opt.XRAY2_MEAN_STD[1])),

            # (Get_Key_slice(opt.select_slice_num), None),

            (ToTensor(), ToTensor(), ToTensor())

        ])

    def __call__(self, img):
        '''
        :param img: PIL image
        :param boxes: numpy.ndarray
        :param labels: numpy.ndarray
        :return:
        '''
        return self.augment(img)

class CT_Data_Augmentation(object):
    def __init__(self, opt=None):
        self.augment = Compose([
            Permute((1, 0, 2)),
            Normalization(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]),
            Normalization_gaussian(opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1]),
            Get_Key_slice(opt.select_slice_num),
            ToTensor()
        ])

    def __call__(self, img):
        '''
        :param img: PIL image
        :param boxes: numpy.ndarray
        :param labels: numpy.ndarray
        :return:
        '''
        return self.augment(img)


class Xray_Data_Augmentation(object):
    def __init__(self, opt=None):
        self.augment = Compose([
            Normalization(opt.XRAY1_MIN_MAX[0], opt.XRAY1_MIN_MAX[1]),
            Normalization_gaussian(opt.XRAY1_MEAN_STD[0], opt.XRAY1_MEAN_STD[1]),
            ToTensor()
        ])

    def __call__(self, img):
        '''
        :param img: PIL Image
        :return:
        '''
        return self.augment(img)
