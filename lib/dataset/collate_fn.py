
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch


def collate_gan(batch):
    '''
    :param batch: [imgs, boxes, labels] dtype = np.ndarray
    imgs:
      shape = (C H W)
    :return:
    '''
    ct = [x[0] for x in batch]
    xray = [x[1] for x in batch]
    file_path = [x[2] for x in batch]

    return torch.stack(ct), torch.stack(xray), file_path


def collate_gan_views(batch):
    '''
    :param batch: [imgs, boxes, labels] dtype = np.ndarray
    imgs:
      shape = (C H W)
    :return:
    '''
    ct = [x[0] for x in batch]
    xray1 = [x[1] for x in batch]
    xray2 = [x[2] for x in batch]
    file_path = [x[3] for x in batch]

    return torch.stack(ct), [torch.stack(xray1), torch.stack(xray2)], file_path

def collate_gan_views_5Resolution(batch):
    '''
    :param batch: [imgs, boxes, labels] dtype = np.ndarray
    imgs:
      shape = (C H W)
    :return:
    '''
    ct_1 = [x[0] for x in batch]
    ct_2 = [x[1] for x in batch]
    ct_3 = [x[2] for x in batch]
    ct_4 = [x[3] for x in batch]
    ct_5 = [x[4] for x in batch]
    xray1 = [x[5] for x in batch]
    xray2 = [x[6] for x in batch]
    file_path = [x[7] for x in batch]

    return [torch.stack(ct_1), torch.stack(ct_2), torch.stack(ct_3), torch.stack(ct_4), torch.stack(ct_5)], [torch.stack(xray1), torch.stack(xray2)], file_path