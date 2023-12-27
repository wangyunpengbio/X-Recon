from __future__ import print_function
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from util import init_rtvec
from module import ProST

from util import calculate_param, detector_corner_prepare
import nibabel as nib

device = torch.device("cuda")
PI = 3.1415926

input_file_path = '/home/u18111510027/X2CT/3DGAN/9PtxMultiView/all/00100876_1.2.840.113704.1.111.6792.1510885723.7/real_ct_body_outside_below1k.mha'
BATCH_SIZE = 1
CT_PATH = '/home/u18111510027/Projective-Spatial-Transformers-master/ProSTModule/demo.nii'

def preprocess():
    import SimpleITK as sitk
    img = sitk.ReadImage(input_file_path)
    # spacing = file.GetSpacing()  #读取该数据的spacing
    # 经过一系列操作，生成三维数组 result_file,然后将result_file保存为mha.格式的文件
    # result_file = sitk.GetImageFromArray(file)
    sitk.WriteImage(img, CT_PATH)

def main():
    CT_vol = nib.load(CT_PATH)
    CT_vol = CT_vol.get_data()
    # Calculate geometric parameters
    param, det_size, norm_factor = calculate_param(CT_vol, vol_spacing=1)
    CT_vol, ray_proj_mov, corner_pt = detector_corner_prepare(CT_vol, det_size, ISFlip=False, device='cuda')
    # Initialize projection model
    projmodel = ProST(param).to(device)

    ########## Hard Code test groundtruth and initialize poses ##########
    # [rx, ry, rz, tx, ty, tz]
    manual_rtvec_gt = np.array([[90.0, 180.0, 0, 0, 0, 0]])
    manual_rtvec_smp = np.array([[180.0, 180.0, 0.0, 0.0, 0.0, 0.0]])

    # Normalization and conversion to transformation matrix
    manual_rtvec_gt[:, :3] = manual_rtvec_gt[:, :3]*PI/180
    manual_rtvec_gt[:, 3:] = manual_rtvec_gt[:, 3:]/norm_factor
    manual_rtvec_smp[:, :3] = manual_rtvec_smp[:, :3]*PI/180
    manual_rtvec_smp[:, 3:] = manual_rtvec_smp[:, 3:]/norm_factor
    # 将真实角度，转换成vector向量矩阵
    transform_mat3x4_gt, rtvec, rtvec_gt = init_rtvec(device, manual_rtvec_gt=manual_rtvec_gt,
                                                           manual_rtvec_smp=manual_rtvec_smp)

    def CT_rtvec_toXRay(rtvec, filename):
        with torch.no_grad():
            # CT_vol为CT图像，ray_proj_mov为空的探测板[1,1,128,128]，传入进来只是为了获得探测板的大小
            # rtvec_gt为真实位移矩阵，corner_pt角点坐标
            target = projmodel(CT_vol, ray_proj_mov, rtvec, corner_pt)
            # Min-Max to [0,1] normalization for target image
            min_tar, _ = torch.min(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
            max_tar, _ = torch.max(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
            target = (target.reshape(BATCH_SIZE, -1) - min_tar) / (max_tar - min_tar)
            target = target.reshape(BATCH_SIZE, 1, det_size, det_size)
            plt.imshow(target.detach().cpu().numpy()[0,0,:,:].reshape(det_size, det_size), cmap='gray')
            plt.title('target img')
            plt.savefig(filename)

    CT_rtvec_toXRay(rtvec_gt, "./img-target-1024.png")
    CT_rtvec_toXRay(rtvec, "./img-rotate-1024.png")


if __name__ == "__main__":
    # preprocess()
    main()
