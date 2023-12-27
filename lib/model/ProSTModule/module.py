import torch
import ProSTGrid
import torch.nn as nn
from lib.model.ProSTModule.util import _bilinear_interpolate_no_torch_5D
import torchgeometry as tgm
from lib.model.ProSTModule.posevec2mat import raydist_range

device = torch.device("cuda")

class ProST(nn.Module):
    def __init__(self, param):
        super(ProST, self).__init__()
        self.src = param[0]
        self.det = param[1]
        self.pix_spacing = param[2]
        self.step_size = param[3]

    def forward(self, x, y, rtvec, corner_pt): # _3D_vol, ray_proj_mov, rtvec_gt, corner_pt
        BATCH_SIZE = rtvec.size()[0]
        H = y.size()[2]
        W = y.size()[3]

        transform_mat4x4 = tgm.rtvec_to_pose(rtvec)
        transform_mat3x4 = transform_mat4x4[:, :3, :]
        dist_min, dist_max = raydist_range(transform_mat3x4, corner_pt, self.src)
        grid = ProSTGrid.forward(corner_pt, y.size(), dist_min.data, dist_max.data,\
                                     self.src, self.det, self.pix_spacing, self.step_size, False)
        grid_trans = grid.bmm(transform_mat3x4.transpose(1,2)).view(BATCH_SIZE, H, W, -1, 3)

        x_3d_ad = _bilinear_interpolate_no_torch_5D(x, grid_trans)
        x_2d_ad = torch.sum(x_3d_ad, dim=-1)

        return x_2d_ad