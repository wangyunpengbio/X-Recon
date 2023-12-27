
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import numpy as np
from lib.model.base_model import Base_Model
import lib.model.nets.factory as factory
from .loss.multi_gan_loss import GANLoss, RestructionLoss, WGANLoss
from lib.utils.image_pool import ImagePool
import lib.utils.metrics as Metrics
from torch_ema import ExponentialMovingAverage

from lib.model.ProSTModule.util import rtvec_transform, calculate_param, detector_corner_prepare
from lib.model.ProSTModule.module import ProST
import random

class CTGAN(Base_Model):
    def __init__(self):
        super(CTGAN, self).__init__()

    @property
    def name(self):
        return 'multiView_CTGAN'

    '''
    Init network architecture
    '''

    def init_network(self, opt):
        Base_Model.init_network(self, opt)
        self.num_forward = 0 # 用于记录向前传播的次数
        self.if_pool = opt.if_pool
        self.multi_view = opt.multi_view
        self.conditional_D = opt.conditional_D
        self.auxiliary_loss = False
        self.if_ema = opt.if_ema
        assert len(self.multi_view) > 0

        self.loss_names = ['D', 'G']
        self.metrics_names = ['Mse', 'CosineSimilarity', 'PSNR']
        self.visual_names = ['G_real', 'G_fake', 'G_input1', 'G_input2', 'G_Map_fake_axial', 'G_Map_real_axial', 'G_Map_fake_sagittal',
                             'G_Map_real_sagittal', 'G_Map_fake_coronal', 'G_Map_real_coronal']

        # identity loss
        if self.opt.idt_lambda > 0:
            self.loss_names += ['idt']

        # feature metric loss
        if self.opt.feature_D_lambda > 0:
            self.loss_names += ['fea_m']

        # map loss
        if self.opt.map_projection_lambda > 0:
            self.loss_names += ['map_m']

        # auxiliary loss
        if self.opt.auxiliary_lambda > 0:
            self.loss_names += ['auxiliary']
            self.auxiliary_loss = True

        if self.training:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        if self.if_ema:
            self.model_names.append('G_ema')

        self.netG = factory.define_3DG(opt.noise_len, opt.input_shape, opt.output_shape,
                                       opt.input_nc_G, opt.output_nc_G, opt.ngf, opt.which_model_netG,
                                       opt.n_downsampling, opt.norm_G, not opt.no_dropout,
                                       opt.init_type, self.gpu_ids, opt.n_blocks,
                                       opt.encoder_input_shape, opt.encoder_input_nc, opt.encoder_norm,
                                       opt.encoder_blocks, opt.skip_number, opt.activation_type, opt=opt)

        if self.if_ema:
            self.netG_ema = ExponentialMovingAverage(self.netG.parameters(), decay=opt.beta_ema)
            # self.netD_ema = ExponentialMovingAverage(self.netD.parameters(), decay=0.95)

        if self.training:
            # out of discriminator is not probability when GAN loss is LSGAN
            if opt.ganloss == 'gan':
                use_sigmoid = True
            elif opt.ganloss == 'lsgan':
                use_sigmoid = False
            elif opt.ganloss == 'wgan':
                self.loss_names += ['wasserstein']
                use_sigmoid = False
            elif opt.ganloss == 'wgan_gp':
                self.loss_names += ['wasserstein', 'grad_penalty']
                use_sigmoid = False
            else:
                raise ValueError()

            # conditional Discriminator
            if self.conditional_D:
                d_input_channels = opt.input_nc_D + 2
            else:
                d_input_channels = opt.input_nc_D
            self.netD = factory.define_D(d_input_channels, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, opt.norm_D,
                                         use_sigmoid, opt.init_type, self.gpu_ids,
                                         opt.discriminator_feature, num_D=opt.num_D, n_out_channels=opt.n_out_ChannelsD)
            if self.if_pool:
                self.fake_pool = ImagePool(opt.pool_size)

        self.set_ProST()

    # correspond to visual_names
    def get_normalization_list(self):
        return [
            [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
            [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
            [self.opt.XRAY1_MEAN_STD[0], self.opt.XRAY1_MEAN_STD[1]],
            [self.opt.XRAY2_MEAN_STD[0], self.opt.XRAY2_MEAN_STD[1]],
            [self.opt.XRAY1_MEAN_STD[0], self.opt.XRAY1_MEAN_STD[1]],
            [self.opt.XRAY2_MEAN_STD[0], self.opt.XRAY2_MEAN_STD[1]],
            [self.opt.XRAY1_MEAN_STD[0], self.opt.XRAY1_MEAN_STD[1]],
            [self.opt.XRAY2_MEAN_STD[0], self.opt.XRAY2_MEAN_STD[1]],
            [self.opt.XRAY1_MEAN_STD[0], self.opt.XRAY1_MEAN_STD[1]],
            [self.opt.XRAY2_MEAN_STD[0], self.opt.XRAY2_MEAN_STD[1]]
        ]

    def init_loss(self, opt):
        Base_Model.init_loss(self, opt)

        # #####################
        # define loss functions
        # #####################
        # GAN loss
        if opt.ganloss == 'gan':
            self.criterionGAN = GANLoss(use_lsgan=False).to(self.device)
        elif opt.ganloss == 'lsgan':
            self.criterionGAN = GANLoss(use_lsgan=True).to(self.device)
        elif opt.ganloss == 'wgan':
            self.criterionGAN = WGANLoss(grad_penalty=False).to(self.device)
        elif opt.ganloss == 'wgan_gp':
            self.criterionGAN = WGANLoss(grad_penalty=True).to(self.device)
        else:
            raise ValueError()

        # identity loss
        self.criterionIdt = RestructionLoss(opt.idt_loss, opt.idt_reduction).to(self.device)

        # feature metric loss
        self.criterionFea = torch.nn.L1Loss().to(self.device)

        # map loss
        self.criterionMap = RestructionLoss(opt.map_projection_loss).to(self.device)

        # auxiliary loss
        self.criterionAuxiliary = RestructionLoss(opt.auxiliary_loss).to(self.device)

        # #####################
        # initialize optimizers
        # #####################
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

    '''
      Train -Forward and Backward
    '''

    def set_input(self, input):
        self.G_input1 = input[1][0].to(self.device)
        self.G_input2 = input[1][1].to(self.device)
        self.G_real = input[0].to(self.device)
        self.image_paths = input[2:]


    def set_ProST(self): # 使用target = projmodel(CT_vol, ray_proj_mov, rtvec, corner_pt)
        CT_vol_shape = self.opt.output_shape
        # Calculate geometric parameters
        param, det_size, self.norm_factor = calculate_param(CT_vol_shape, vol_spacing=1) # 这一步只用了shape
        self.ray_proj_mov, self.corner_pt = detector_corner_prepare(det_size, device='cuda')
        # Initialize projection model
        self.projmodel = ProST(param).to(self.device)

    def set_pose(self):
        PI = 3.1415926
        ########## Hard Code test groundtruth and initialize poses ##########
        # [rx, ry, rz, tx, ty, tz]
        # 第一个参数影像绕着左右的y轴旋转，也是从上往下，还是从下往上
        # 第二个参数影像环绕z轴旋转
        # 第三个参数影像环绕着前后的x轴旋转，即从上往下，还是从下往上
        mu, sigma_r, sigma_t = 0, 10, 8
        if self.training:
            rotate_a, rotate_b, rotate_c = np.random.normal(mu, sigma_r, 1)[0], np.random.normal(mu, sigma_r, 1)[0], np.random.normal(mu, sigma_r, 1)[0]
            translate_a, translate_b, translate_c = np.random.normal(mu, sigma_t, 1)[0], np.random.normal(mu, sigma_t, 1)[0], np.random.normal(mu, sigma_t, 1)[0]
        else:
            rotate_a, rotate_b, rotate_c = 0, 0, 0
            translate_a, translate_b, translate_c = 0, 0, 0
        manual_rtvec_1 = np.array([[0.0 + rotate_a, 0 + rotate_b, 90.0 + rotate_c,
                                    0.0 + translate_a, 0.0 + translate_b, 0.0 + translate_c]]) # axial 横断面 上下
        manual_rtvec_2 = np.array([[90.0 + rotate_a, 270.0 + rotate_b, 90.0 + rotate_c,
                                    0.0 + translate_a, 0.0 + translate_b, 0.0 + translate_c]]) # sagittal 矢状面 左右
        manual_rtvec_3 = np.array([[90.0 + rotate_a, 180.0 + rotate_b, 90.0 + rotate_c,
                                    0.0 + translate_a, 0.0 + translate_b, 0.0 + translate_c]]) # coronal 冠状面 前后
        # print(rotate_a, rotate_b, rotate_c)
        # print(translate_a, translate_b, translate_c)
        # manual_rtvec_1 = np.array([[0.0, 0, 90.0,
        #                             0.0, 0.0, 0.0]]) # axial 横断面 上下
        # manual_rtvec_2 = np.array([[90.0, 270.0, 90.0,
        #                             0.0, 0.0, 0.0]]) # sagittal 矢状面 左右
        # manual_rtvec_3 = np.array([[90.0, 180.0, 90.0,
        #                             0.0, 0.0, 0.0]]) # coronal 冠状面 前后
        # Normalization and conversion to transformation matrix
        def normalize_rtvec(manual_rtvec, PI, norm_factor):
            manual_rtvec[:, :3] = manual_rtvec[:, :3] * PI / 180
            manual_rtvec[:, 3:] = manual_rtvec[:, 3:] / norm_factor
            return manual_rtvec

        manual_rtvec_1 = normalize_rtvec(manual_rtvec_1, PI, self.norm_factor)
        manual_rtvec_2 = normalize_rtvec(manual_rtvec_2, PI, self.norm_factor)
        manual_rtvec_3 = normalize_rtvec(manual_rtvec_3, PI, self.norm_factor)

        # 将真实角度，转换成vector向量矩阵
        self.rtvec_1 = rtvec_transform(self.device, manual_rtvec_smp=manual_rtvec_1)
        self.rtvec_2 = rtvec_transform(self.device, manual_rtvec_smp=manual_rtvec_2)
        self.rtvec_3 = rtvec_transform(self.device, manual_rtvec_smp=manual_rtvec_3)
        self.rtvec_list = (self.rtvec_1, self.rtvec_2, self.rtvec_3)

    def calculate_gradient_penalty(self, netD, real_data, fake_data, LAMBDA):
        assert real_data.size(0) == fake_data.size(0), 'Batch size is not consistent'

        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, requires_grad=True).expand_as(real_data).to(real_data)
        interpolates = (1 - alpha) * real_data + alpha * fake_data
        assert interpolates.requires_grad == True, 'input need to be derivable'

        disc_interpolates = netD(interpolates)[-1]

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones_like(disc_interpolates),
                                        create_graph=True, retain_graph=True, only_inputs=True,
                                        allow_unused=False)

        gradients = gradients[0].view(gradients[0].size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

        return gradient_penalty

    # map function
    def output_map(self, v, dim):
        '''
        :param v: tensor
        :param dim:  dimension be reduced
        :return:
          N1HW
        '''
        ori_dim = v.dim()
        # tensor [NDHW]
        if ori_dim == 4:
            map = torch.mean(torch.abs(v), dim=dim)
            # [NHW] => [NCHW]
            return map.unsqueeze(1)
        # tensor [NCDHW] and c==1
        elif ori_dim == 5:
            # [NCHW]
            map = torch.mean(torch.abs(v), dim=dim)
            return map
        else:
            raise NotImplementedError()

    # ProST function
    def prost_map(self, CT_vol, i):
        # CT_vol为CT图像，ray_proj_mov为空的探测板[1,1,128,128]，传入进来只是为了获得探测板的大小
        # rtvec_gt为真实位移矩阵，corner_pt角点坐标
        ori_dim = CT_vol.dim()
        # tensor [NDHW]
        if ori_dim == 5:
            target = self.projmodel(CT_vol, self.ray_proj_mov, self.rtvec_list[i], self.corner_pt)
        else:
            CT_vol = CT_vol.unsqueeze(1)
            target = self.projmodel(CT_vol, self.ray_proj_mov, self.rtvec_list[i], self.corner_pt)
        return target


    def transition(self, predict):
        p_max, p_min = predict.max(), predict.min()
        new_predict = (predict - p_min) / (p_max - p_min)
        return new_predict

    def ct_unGaussian(self, value):
        return value * self.opt.CT_MEAN_STD[1] + self.opt.CT_MEAN_STD[0]

    def ct_Gaussian(self, value):
        return (value - self.opt.CT_MEAN_STD[0]) / self.opt.CT_MEAN_STD[1]

    # def post_process(self, attributes_name): # 估计这个纯粹是为了visualizer里面的tensor_to_image，网页显示
    #   if not self.training:
    #     if self.opt.CT_MEAN_STD[0] == 0 and self.opt.CT_MEAN_STD[0] == 0: # 这个and前后是一模一样的
    #       for name in attributes_name:
    #         setattr(self, name, torch.clamp(getattr(self, name), 0, 1))
    #     elif self.opt.CT_MEAN_STD[0] == 0.5 and self.opt.CT_MEAN_STD[0] == 0.5: # 这个and前后是一模一样的
    #       for name in attributes_name:
    #         setattr(self, name, torch.clamp(getattr(self, name), -1, 1))
    #     else:
    #       print("multiView_CTGAN.py: NotImplementedError post_process")
    #       # raise NotImplementedError()

    # def projection_visual(self):
    #     # map F is projected in dimension of H
    #     self.G_Map_real_F = self.transition(self.output_map(self.ct_unGaussian(self.G_real), 2))
    #     self.G_Map_fake_F = self.transition(self.output_map(self.ct_unGaussian(self.G_fake), 2))
    #     # map S is projected in dimension of W
    #     self.G_Map_real_S = self.transition(self.output_map(self.ct_unGaussian(self.G_real), 3))
    #     self.G_Map_fake_S = self.transition(self.output_map(self.ct_unGaussian(self.G_fake), 3))

    def projection_visual(self):
        self.set_pose()
        # 通过旋转角度控制三个面的投影
        self.G_Map_real_axial = self.transition(self.prost_map(self.ct_unGaussian(self.G_real), 0))
        self.G_Map_fake_axial = self.transition(self.prost_map(self.ct_unGaussian(self.G_fake), 0))

        self.G_Map_real_sagittal = self.transition(self.prost_map(self.ct_unGaussian(self.G_real), 1))
        self.G_Map_fake_sagittal = self.transition(self.prost_map(self.ct_unGaussian(self.G_fake), 1))

        self.G_Map_real_coronal = self.transition(self.prost_map(self.ct_unGaussian(self.G_real), 2))
        self.G_Map_fake_coronal = self.transition(self.prost_map(self.ct_unGaussian(self.G_fake), 2))

    def metrics_evaluation(self):
        # 3D metrics including mse, cs and psnr
        g_fake_unNorm = self.ct_unGaussian(self.G_fake)
        g_real_unNorm = self.ct_unGaussian(self.G_real)

        self.metrics_Mse = Metrics.Mean_Squared_Error(g_fake_unNorm, g_real_unNorm)
        self.metrics_CosineSimilarity = Metrics.Cosine_Similarity(g_fake_unNorm, g_real_unNorm)
        self.metrics_PSNR = Metrics.Peak_Signal_to_Noise_Rate(g_fake_unNorm, g_real_unNorm, PIXEL_MAX=1.0)

    def dimension_order_std(self, value, order):
        # standard CT dimension
        return value.permute(*tuple(np.argsort(order)))

    def forward(self):
        '''
        self.G_fake is generated object
        self.G_real is GT object
        '''
        # G_fake_D is [B 1 D H W]
        # if self.num_forward % 100 == 0 or self.num_forward % 100 == 1:
        #     print()
        #     print("netG", [i for i in self.netG.parameters()][0][32, 0, 3])
        #     print("emaGet", self.netG_ema._get_parameters(None)[0][32, 0, 3])
        #     print("emaCurrent", [i for i in self.netG_ema.shadow_params][0][32, 0, 3])
        if not self.training and self.if_ema: # 在测试，且用ema的时候
            with self.netG_ema.average_parameters():
                self.G_fake_D1, self.G_fake_D2, self.G_fake_D = self.netG([self.G_input1, self.G_input2])
        else:
            self.G_fake_D1, self.G_fake_D2, self.G_fake_D = self.netG([self.G_input1, self.G_input2])
        # visual object should be [B D H W]
        self.G_fake = torch.squeeze(self.G_fake_D, 1)
        # input of Discriminator is [B 1 D H W]
        self.G_real_D = torch.unsqueeze(self.G_real, 1)
        # if add auxiliary loss to generator
        if self.auxiliary_loss and self.training:  # 示例yaml中会跳过该步骤
            self.G_real_D1 = self.G_real_D.permute(*self.opt.CTOrder_Xray1).detach()
            self.G_real_D2 = self.G_real_D.permute(*self.opt.CTOrder_Xray2).detach()
        # if add condition to discriminator, expanding x-ray
        # as the same shape and dimension order as CT
        if self.conditional_D and self.training:
            self.G_condition_D = torch.cat((
                self.dimension_order_std(self.G_input1.unsqueeze(1).expand_as(self.G_real_D), self.opt.CTOrder_Xray1),
                self.dimension_order_std(self.G_input2.unsqueeze(1).expand_as(self.G_real_D), self.opt.CTOrder_Xray2)
            ), dim=1).detach()
        # post processing, used only in testing
        # self.post_process(['G_fake'])
        if not self.training:
            # visualization of x-ray projection
            self.projection_visual()
            # metrics
            self.metrics_evaluation()
        # multi-view projection maps for training
        # Note: self.G_real_D and self.G_fake_D are in dimension order of 'NCDHW'
        # if self.training:
        #     for i in self.multi_view:
        #         out_map = self.output_map(self.ct_unGaussian(self.G_real_D), i + 1)
        #         out_map = self.ct_Gaussian(out_map)
        #         setattr(self, 'G_Map_{}_real'.format(i), out_map) # 原本的shape [1, 1, 128, 128]
        #
        #         out_map = self.output_map(self.ct_unGaussian(self.G_fake_D), i + 1)
        #         out_map = self.ct_Gaussian(out_map)
        #         setattr(self, 'G_Map_{}_fake'.format(i), out_map)
        if self.training:
            self.set_pose()
            for i in self.multi_view:
                out_map = self.prost_map(self.ct_unGaussian(self.G_real_D), i - 1)
                out_map = self.ct_Gaussian(out_map)
                setattr(self, 'G_Map_{}_real'.format(i), out_map) # 原本的shape [1, 1, 128, 128]

                out_map = self.prost_map(self.ct_unGaussian(self.G_fake_D), i - 1)
                out_map = self.ct_Gaussian(out_map)
                setattr(self, 'G_Map_{}_fake'.format(i), out_map)

    # feature metrics loss
    def feature_metric_loss(self, D_fake_pred, D_real_pred, loss_weight, num_D, feat_weights, criterionFea):
        fea_m_lambda = loss_weight
        loss_G_fea = 0
        feat_weights = feat_weights
        D_weights = 1.0 / num_D
        weight = feat_weights * D_weights

        # multi-scale discriminator
        if isinstance(D_fake_pred[0], list):
            for di in range(num_D):
                for i in range(len(D_fake_pred[di]) - 1):
                    loss_G_fea += weight * criterionFea(D_fake_pred[di][i], D_real_pred[di][i].detach()) * fea_m_lambda
        # single discriminator
        else:
            for i in range(len(D_fake_pred) - 1):
                loss_G_fea += feat_weights * criterionFea(D_fake_pred[i], D_real_pred[i].detach()) * fea_m_lambda

        return loss_G_fea

    def backward_D_basic(self, D_network, input_real, input_fake, fake_pool, criterionGAN, loss_weight):
        if self.opt.ganloss == 'gan' or self.opt.ganloss == 'lsgan':
            D_real_pred = D_network(input_real)
            gan_loss_real = criterionGAN(D_real_pred, True)

            if self.if_pool:  # 此处是false
                g_fake_pool = fake_pool.query(input_fake)
            else:
                g_fake_pool = input_fake
            D_fake_pool_pred = D_network(g_fake_pool.detach())
            gan_loss_fake = criterionGAN(D_fake_pool_pred, False)

            gan_loss = (gan_loss_fake + gan_loss_real) * loss_weight
            gan_loss.backward()
        elif self.opt.ganloss == 'wgan':
            D_real_pred = D_network(input_real)
            if self.if_pool:  # 此处是false
                g_fake_pool = fake_pool.query(input_fake)
            else:
                g_fake_pool = input_fake
            D_fake_pool_pred = D_network(g_fake_pool.detach())
            gan_loss = torch.mean(D_fake_pool_pred[-1]) - torch.mean(D_real_pred[-1])

            gan_loss.backward()
        elif self.opt.ganloss == 'wgan_gp':
            D_real_pred = D_network(input_real)
            if self.if_pool:  # 此处是false
                g_fake_pool = fake_pool.query(input_fake)
            else:
                g_fake_pool = input_fake
            D_fake_pool_pred = D_network(g_fake_pool.detach())
            gan_loss = torch.mean(D_fake_pool_pred[-1]) - torch.mean(D_real_pred[-1])
            self.loss_grad_penalty = self.calculate_gradient_penalty(D_network, input_real, g_fake_pool.detach(),
                                                                self.opt.wgan_gp_lambda)

            gan_loss += self.loss_grad_penalty

            gan_loss.backward(retain_graph=True)
        return gan_loss

    def backward_D(self):
        if self.conditional_D:
            fake_input = torch.cat([self.G_condition_D, self.G_fake_D], 1)
            real_input = torch.cat([self.G_condition_D, self.G_real_D], 1)
        else:
            fake_input = self.G_fake_D
            real_input = self.G_real_D
        self.loss_D = self.backward_D_basic(self.netD, real_input, fake_input, self.fake_pool if self.if_pool else None,
                                            self.criterionGAN, self.opt.gan_lambda)

    def backward_G_basic(self, D_network, input_real, input_fake, criterionGAN, loss_weight):
        D_fake_pred = D_network(input_fake)
        D_real_pred = D_network(input_real)
        if self.opt.ganloss == 'gan' or self.opt.ganloss == 'lsgan':
            loss_G = criterionGAN(D_fake_pred, True) * loss_weight
        elif self.opt.ganloss == 'wgan':
            loss_G = - torch.mean(D_fake_pred[-1])
            self.loss_wasserstein = torch.mean(D_real_pred[-1]) - torch.mean(D_fake_pred[-1])
        elif self.opt.ganloss == 'wgan_gp':
            loss_G = - torch.mean(D_fake_pred[-1])
            self.loss_wasserstein = torch.mean(D_real_pred[-1]) - torch.mean(D_fake_pred[-1])
        else:
            raise ValueError()
        return loss_G, D_fake_pred

    def backward_G(self):
        idt_lambda = self.opt.idt_lambda
        fea_m_lambda = self.opt.feature_D_lambda
        map_m_lambda = self.opt.map_projection_lambda

        ############################################
        # BackBone GAN
        ############################################
        # GAN loss
        if self.conditional_D:
            fake_input = torch.cat([self.G_condition_D, self.G_fake_D], 1)
            real_input = torch.cat([self.G_condition_D, self.G_real_D], 1)
        else:
            fake_input = self.G_fake_D
            real_input = self.G_real_D

        self.loss_G, D_fake_pred = self.backward_G_basic(self.netD, real_input, fake_input, self.criterionGAN, self.opt.gan_lambda)

        # identity loss
        if idt_lambda > 0: # 值为10
            # focus area weight assignment
            if self.opt.idt_reduction == 'none' and self.opt.idt_weight > 0:  # 前者是'elementwise_mean'，后者是0
                idt_low, idt_high = self.opt.idt_weight_range
                idt_weight = self.opt.idt_weight
                loss_idt = self.criterionIdt(self.G_fake_D, self.G_real_D)
                mask = (self.G_real_D > idt_low) & (self.G_real_D < idt_high)
                loss_idt[mask] = loss_idt[mask] * idt_weight
                self.loss_idt = loss_idt.mean() * idt_lambda
            else:
                self.loss_idt = self.criterionIdt(self.G_fake_D, self.G_real_D) * idt_lambda

        D_real_pred = self.netD(real_input)

        # feature metric loss 值为0，没用到
        if fea_m_lambda > 0:
            self.loss_fea_m = self.feature_metric_loss(D_fake_pred, D_real_pred, loss_weight=fea_m_lambda,
                                                       num_D=self.opt.num_D,
                                                       feat_weights=4.0 / (self.opt.n_layers_D + 1),
                                                       criterionFea=self.criterionFea)

        # map loss 值为10
        if map_m_lambda > 0:
            self.loss_map_m = 0.
            for direction in self.multi_view:
                self.loss_map_m += self.criterionMap(
                    getattr(self, 'G_Map_{}_fake'.format(direction)),
                    getattr(self, 'G_Map_{}_real'.format(direction))) * map_m_lambda
            self.loss_map_m = self.loss_map_m / len(self.multi_view)

        # auxiliary loss
        if self.auxiliary_loss:
            self.loss_auxiliary = (self.criterionAuxiliary(self.G_fake_D1, self.G_real_D1) +
                                   self.criterionAuxiliary(self.G_fake_D2, self.G_real_D2)) * \
                                  self.opt.auxiliary_lambda

        # 0.0 must be add to loss_total, otherwise loss_G will
        # be changed when loss_total_G change
        self.loss_total_G = self.loss_G + 0.0
        if idt_lambda > 0:
            self.loss_total_G += self.loss_idt
        if fea_m_lambda > 0:
            self.loss_total_G += self.loss_fea_m
        if map_m_lambda > 0:
            self.loss_total_G += self.loss_map_m
        if self.auxiliary_loss:
            self.loss_total_G += self.loss_auxiliary
        self.loss_total_G.backward()

    def optimize_parameters(self):
        # forward
        self() # 会默认进行forward
        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        if self.if_ema:
            self.netG_ema.update()
            # if self.num_forward % 100 == 0:
            #     # print("Update ema:{}".format(self.num_forward))
            #     self.netG_ema.update()
        self.num_forward += 1


    def optimize_D(self):
        # forward
        self()
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
