#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 7/31/2020 1:43 PM
# @Author : Zhicheng Zhang
# @E-mail : zhicheng0623@gmail.com
# @Site :
# @File : train_main.py
# @Software: PyCharm
# from apex import amp
# import torch.cuda.nvtx as nvtx
import torch.nn.utils.prune as prune
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np

import os
from os import path
from PIL import Image

from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import torch.cuda.amp as amp
# from dataload_optimization import CTDataset
# vizualize_folder = "./visualize"
# loss_folder = "./loss"
# reconstructed_images = "reconstructed_images"


INPUT_CHANNEL_SIZE = 1

def ln_struc_spar(model):
    parm = []
    for name, module in model.named_modules():
        if hasattr(module, "weight") and hasattr(module.weight, "requires_grad"):
                parm.append((module, "weight"))
                # parm.append((module, "bias"))
    for item in parm:
        try:
            prune.ln_structured(item[0], amount=0.5, name="weight", n=1, dim=0)
        except Exception as e:
            print('Error pruning: ', item[1], "exception: ", e)
    for module, name in parm:
        try:
            prune.remove(module, "weight")
            prune.remove(module, "bias")
        except  Exception as e:
            print('error pruning weight/bias for ',name,  e)
    print('pruning operation finished')

def structured_sparsity(model):
    parm = []
    for name, module in model.named_modules():
        if "conv" in name:
            if hasattr(module, "weight") and hasattr(module.weight, "requires_grad"):
                parm.append((module, "weight"))
                # parm.append((module, "bias"))

    # layerwise_sparsity(model,0.3)

    for item in parm:
        try:
            prune.random_structured(item[0], amount=0.5, name="weight", dim=0)
            prune.random_unstructured(item[0], amount=0.5, name="bias")

        except Exception as e:
            print('Error pruning: ', item[1], "exception: ", e)
    for module, name in parm:
        try:
            prune.remove(module, "weight")
            prune.remove(module, "bias")
        except  Exception as e:
            print('error pruing as ', e)

def module_sparsity(module : nn.Module, usemasks = False):
    z  =0.0
    n  = 0
    if usemasks == True:
        for bname, bu in module.named_buffers():
            if "weight_mask" in bname:
                z += torch.sum(bu == 0).item()
                n += bu.nelement()
            if "bias_mask" in bname:
                z += torch.sum(bu == 0).item()
                n += bu.nelement()

    else:
        for name,p in module.named_parameters():
            if "weight" in name :
                z += torch.sum(p==0).item()
                n += p.nelement()
            if "bias" in name:
                z+= torch.sum(p==0).item()
                n += p.nelement()
    return  n , z

def calculate_global_sparsity(model: nn.Module):
    total_zeros = 0.0
    total_n = 0.0

    # global_sparsity = 100 * total_n / total_nonzero
    for name,m in model.named_modules():
        n , z = module_sparsity(m)
        total_zeros += z
        total_n += n


    global_sparsity = 100  * ( total_zeros  / total_n
                               )
    # global_sparsity = (
    #     100.0
    #     * float(
    #         torch.sum(model.conv1.weight == 0)
    #         + torch.sum(model.conv2.weight == 0)
    #         + torch.sum(model.fc1.weight == 0)
    #         + torch.sum(model.fc2.weight == 0)
    #     )
    #     / float(
    #         model.conv1.weight.nelement()
    #         + model.conv2.weight.nelement()
    #         + model.fc1.weight.nelement()
    #         + model.fc2.weight.nelement()
    #     )
    # )

    global_compression = 100 / (100 - global_sparsity)
    print('global sparsity', global_sparsity, 'global compression: ',global_compression)
    return global_sparsity, global_compression

def count_parameters(model):
    #print("Modules  Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param
    return total_params



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window




def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.001:
            min_val = -0.1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (batch, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)

        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    # TODO: remove support for normalize == True (kept for backward support)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    if (torch.isnan(output)):
        print("NAN")
        print(pow1)
        print(pow2)
        print(ssims)
        print(mcs)
        exit()

    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize="simple")
        # return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


class denseblock(nn.Module):
    def __init__(self, nb_filter=16, filter_wh=5):
        super(denseblock, self).__init__()
        self.input = None  ######CHANGE
        self.nb_filter = nb_filter
        self.nb_filter_wh = filter_wh
        ##################CHANGE###############
        self.conv1_0 = nn.Conv2d(in_channels=nb_filter, out_channels=self.nb_filter * 4, kernel_size=1)
        self.conv2_0 = nn.Conv2d(in_channels=self.conv1_0.out_channels, out_channels=self.nb_filter,
                                 kernel_size=self.nb_filter_wh, padding=(2, 2))
        self.conv1_1 = nn.Conv2d(in_channels=nb_filter + self.conv2_0.out_channels, out_channels=self.nb_filter * 4,
                                 kernel_size=1)
        self.conv2_1 = nn.Conv2d(in_channels=self.conv1_1.out_channels, out_channels=self.nb_filter,
                                 kernel_size=self.nb_filter_wh, padding=(2, 2))
        self.conv1_2 = nn.Conv2d(in_channels=nb_filter + self.conv2_0.out_channels + self.conv2_1.out_channels,
                                 out_channels=self.nb_filter * 4, kernel_size=1)
        self.conv2_2 = nn.Conv2d(in_channels=self.conv1_2.out_channels, out_channels=self.nb_filter,
                                 kernel_size=self.nb_filter_wh, padding=(2, 2))
        self.conv1_3 = nn.Conv2d(
            in_channels=nb_filter + self.conv2_0.out_channels + self.conv2_1.out_channels + self.conv2_2.out_channels,
            out_channels=self.nb_filter * 4, kernel_size=1)
        self.conv2_3 = nn.Conv2d(in_channels=self.conv1_3.out_channels, out_channels=self.nb_filter,
                                 kernel_size=self.nb_filter_wh, padding=(2, 2))
        self.conv1 = [self.conv1_0, self.conv1_1, self.conv1_2, self.conv1_3]
        self.conv2 = [self.conv2_0, self.conv2_1, self.conv2_2, self.conv2_3]

        self.batch_norm1_0 = nn.BatchNorm2d(nb_filter)
        self.batch_norm2_0 = nn.BatchNorm2d(self.conv1_0.out_channels)
        self.batch_norm1_1 = nn.BatchNorm2d(nb_filter + self.conv2_0.out_channels)
        self.batch_norm2_1 = nn.BatchNorm2d(self.conv1_1.out_channels)
        self.batch_norm1_2 = nn.BatchNorm2d(nb_filter + self.conv2_0.out_channels + self.conv2_1.out_channels)
        self.batch_norm2_2 = nn.BatchNorm2d(self.conv1_2.out_channels)
        self.batch_norm1_3 = nn.BatchNorm2d(
            nb_filter + self.conv2_0.out_channels + self.conv2_1.out_channels + self.conv2_2.out_channels)
        self.batch_norm2_3 = nn.BatchNorm2d(self.conv1_3.out_channels)

        self.batch_norm1 = [self.batch_norm1_0, self.batch_norm1_1, self.batch_norm1_2, self.batch_norm1_3]
        self.batch_norm2 = [self.batch_norm2_0, self.batch_norm2_1, self.batch_norm2_2, self.batch_norm2_3]

    # def Forward(self, inputs):
    def forward(self, inputs):  ######CHANGE
        # x = self.input
        x = inputs
        # for i in range(4):
        #    #conv = nn.BatchNorm2d(x.size()[1])(x)
        #    conv = self.batch_norm1[i](x)
        #    #if(self.conv1[i].weight.grad != None ):
        #    #    print("weight_grad_" + str(i) + "_1", self.conv1[i].weight.grad.max())
        #    conv = self.conv1[i](conv)      ######CHANGE
        #    conv = F.leaky_relu(conv)

        #    #conv = nn.BatchNorm2d(conv.size()[1])(conv)
        #    conv = self.batch_norm2[i](conv)
        #    #if(self.conv2[i].weight.grad != None ):
        #    #    print("weight_grad_" + str(i) + "_2", self.conv2[i].weight.grad.max())
        #    conv = self.conv2[i](conv)      ######CHANGE
        #    conv = F.leaky_relu(conv)
        #    x = torch.cat((x, conv),dim=1)
        nvtx.range_push("dense block 1 forward")
        conv_1 = self.batch_norm1_0(x)
        conv_1 = self.conv1_0(conv_1)
        conv_1 = F.leaky_relu(conv_1)
        conv_2 = self.batch_norm2_0(conv_1)
        conv_2 = self.conv2_0(conv_2)
        conv_2 = F.leaky_relu(conv_2)
        nvtx.range_pop()

        nvtx.range_push("dense block 2 forward")
        x = torch.cat((x, conv_2), dim=1)
        conv_1 = self.batch_norm1_1(x)
        conv_1 = self.conv1_1(conv_1)
        conv_1 = F.leaky_relu(conv_1)
        conv_2 = self.batch_norm2_1(conv_1)
        conv_2 = self.conv2_1(conv_2)
        conv_2 = F.leaky_relu(conv_2)
        nvtx.range_pop()
        
        nvtx.range_push("dense block 1 forward")
        x = torch.cat((x, conv_2), dim=1)
        conv_1 = self.batch_norm1_2(x)
        conv_1 = self.conv1_2(conv_1)
        conv_1 = F.leaky_relu(conv_1)
        conv_2 = self.batch_norm2_2(conv_1)
        conv_2 = self.conv2_2(conv_2)
        conv_2 = F.leaky_relu(conv_2)
        nvtx.range_pop()

        nvtx.range_push("dense block 1 forward")
        x = torch.cat((x, conv_2), dim=1)
        conv_1 = self.batch_norm1_3(x)
        conv_1 = self.conv1_3(conv_1)
        conv_1 = F.leaky_relu(conv_1)
        conv_2 = self.batch_norm2_3(conv_1)
        conv_2 = self.conv2_3(conv_2)
        conv_2 = F.leaky_relu(conv_2)
        x = torch.cat((x, conv_2), dim=1)
        nvtx.range_pop()

        return x


class DD_net(nn.Module):
    def __init__(self):
        super(DD_net, self).__init__()
        self.input = None  #######CHANGE
        self.nb_filter = 16

        ##################CHANGE###############
        self.conv1 = nn.Conv2d(in_channels=INPUT_CHANNEL_SIZE, out_channels=self.nb_filter, kernel_size=(7, 7),
                               padding=(3, 3))
        self.dnet1 = denseblock(self.nb_filter, filter_wh=5)
        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels * 5, out_channels=self.nb_filter, kernel_size=(1, 1))
        self.dnet2 = denseblock(self.nb_filter, filter_wh=5)
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels * 5, out_channels=self.nb_filter, kernel_size=(1, 1))
        self.dnet3 = denseblock(self.nb_filter, filter_wh=5)
        self.conv4 = nn.Conv2d(in_channels=self.conv3.out_channels * 5, out_channels=self.nb_filter, kernel_size=(1, 1))
        self.dnet4 = denseblock(self.nb_filter, filter_wh=5)

        self.conv5 = nn.Conv2d(in_channels=self.conv4.out_channels * 5, out_channels=self.nb_filter, kernel_size=(1, 1))

        self.convT1 = nn.ConvTranspose2d(in_channels=self.conv4.out_channels + self.conv4.out_channels,
                                         out_channels=2 * self.nb_filter, kernel_size=5, padding=(2, 2))
        self.convT2 = nn.ConvTranspose2d(in_channels=self.convT1.out_channels, out_channels=self.nb_filter,
                                         kernel_size=1)
        self.convT3 = nn.ConvTranspose2d(in_channels=self.convT2.out_channels + self.conv3.out_channels,
                                         out_channels=2 * self.nb_filter, kernel_size=5, padding=(2, 2))
        self.convT4 = nn.ConvTranspose2d(in_channels=self.convT3.out_channels, out_channels=self.nb_filter,
                                         kernel_size=1)
        self.convT5 = nn.ConvTranspose2d(in_channels=self.convT4.out_channels + self.conv2.out_channels,
                                         out_channels=2 * self.nb_filter, kernel_size=5, padding=(2, 2))
        self.convT6 = nn.ConvTranspose2d(in_channels=self.convT5.out_channels, out_channels=self.nb_filter,
                                         kernel_size=1)
        self.convT7 = nn.ConvTranspose2d(in_channels=self.convT6.out_channels + self.conv1.out_channels,
                                         out_channels=2 * self.nb_filter, kernel_size=5, padding=(2, 2))
        self.convT8 = nn.ConvTranspose2d(in_channels=self.convT7.out_channels, out_channels=1, kernel_size=1)
        self.batch1 = nn.BatchNorm2d(1)
        self.max1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.batch2 = nn.BatchNorm2d(self.nb_filter * 5)
        self.max2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.batch3 = nn.BatchNorm2d(self.nb_filter * 5)
        self.max3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.batch4 = nn.BatchNorm2d(self.nb_filter * 5)
        self.max4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.batch5 = nn.BatchNorm2d(self.nb_filter * 5)

        self.batch6 = nn.BatchNorm2d(self.conv5.out_channels + self.conv4.out_channels)
        self.batch7 = nn.BatchNorm2d(self.convT1.out_channels)
        self.batch8 = nn.BatchNorm2d(self.convT2.out_channels + self.conv3.out_channels)
        self.batch9 = nn.BatchNorm2d(self.convT3.out_channels)
        self.batch10 = nn.BatchNorm2d(self.convT4.out_channels + self.conv2.out_channels)
        self.batch11 = nn.BatchNorm2d(self.convT5.out_channels)
        self.batch12 = nn.BatchNorm2d(self.convT6.out_channels + self.conv1.out_channels)
        self.batch13 = nn.BatchNorm2d(self.convT7.out_channels)

    # def Forward(self, inputs):
    def forward(self, inputs):
        self.input = inputs
        # print("Size of input: ", inputs.size())
        # conv = nn.BatchNorm2d(self.input)
        conv = self.batch1(self.input)  #######CHANGE
        # conv = nn.Conv2d(in_channels=conv.get_shape().as_list()[1], out_channels=self.nb_filter, kernel_size=(7, 7))(conv)
        conv = self.conv1(conv)  #####CHANGE
        c0 = F.leaky_relu(conv)

        p0 = self.max1(c0)
        D1 = self.dnet1(p0)

        #######################################################################################
        conv = self.batch2(D1)
        conv = self.conv2(conv)
        c1 = F.leaky_relu(conv)

        p1 = self.max2(c1)
        D2 = self.dnet2(p1)
        #######################################################################################

        conv = self.batch3(D2)
        conv = self.conv3(conv)
        c2 = F.leaky_relu(conv)

        p2 = self.max3(c2)
        D3 = self.dnet3(p2)
        #######################################################################################

        conv = self.batch4(D3)
        conv = self.conv4(conv)
        c3 = F.leaky_relu(conv)

        # p3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0)(c3)
        p3 = self.max4(c3)  ######CHANGE
        D4 = self.dnet4(p3)

        conv = self.batch5(D4)
        conv = self.conv5(conv)
        c4 = F.leaky_relu(conv)

        x = torch.cat((nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(c4), c3), dim=1)
        dc4 = F.leaky_relu(self.convT1(self.batch6(x)))  ######size() CHANGE
        dc4_1 = F.leaky_relu(self.convT2(self.batch7(dc4)))

        x = torch.cat((nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(dc4_1), c2), dim=1)
        dc5 = F.leaky_relu(self.convT3(self.batch8(x)))
        dc5_1 = F.leaky_relu(self.convT4(self.batch9(dc5)))

        x = torch.cat((nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(dc5_1), c1), dim=1)
        dc6 = F.leaky_relu(self.convT5(self.batch10(x)))
        dc6_1 = F.leaky_relu(self.convT6(self.batch11(dc6)))

        x = torch.cat((nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(dc6_1), c0), dim=1)
        dc7 = F.leaky_relu(self.convT7(self.batch12(x)))
        dc7_1 = F.leaky_relu(self.convT8(self.batch13(dc7)))

        output = dc7_1

        return output


def gen_visualization_files(outputs, targets, inputs, file_names, val_test, maxs, mins):
    mapped_root = "./visualize/" + val_test + "/mapped/"
    diff_target_out_root = "./visualize/" + val_test + "/diff_target_out/"
    diff_target_in_root = "./visualize/" + val_test + "/diff_target_in/"
    ssim_root = "./visualize/" + val_test + "/ssim/"
    out_root = "./visualize/" + val_test + "/"
    in_img_root = "./visualize/" + val_test + "/input/"
    out_img_root = "./visualize/" + val_test + "/target/"

    # if not os.path.exists("./visualize"):
    #     os.makedirs("./visualize")
    # if not os.path.exists(out_root):
    #    os.makedirs(out_root)
    # if not os.path.exists(mapped_root):
    #    os.makedirs(mapped_root)
    # if not os.path.exists(diff_target_in_root):
    #    os.makedirs(diff_target_in_root)
    # if not os.path.exists(diff_target_out_root):
    #    os.makedirs(diff_target_out_root)
    # if not os.path.exists(in_img_root):
    #    os.makedirs(in_img_root)
    # if not os.path.exists(out_img_root):
    #    os.makedirs(out_img_root)

    MSE_loss_out_target = []
    MSE_loss_in_target = []
    MSSSIM_loss_out_target = []
    MSSSIM_loss_in_target = []

    outputs_size = list(outputs.size())
    # num_img = outputs_size[0]
    (num_img, channel, height, width) = outputs.size()
    for i in range(num_img):
        # output_img = outputs[i, 0, :, :].cpu().detach().numpy()
        output_img = outputs[i, 0, :, :].cpu().detach().numpy()
        target_img = targets[i, 0, :, :].cpu().numpy()
        input_img = inputs[i, 0, :, :].cpu().numpy()

        output_img_mapped = (output_img * (maxs[i].item() - mins[i].item())) + mins[i].item()
        target_img_mapped = (target_img * (maxs[i].item() - mins[i].item())) + mins[i].item()
        input_img_mapped = (input_img * (maxs[i].item() - mins[i].item())) + mins[i].item()

        # target_img = targets[i, 0, :, :].cpu().numpy()
        # input_img = inputs[i, 0, :, :].cpu().numpy()

        file_name = file_names[i]
        file_name = file_name.replace(".IMA", ".tif")
        im = Image.fromarray(target_img_mapped)
        im.save(out_img_root + file_name)

        file_name = file_names[i]
        file_name = file_name.replace(".IMA", ".tif")
        im = Image.fromarray(input_img_mapped)
        im.save(in_img_root + file_name)
        # jy
        # im.save(folder_ori_HU+'/'+file_name)

        file_name = file_names[i]
        file_name = file_name.replace(".IMA", ".tif")
        im = Image.fromarray(output_img_mapped)
        im.save(mapped_root + file_name)
        # jy
        # im.save(folder_enh_HU+'/'+file_name)

        difference_target_out = (target_img - output_img)
        difference_target_out = np.absolute(difference_target_out)
        fig = plt.figure()
        plt.imshow(difference_target_out)
        plt.colorbar()
        plt.clim(0, 0.2)
        plt.axis('off')
        file_name = file_names[i]
        file_name = file_name.replace(".IMA", ".tif")
        fig.savefig(diff_target_out_root + file_name)
        plt.clf()
        plt.close()

        difference_target_in = (target_img - input_img)
        difference_target_in = np.absolute(difference_target_in)
        fig = plt.figure()
        plt.imshow(difference_target_in)
        plt.colorbar()
        plt.clim(0, 0.2)
        plt.axis('off')
        file_name = file_names[i]
        file_name = file_name.replace(".IMA", ".tif")
        fig.savefig(diff_target_in_root + file_name)
        plt.clf()
        plt.close()

        output_img = torch.reshape(outputs[i, 0, :, :], (1, 1, height, width))
        target_img = torch.reshape(targets[i, 0, :, :], (1, 1, height, width))
        input_img = torch.reshape(inputs[i, 0, :, :], (1, 1, height, width))

        MSE_loss_out_target.append(nn.MSELoss()(output_img, target_img))
        MSE_loss_in_target.append(nn.MSELoss()(input_img, target_img))
        MSSSIM_loss_out_target.append(1 - MSSSIM()(output_img, target_img))
        MSSSIM_loss_in_target.append(1 - MSSSIM()(input_img, target_img))

    with open(out_root + "msssim_loss_target_out", 'a') as f:
        for item in MSSSIM_loss_out_target:
            f.write("%f\n" % item)

    with open(out_root + "msssim_loss_target_in", 'a') as f:
        for item in MSSSIM_loss_in_target:
            f.write("%f\n" % item)

    with open(out_root + "mse_loss_target_out", 'a') as f:
        for item in MSE_loss_out_target:
            f.write("%f\n" % item)

    with open(out_root + "mse_loss_target_in", 'a') as f:
        for item in MSE_loss_in_target:
            f.write("%f\n" % item)




# jy
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

import nvidia_dlprof_pytorch_nvtx
nvidia_dlprof_pytorch_nvtx.init(enable_function_stack=True)
def cleanup():
    dist.destroy_process_group()
torch.backends.cudnn.benchmark=True
# import nvidia_dlprof_pytorch_nvtx
# nvidia_dlprof_pytorch_nvtx.init(enable_function_stack=True)
# from apex.contrib.sparsity import ASP
def dd_train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group("gloo", rank=rank, world_size=args.world_size)
    batch = args.batch
    epochs = args.epochs
    retrain = args.retrain
    amp_enabled = (args.amp == "enable")
    new_loader = (args.new_load == 'enable')
    global dir_pre
    dir_pre = args.out_dir
    num_w = args.num_w
    en_wan = args.wan
    print('amp: ', amp_enabled)
    print('num of workers: ', num_w)
    root_train_h = "/projects/synergy_lab/garvit217/enhancement_data/train/HQ/"
    root_train_l = "/projects/synergy_lab/garvit217/enhancement_data/train/LQ/"
    root_val_h = "/projects/synergy_lab/garvit217/enhancement_data/val/HQ/"
    root_val_l = "/projects/synergy_lab/garvit217/enhancement_data/val/LQ/"
    root_test_h = "/projects/synergy_lab/garvit217/enhancement_data/test/HQ/"
    root_test_l = "/projects/synergy_lab/garvit217/enhancement_data/test/LQ/"

    from data_loader.custom_load import CTDataset
    train_loader = CTDataset(root_train_h,root_train_l,5120,gpu,batch)
    test_loader = CTDataset(root_test_h,root_test_l,784,gpu,batch)
    val_loader = CTDataset(root_val_h,root_val_l,784,gpu,batch)
    model = DD_net()

    # torch.cuda.set_device(rank)
    # model.cuda(rank)
    model.to(gpu)
    model = DDP(model, device_ids=[gpu])
    learn_rate = 0.0001;
    epsilon = 1e-8

    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, eps=epsilon)  #######ADAM CHANGE
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O0")
    # model = DDP(model)
    # optimizer1 = torch.optim.Adam(model.dnet1.parameters(), lr=learn_rate, eps=epsilon)     #######ADAM CHANGE
    # optimizer2 = torch.optim.Adam(model.dnet2.parameters(), lr=learn_rate, eps=epsilon)     #######ADAM CHANGE
    # optimizer3 = torch.optim.Adam(model.dnet3.parameters(), lr=learn_rate, eps=epsilon)     #######ADAM CHANGE
    # optimizer4 = torch.optim.Adam(model.dnet4.parameters(), lr=learn_rate, eps=epsilon)     #######ADAM CHANGE
    decayRate = 0.95
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    # scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer1, gamma=decayRate)
    # scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer2, gamma=decayRate)
    # scheduler3 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer3, gamma=decayRate)
    # scheduler4 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer4, gamma=decayRate)

    # outputs = ddp_model(torch.randn(20, 10).to(rank))

    # max_train_img_init = 5120;
    # max_train_img_init = 32;
    # max_val_img_init = 784;
    # max_val_img_init = 16;
    # max_test_img = 784;

    train_MSE_loss = [0]
    train_MSSSIM_loss = [0]
    train_total_loss = [0]
    val_MSE_loss = [0]
    val_MSSSIM_loss = [0]
    val_total_loss = [0]
    test_MSE_loss = [0]
    test_MSSSIM_loss = [0]
    test_total_loss = [0]



    model_file = "weights_" + str(epochs) + "_" + str(batch) + ".pt"

    map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}

    if (not (path.exists(model_file))):
        with torch.autograd.profiler.emit_nvtx():
            train_eval_ddnet(epochs, gpu, model, optimizer, rank, scheduler, train_MSE_loss, train_MSSSIM_loss,
                         train_loader, train_total_loss, val_MSE_loss, val_MSSSIM_loss, val_loader,
                         val_total_loss, amp_enabled, retrain, en_wan)
        print("train end")
        serialize_trainparams(model, model_file, rank, train_MSE_loss, train_MSSSIM_loss, train_total_loss, val_MSE_loss,
                              val_MSSSIM_loss, val_total_loss)

    else:
        print("Loading model parameters")
        model.load_state_dict(torch.load(model_file, map_location=map_location))
        calculate_global_sparsity(model)
        if retrain > 0:
            model.load_state_dict(torch.load(model_file, map_location=map_location))
            print("sparifying the model....")
            ln_struc_spar(model)
            # ASP.prune_trained_model(model,optimizer)
            print('weights updated and masks removed... Model is sucessfully pruned')
            # create new OrderedDict that does not contain `module.`
            calculate_global_sparsity(model)
            print('fine tune retraining for ', retrain, ' epochs...')
            # with torch.autograd.profiler.emit_nvtx():
            train_eval_ddnet(epochs, gpu, model, optimizer, rank, scheduler, train_MSE_loss, train_MSSSIM_loss,
                              train_loader, train_total_loss, val_MSE_loss, val_MSSSIM_loss, val_loader,
                              val_total_loss, amp_enabled, retrain, en_wan)
    test_ddnet(gpu, model, test_loader, test_MSE_loss, test_MSSSIM_loss, test_total_loss, rank)
    print("testing end")
    with open('loss/test_MSE_loss_' + str(rank), 'w') as f:
        for item in test_MSE_loss:
            f.write("%f " % item)
    with open('loss/test_MSSSIM_loss_' + str(rank), 'w') as f:
        for item in test_MSSSIM_loss:
            f.write("%f " % item)
    with open('loss/test_total_loss_' + str(rank), 'w') as f:
        for item in test_total_loss:
            f.write("%f " % item)
    print("everything complete.......")

    print("Final avergae MSE: ", np.average(test_MSE_loss), "std dev.: ", np.std(test_MSE_loss))
    print("Final average MSSSIM LOSS: " + str(100 - (100 * np.average(test_MSSSIM_loss))), 'std dev : ', np.std(test_MSSSIM_loss))
    # psnr_calc(test_MSE_loss)


import torch.cuda.nvtx as nvtx
def train_eval_ddnet(epochs, gpu, model, optimizer, rank, scheduler, train_MSE_loss, train_MSSSIM_loss,
                     train_loader, train_total_loss, val_MSE_loss, val_MSSSIM_loss, val_loader,
                     val_total_loss, amp_enabled, retrain, en_wan):
    start = datetime.now()
    scaler = amp.GradScaler()
    sparsified = False
    for k in range(epochs + retrain):
        print("Training for Epocs: ", epochs+retrain)
        print('epoch: ', k, ' train loss: ', train_total_loss[k], ' mse: ', train_MSE_loss[k], ' mssi: ',
              train_MSSSIM_loss[k])
#         train_sampler.set_epoch(epochs)
        train_index_list = np.random.default_rng(seed=22).permutation(range(len(train_loader)))
        val_index_list = np.random.default_rng(seed=22).permutation(range(len(val_loader)))
        nvtx.range_push("Training")
        for idx in train_index_list:
            
            
            sample_batched = train_loader.get_item(idx)
            HQ_img, LQ_img, maxs, mins, file_name =  sample_batched['HQ'], sample_batched['LQ'], \
                                                        sample_batched['max'], sample_batched['min'], sample_batched['vol']
#             maxs, mins =  batch_samples['max'], batch_samples['min']
#             print(maxs)
#             print(mins)
            targets = HQ_img
            inputs = LQ_img
            if not sparsified:
                nvtx.range_push("Batch: " + str(idx))
            else:
                nvtx.range_push("Sp-Batch: " + str(idx))

            with amp.autocast(enabled=amp_enabled):
                nvtx.range_push("copy to device")
#                 if not new_loader == True:
#                     inputs = LQ_img.to(gpu)
#                     targets = HQ_img.to(gpu)
                
#                 targets = HQ_img.to(gpu)
                nvtx.range_pop()
    
                nvtx.range_push("forward pass,epoch:"+ str(k))

                outputs = model(inputs)
                MSE_loss = nn.MSELoss()(outputs, targets)
                MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
                loss = MSE_loss + 0.1 * (MSSSIM_loss)
                nvtx.range_pop()
            # print(outputs.shape)

            train_MSE_loss.append(MSE_loss.item())
            train_MSSSIM_loss.append(MSSSIM_loss.item())
            train_total_loss.append(loss.item())
            optimizer.zero_grad(set_to_none=True)
            # for param in model.parameters():
            #     param.grad = 0
            nvtx.range_push("backward pass")
            # model.zero_grad()

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            nvtx.range_pop()
            nvtx.range_pop()
            
            
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
        # print('loss: ',loss, ' mse: ', mse
        scheduler.step()
        nvtx.range_pop()
        print("Validation")
        nvtx.range_push("Validation")
        for idx in val_index_list:
            sample_batched = val_loader.get_item(idx)
            HQ_img, LQ_img, maxs, mins, fname =  sample_batched['HQ'], sample_batched['LQ'], \
                                                        sample_batched['max'], sample_batched['min'], sample_batched['vol']
            inputs = LQ_img
            targets = HQ_img
            with amp.autocast(enabled=amp_enabled):
#                 if not new_loader == True:
#                     inputs = LQ_img.to(gpu)
#                     targets = HQ_img.to(gpu)
#                 inputs = LQ_img.to(gpu)
#                 targets = HQ_img.to(gpu)
                outputs = model(inputs)
                # outputs = model(inputs)
                MSE_loss = nn.MSELoss()(outputs, targets)
                MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
                # loss = nn.MSELoss()(outputs , targets_val) + 0.1*(1-MSSSIM()(outputs,targets_val))
                loss = MSE_loss + 0.1 * (MSSSIM_loss)

            val_MSE_loss.append(MSE_loss.item())
            val_total_loss.append(loss.item())
            val_MSSSIM_loss.append(MSSSIM_loss.item())
            # print(len(outputs_np))
            # print("shape: ", outputs.shape)
            if (k == epochs - 1):
                if (rank == 0):
                    print("Training complete in: " + str(datetime.now() - start))
                # outputs_np = outputs.cpu().detach().numpy()
                # (batch_size, channel, height, width) = outputs.size()
                # for m in range(batch_size):
                #     file_name1 = file_name[m]
                #     file_name1 = file_name1.replace(".IMA", ".tif")
                #     im = Image.fromarray(outputs_np[m, 0, :, :])
                #     im.save('reconstructed_images/val/' + file_name1)
                #     # gen_visualization_files(outputs, targets, inputs, val_files[l_map:l_map+batch], "val")
                #     gen_visualization_files(outputs, targets, inputs, file_name, "val", maxs, mins)
        nvtx.range_pop()
        if  sparsified == False and retrain > 0 and k == (epochs-1) :
            print("dense training done for " + k + " epochs: " + " in : " , str(datetime.now()- start))
            print('pruning model')
            ln_struc_spar(model)
            print("sparse retraining now starting")
            sparsified = True
            print('pruning model on epoch: ', k)
        # sparsified = True

def serialize_trainparams(model, model_file, rank, train_MSE_loss, train_MSSSIM_loss, train_total_loss, val_MSE_loss,
                          val_MSSSIM_loss, val_total_loss):
    if (rank == 0):
        print("Saving model parameters")
        torch.save(model.state_dict(), model_file)
    with open('loss/train_MSE_loss_' + str(rank), 'w') as f:
        for item in train_MSE_loss:
            f.write("%f " % item)
    with open('loss/train_MSSSIM_loss_' + str(rank), 'w') as f:
        for item in train_MSSSIM_loss:
            f.write("%f " % item)
    with open('loss/train_total_loss_' + str(rank), 'w') as f:
        for item in train_total_loss:
            f.write("%f " % item)
    with open('loss/val_MSE_loss_' + str(rank), 'w') as f:
        for item in val_MSE_loss:
            f.write("%f " % item)
    with open('loss/val_MSSSIM_loss_' + str(rank), 'w') as f:
        for item in val_MSSSIM_loss:
            f.write("%f " % item)
    with open('loss/val_total_loss_' + str(rank), 'w') as f:
        for item in val_total_loss:
            f.write("%f " % item)


def test_ddnet(gpu, model,test_loader, test_MSE_loss, test_MSSSIM_loss, test_total_loss, rank):
    index_list = np.random.default_rng(seed=22).permutation(range(len(test_loader)))
    for idx in index_list:
        batch_samples = test_loader.get_item(idx)
        HQ_img, LQ_img, maxs, mins, file_name = batch_samples['HQ'], batch_samples['LQ'], \
                                                batch_samples['max'], batch_samples['min'], batch_samples['vol']
        inputs = LQ_img
        targets = HQ_img     
        outputs = model(inputs)
        MSE_loss = nn.MSELoss()(outputs, targets)
        MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
        # loss = nn.MSELoss()(outputs , targets_test) + 0.1*(1-MSSSIM()(outputs,targets_test))
        loss = MSE_loss + 0.1 * (MSSSIM_loss)
        # loss = MSE_loss
        print("MSE_loss", MSE_loss.item())
        print("MSSSIM_loss", MSSSIM_loss.item())
        print("Total_loss", loss.item())
        print("====================================")
        test_MSE_loss.append(MSE_loss.item())
        test_MSSSIM_loss.append(MSSSIM_loss.item())
        test_total_loss.append(loss.item())
        outputs_np = outputs.cpu().detach().numpy()
        (batch_size, channel, height, width) = outputs.size()
        for m in range(batch_size):
            file_name1 = file_name[m]
            file_name1 = file_name1.replace(".IMA", ".tif")
            im = Image.fromarray(outputs_np[m, 0, :, :])
            im.save('reconstructed_images/test/' + file_name1)
        # outputs.cpu()
        # targets_test[l_map:l_map+batch, :, :, :].cpu()
        # inputs_test[l_map:l_map+batch, :, :, :].cpu()
        # gen_visualization_files(outputs, targets, inputs, test_files[l_map:l_map+batch], "test" )
        gen_visualization_files(outputs, targets, inputs, file_name, "test", maxs, mins)


def psnr_calc(mse_t):
    psnr = []
    for i in range(len(mse_t)):
        #     x = read_correct_image(pa +"/"+ ll[i])
        mse_sqrt = pow(mse_t[i], 0.5)
        psnr_ = 20 * np.log10(1 / mse_sqrt)
        psnr.insert(i, psnr_)
    print('psnr: ', np.mean(psnr), ' std dev', np.std(psnr))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    # parser.add_argument('-nr', '--nr', default=0, type=int,
    #                    help='ranking within the nodes')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--epochs', default=2, type=int, metavar='e',
                        help='number of total epochs to run')
    parser.add_argument('--batch', default=2, type=int, metavar='b',
                        help='number of batch per gpu')
    parser.add_argument('--retrain', default=0, type=int, metavar='r',
                        help='retrain epochs')
    parser.add_argument('--amp', default="disable", type=str, metavar='m',
                        help='mixed precision')
    parser.add_argument('--out_dir', default=".", type=str, metavar='o',
                        help='default directory to output files')
    parser.add_argument('--num_w', default=1, type=int, metavar='w',
                        help='num of data loader workers')
    parser.add_argument('--new_load', default="disable", type=str, metavar='l',
                        help='new data loader')
    parser.add_argument('--wan', default=-1, type=int, metavar='h',
                        help='enable wandb configuration')

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    # init_env_variable()
    args.nr = int(os.environ['SLURM_PROCID'])
    print("SLURM_PROCID: " + str(args.nr))
    # world_size = 4
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_ADDR'] = '10.21.10.4'
    # os.environ['MASTER_PORT'] = '12355'
    # os.environ['MASTER_PORT'] = '8888'
    mp.spawn(dd_train,
             args=(args,),
             nprocs=args.gpus,
             join=True)


if __name__ == '__main__':
    # def __main__():

    ####################DATA DIRECTORY###################
    # jy
    # global root

    # if not os.path.exists("./loss"):
    #    os.makedirs("./loss")
    # if not os.path.exists("./reconstructed_images/val"):
    #    os.makedirs("./reconstructed_images/val")
    # if not os.path.exists("./reconstructed_images/test"):
    #    os.makedirs("./reconstructed_images/test")
    # if not os.path.exists("./reconstructed_images"):
    #    os.makedirs("./reconstructed_images")

    main();
    exit()
