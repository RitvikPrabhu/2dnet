import torch.nn.utils.prune as prune
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np
import parser_util as prs
import os
from os import path
# from PIL import Image

# from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import torch.cuda.amp as amp
# from apex.contrib.sparsity import ASP
from socket import gethostname
# from dataload import CTDataset
# from dataload_optimization import CTDataset



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

        conv_1 = self.batch_norm1_0(x)
        conv_1 = self.conv1_0(conv_1)
        conv_1 = F.leaky_relu(conv_1)
        conv_2 = self.batch_norm2_0(conv_1)
        conv_2 = self.conv2_0(conv_2)
        conv_2 = F.leaky_relu(conv_2)

        x = torch.cat((x, conv_2), dim=1)
        conv_1 = self.batch_norm1_1(x)
        conv_1 = self.conv1_1(conv_1)
        conv_1 = F.leaky_relu(conv_1)
        conv_2 = self.batch_norm2_1(conv_1)
        conv_2 = self.conv2_1(conv_2)
        conv_2 = F.leaky_relu(conv_2)

        x = torch.cat((x, conv_2), dim=1)
        conv_1 = self.batch_norm1_2(x)
        conv_1 = self.conv1_2(conv_1)
        conv_1 = F.leaky_relu(conv_1)
        conv_2 = self.batch_norm2_2(conv_1)
        conv_2 = self.conv2_2(conv_2)
        conv_2 = F.leaky_relu(conv_2)

        x = torch.cat((x, conv_2), dim=1)
        conv_1 = self.batch_norm1_3(x)
        conv_1 = self.conv1_3(conv_1)
        conv_1 = F.leaky_relu(conv_1)
        conv_2 = self.batch_norm2_3(conv_1)
        conv_2 = self.conv2_3(conv_2)
        conv_2 = F.leaky_relu(conv_2)
        x = torch.cat((x, conv_2), dim=1)

        return x

INPUT_CHANNEL_SIZE = 1

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


    # @jit.script_method
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

def read_correct_image(path):
    offset = 0
    ct_org = None
    with Image.open(path) as img:
        ct_org = np.float32(np.array(img))
        if 270 in img.tag.keys():
            for item in img.tag[270][0].split("\n"):
                if "c0=" in item:
                    loi = item.strip()
                    offset = re.findall(r"[-+]?\d*\.\d+|\d+", loi)
                    offset = (float(offset[1]))
    ct_org = ct_org + offset
    neg_val_index = ct_org < (-1024)
    ct_org[neg_val_index] = -1024
    return ct_org
class CTDataset(Dataset):
    def __init__(self, root_dir_h, root_dir_l, length, transform=None):
        self.data_root_l = root_dir_l + "/"
        self.data_root_h = root_dir_h + "/"
        self.img_list_l = os.listdir(self.data_root_l)
        self.img_list_h = os.listdir(self.data_root_h)
        self.img_list_l.sort()
        self.img_list_h.sort()
        self.img_list_l = self.img_list_l[0:length]
        self.img_list_h = self.img_list_h[0:length]
        self.transform = transform

    def __len__(self):
        return len(self.img_list_l)

    def __getitem__(self, idx):
        # print("Dataloader idx: ", idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        inputs_np = torch.randn((1,512,512))
        targets_np = torch.randn((1,512,512))

        # inputs = torch.from_numpy(inputs_np)
        # targets = torch.from_numpy(targets_np)
        inputs = inputs_np.type(torch.FloatTensor)
        targets = targets_np.type(torch.FloatTensor)

        sample = {'vol': 'qqq',
                  'HQ': targets,
                  'LQ': inputs,
                  'max': 1,
                  'min': 0}
        return sample

root_train_h = "/projects/synergy_lab/garvit217/enhancement_data/train/HQ/"
root_train_l = "/projects/synergy_lab/garvit217/enhancement_data/train/LQ/"
root_val_h = "/projects/synergy_lab/garvit217/enhancement_data/val/HQ/"
root_val_l = "/projects/synergy_lab/garvit217/enhancement_data/val/LQ/"
root_test_h = "/projects/synergy_lab/garvit217/enhancement_data/test/HQ/"
root_test_l = "/projects/synergy_lab/garvit217/enhancement_data/test/LQ/"

trainset = CTDataset(root_dir_h=root_train_h, root_dir_l=root_train_l, length=5120)
testset = CTDataset(root_dir_h=root_test_h, root_dir_l=root_test_l, length=784)
valset = CTDataset(root_dir_h=root_val_h, root_dir_l=root_val_l, length=784)
# trainset = CTDataset(root_dir_h=root_train_h, root_dir_l=root_train_l, length=32)
# testset = CTDataset(root_dir_h=root_val_h, root_dir_l=root_val_l, length=16)
# valset = CTDataset(root_dir_h=root_test_h, root_dir_l=root_test_l, length=16)

train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=1, rank=0)
test_sampler = torch.utils.data.distributed.DistributedSampler(testset, num_replicas=1, rank=0)
val_sampler = torch.utils.data.distributed.DistributedSampler(valset, num_replicas=1, rank=0)
# train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

train_loader = DataLoader(trainset, batch_size=4, drop_last=False, shuffle=False, num_workers=1, pin_memory=True,
                          sampler=train_sampler)
test_loader = DataLoader(testset, batch_size=4, drop_last=False, shuffle=False, num_workers=1,
                         pin_memory=True, sampler=test_sampler)
val_loader = DataLoader(valset, batch_size=4, drop_last=False, shuffle=False, num_workers=1, pin_memory=True,
                        sampler=val_sampler)

model = DD_net()
device = torch.device("cpu")
torch.cuda.manual_seed(1111)
# necessary for AMP to work
# torch.cuda.set_device(device)
model.to(device)
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)  #######ADAM CHANGE
decayRate = 0.95
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
amp_enabled = False
scaler = torch.cuda.amp.GradScaler()
for k in range(2):
    for batch_index, batch_samples in enumerate(train_loader):
        file_name, HQ_img, LQ_img, maxs, mins = batch_samples['vol'], batch_samples['HQ'], batch_samples['LQ'], \
            batch_samples['max'], batch_samples['min']
        targets = HQ_img
        inputs = LQ_img
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(enabled=amp_enabled):

            outputs = model(inputs)
            MSE_loss = nn.MSELoss()(outputs, targets)
            MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
            loss = MSE_loss + 0.1 * (MSSSIM_loss)
        if amp_enabled:
            # print('bw pass')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            #                 print('loss_bacl')
            loss.backward()
            #                 print('optimia')
            optimizer.step()
    scheduler.step()