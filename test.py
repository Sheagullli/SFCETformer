import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import utils

from data_RGB import get_test_data
from model import D2CET_mulscale
from model import D2CET_mulscale_plus3n

from layers import *
from thop import profile
from skimage import img_as_ubyte
import time
parser = argparse.ArgumentParser(description='Image Deraining using D2CETNet')
#-------------------------------------------------------------------------------------------------------------#
# for rain 100H with 1800 images
parser.add_argument('--input_dir', default='E:/Derain_AGUNetcode/rain100H/test/input', type=str, help='Directory of validation images')

#-------------------------------------------------------------------------------------------------------------#
# for rain12
# parser.add_argument('--input_dir', default='E:/Derain_AGUNetcode/rain12/input', type=str, help='Directory of validation images')

#------------------------------------------------------------------------------------------------------------#
# for rain100L
# parser.add_argument('--input_dir', default='E:/Derain_AGUNetcode/rain100L/test/input', type=str, help='Directory of validation images')

#------------------------------------------------------------------------------------------------------------#
# for rain14000
# parser.add_argument('--input_dir', default='E:/Derain_AGUNetcode/rain1400/test/small/input', type=str, help='Directory of validation images')

#------------------------------------------------------------------------------------------------------------#
# for real-word dataset
# parser.add_argument('--input_dir', default='E:/Derain_AGUNetcode/dataset/real', type=str, help='Directory of validation images')

#------------------------------------------------------------------------------------------------------------#
# for spa_data test images
# parser.add_argument('--input_dir', default='./dataset/spa-data/test/small/rain', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='E:/2014-Publication/2026 AttentionNet deraining/myarticle/results', type=str, help='Directory for test_results')

#===========================================================================================================#
parser.add_argument('--weights', default='E:/2014-Publication/2026 AttentionNet deraining/myarticle/paraRain100H_new/model_best_plus3n.pth', type=str, help='Path to weights')

# parser.add_argument('--weights', default='E:/2014-Publication/2026 AttentionNet deraining/myarticle/netparameter/dim48/model_best_dim482.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# model_restoration = D2CET_mulscale.D2CETNet()

model_restoration = D2CET_mulscale_plus3n.D2CETNet()

model_restoration.cuda()
utils.load_checkpoint(model_restoration, args.weights)

model_restoration.eval()
win = 256 # window size
# datasets = ['Rain100L', 'Rain100H', 'Test100', 'Test1200', 'Test2800','spadata','practical']
datasets = ['Rain100H']

for dataset in datasets:
    rgb_dir_test = os.path.join(args.input_dir)#, dataset, 'input')
    test_dataset = get_test_data(rgb_dir_test, img_options={})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True)

    result_dir = os.path.join(args.result_dir, dataset)
    utils.mkdir(result_dir)

    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()  # Force在CUDA IPC释放GPU内存后收集GPU内存。
            torch.cuda.empty_cache()  # 释放缓存分配器当前持有的所有未占用的缓存内存

            input_ = data_test[0].cuda()
            filenames = data_test[1]
            start_time = time.time()
            _, _, Hx, Wx = input_.shape
            input_re, batch_list = window_partitionx(input_, win)
            restored = model_restoration(input_re)
            restored = window_reversex(restored[0], win, Hx, Wx, batch_list)

            restored = torch.clamp(restored, 0, 1)
            end_time = time.time()
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            dur_time = end_time - start_time
            print('Avg. time:', dur_time)
            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                utils.save_img((os.path.join(result_dir, filenames[batch] + '.png')), restored_img)
                # utils.save_img((os.path.join(result_dir, filenames[batch] + '_mask.png')), rainmask)

# flops, _ = profile(model_restoration, (input_re,))
# print('FLOPs: ', flops)
