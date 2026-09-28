import random
import numpy as np
import torch
import os
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.optim as optim
import argparse
from torch import nn
from model import D2CET_mulscale
from model import D2CET_mulscale_plus3n
from data_RGB import get_training_data, get_validation_data
from torch.utils.data import DataLoader
from Losses import FCR, SCR,PFCR, MFCR
from torch.optim.lr_scheduler import MultiStepLR
import utils
from warmup_scheduler import GradualWarmupScheduler
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 强制使用第一块GPU
# if torch.cuda.is_available():
#     print(f"CUDA 版本: {torch.version.cuda}")
def get_args():
    # -------------------------------------------------------------------------------------------------------------------------------------#
    # Training setting for our methods
    parser = argparse.ArgumentParser(description="D2CETNet_train")
    parser.add_argument("--batchSize", type=int, required=False, default=20, help="Training batch size")
    parser.add_argument("--patchSize", type=int, required=False, default=64, help="Training patch size")
    parser.add_argument("--epochs", type=int, required=False, default=800, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=8e-4, help="Initial learning rate")
    # -------------------------------------------------------------------------------------------------------------------------------------#
    # Training setting for GDUNet
    # parser = argparse.ArgumentParser(description="GDUNet_train")
    # parser.add_argument("--batchSize", type=int, required=False, default=4, help="Training batch size")
    # parser.add_argument("--patchSize", type=int, required=False, default=64, help="Training patch size")
    # parser.add_argument("--epochs", type=int, required=False, default=250, help="Number of training epochs")
    # parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    # ------------------------------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--save_weights", type=str, required=False, default="./trained_model",
                        help='path of log files')
    parser.add_argument("--train_data", type=str, required=False, default='E:/Derain_AGUNetcode/rain100H/train', help='path to training data')
    parser.add_argument("--val_data", type=str, required=False, default='E:/Derain_AGUNetcode/rain100H/test', help='path to training data')
    # parser.add_argument("--train_data", type=str, required=False, default='E:/Derain_AGUNetcode/rain1400/train/small', help='path to training data')
    # parser.add_argument("--val_data", type=str, required=False, default='E:/Derain_AGUNetcode/rain1400/test/small',  help='path to training data')

    parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
    parser.add_argument('--decay', type=int, default='25', help='learning rate decay type')
    parser.add_argument("--TRAINING.RESUME", type=bool, default=True, help='use GPU or not')

    return parser.parse_args()
def draw_psnr(PSNR_list, epoch):
    plt.cla()
    x1 = range(1, epoch + 1)
    y1 = PSNR_list
    fig1 = plt.figure(1)
    plt.title('Train PSNR vs. epoches', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Train PSNR', fontsize=20)
    plt.grid()
    plt.savefig("./Train_PSNR.png")
    plt.draw()
    plt.pause(4)  # 间隔的秒数： 4s
    plt.close(fig1)
if __name__ == '__main__':
    opt = get_args()

    #============# 一、Set Seeds 设置种子，结果和模型的保存路径###########==================================#
    start_time = time.time()

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    #============#    # 二、 loading datasets    =========================================================#
    train_dataset = get_training_data(opt.train_data, {'patch_size': opt.patchSize})
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=4,
                              drop_last=False, pin_memory=True)

    val_dataset = get_validation_data(opt.val_data, {'patch_size': opt.patchSize})
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=4,
                            drop_last=False, pin_memory=True)

    print('===> Loading datasets')
    # ============#   三、loading model  ================================================================#
    # model_restoration = D2CET_mulscale.D2CETNet()
    model_restoration = D2CET_mulscale_plus3n.D2CETNet()
    model_restoration.cuda()
    total_params = sum(p.numel() for p in model_restoration.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model_restoration.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    #===============  # 四 Optimizer   ===================================================================#
    # 优化器二选一，warm或者multisetplR
    optimizer = optim.Adam(model_restoration.parameters(), lr= opt.lr, betas=(0.9, 0.999), eps=1e-8)
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs - warmup_epochs,
                                                            eta_min= 1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)

 ##========================= 五、Loss =========================================================================================

    criterion = nn.L1Loss().cuda()
    SCR = SCR()
    criterion_SCR = SCR.cuda()
    FCR = FCR()
    criterion_FCR = FCR.cuda()
    MFCR = MFCR()
    criterion_MFCR = MFCR.cuda()
    PFCR = PFCR()
    criterion_PFCR = PFCR.cuda()


    best_psnr = 0
    best_ssim = 0
    PSNR_list = [ ]
    Loss_list = [ ]

    for epoch in range(1, opt.epochs + 1):
        epoch_start_time = time.time()

        epoch_loss = 0
        torch.cuda.empty_cache()
        model_restoration.train()

        for i, data in enumerate(tqdm(train_loader), 0 ):
            optimizer.zero_grad()

            target = data[0].cuda()
            input_ = data[1].cuda()

            restored = model_restoration(input_)


            loss_dif= sum([criterion(restored[j], target) for j in range(len(restored))])

            loss_SCR = sum([criterion_SCR(restored[j], target,input_) for j in range(len(restored))])

            loss_FCR = sum([criterion_FCR(restored[j], target, input_) for j in range(len(restored))])
            loss_MFCR = sum([criterion_MFCR(restored[j], target, input_) for j in range(len(restored))])
            loss_PFCR = sum([criterion_PFCR(restored[j], target, input_) for j in range(len(restored))])

            loss = loss_dif +  loss_SCR +  loss_FCR +  loss_MFCR + loss_PFCR

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        Loss_list.append(epoch_loss)

        scheduler.step()

       # Evaluation
        if epoch % 1 == 0:
            model_restoration.eval()
            psnr_val_rgb = []
            for ii, data_val in enumerate(val_loader,0):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()

                with torch.no_grad():
                    restored = model_restoration(input_)

                restored = restored[0]

                for res, tar in zip(restored, target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))

                restored = torch.clamp(restored,0.,1.)

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

            PSNR_list.append(psnr_val_rgb)

            # draw_psnr(PSNR_list,epoch)

            if psnr_val_rgb > best_psnr:
               best_psnr = psnr_val_rgb
               best_epoch = epoch
               print("------save best epoch!!!-----")
               torch.save({'epoch': epoch,
                           'state_dict': model_restoration.state_dict(),
                           'optimizer': optimizer.state_dict()
                           }, os.path.join(opt.save_weights,"model_best.pth"))

            print("=========epoch:%d"%epoch,"========epoch PSNR:%f"%psnr_val_rgb,"=========Best_epoch:%d"%best_epoch,"========best PSNR:%f"%best_psnr)

            torch.save({'epoch': epoch,
                             'state_dict': model_restoration.state_dict(),
                             'optimizer': optimizer.state_dict()
                             }, os.path.join(opt.save_weights,f"model_epoch_{epoch}.pth"))
        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time() - epoch_start_time,
                                                                                  epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(opt.save_weights, "model_latest.pth"))

    print("Time: {:.4f}".format(time.time() - start_time))



