###################---------Import---------###################
import os
import argparse
import torch
import testtime
import datetime
import random

import torch.nn as nn
import torch.optim as optim
import numpy as np

import torch.optim as optim

from skimage import metrics
import lpips  # 导入LPIPS库
import scipy.io
# from Model import MFSR

torch.backends.cudnn.benchmark = True

###################---------Arguments---------###################
# Training settings
parser = argparse.ArgumentParser(description="ESRT")
parser.add_argument("--batch_size", type=int, default=16, help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=1, help="testing batch size")
parser.add_argument("-nEpochs", type=int, default=50, help="number of epochs to train")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=[225, 350, 400, 450], help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=int, default=0.5, help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=True, help="use cuda")
parser.add_argument("--resume", default="", type=str, help="path to checkpoint")
parser.add_argument("--start-epoch", default=1, type=int, help="manual epoch number")
parser.add_argument("--threads", type=int, default=8, help="number of threads for data loading")
parser.add_argument("--root", type=str, default="./Datasets/DIV2K/", help='dataset directory')
parser.add_argument("--n_train", type=int, default=800, help="number of training set")
parser.add_argument("--n_val", type=int, default=5, help="number of validation set")
parser.add_argument("--test_every", type=int, default=1000)
parser.add_argument("--scale", type=int, default=1, help="super-resolution scale")
# parser.add_argument("--patch_size", type=int, default=256, help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1, help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=1, help="number of color channels to use")
parser.add_argument("--in_channels", type=int, default=72, help="number of channels for transformer")
parser.add_argument("--n_layers", type=int, default=3, help="number of FETB uits to use")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained models")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--isY", action="store_true", default=True)
# parser.add_argument("--ext", type=str, default='.png')
parser.add_argument("--phase", type=str, default='train')
parser.add_argument("--model", type=str, default='ESRT')
parser.add_argument("--channel_in_DE", type=int, default=3)
parser.add_argument("--channel_base_DE", type=int, default=8)
parser.add_argument("--channel_int", type=int, default=16)
# parser.add_argument("--output_dir", type=str, default="Put the adress of checkpoint directory here")
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--log_period", type=int, default=10)
parser.add_argument("--checkpoint_period", type=int, default=1)
# parser.add_argument("--eval_period", type=int, default=2)
# parser.add_argument('--folder_lq', type=str, default="./TestData_LR/", help='input low-quality test image folder')
# parser.add_argument('--folder_gt', type=str, default="./TestData_GT/", help='input ground-truth test image folder')
# parser.add_argument("--output_folder", type=str, default="./TestData_OUT/")
parser.add_argument('--task', type=str, default='classical_sr', help='classical_sr, lightweight_sr, real_sr')
# parser.add_argument('--save_img_only', default=False, action='store_true', help='save image and do not evaluate')
parser.add_argument('--tile', type=int, default=32,
                    help='Tile size, None for no tile during testing (testing as a whole)')

parser.add_argument('--tile_overlap', type=int, default=16, help='Overlapping of different tiles')
parser.add_argument("--psnr1text", type=str, default="/root/VS-CODE/3D-MRI/suibian/mixnum/PSNR-each.txt", help='text file to store validation results')
parser.add_argument("--psnr2text", type=str, default="/root/VS-CODE/3D-MRI/suibian/mixnum/PSNR.txt", help='text file to store validation results')
args = parser.parse_args()

print(args)


###################---------Random_Seed---------###################
if args.seed:
    seed_val = 1
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
else:
    seed_val = random.randint(1, 10000)
    print("Ramdom Seed: ", seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False

###################---------Environment---------###################
cuda = args.cuda
device = torch.device('cuda:0' if cuda else 'cpu')
gpus=[0,1]    #[0, 1, 2, 4] for batch size of 12
def ngpu(gpus):
    """count how many gpus used"""
    gpus = [gpus] if isinstance(gpus, int) else list(gpus)
    return len(gpus)


###################---------Model---------###################
print("===> Building models")
torch.cuda.empty_cache()
args.is_train = True
#
# from model3d import srcnn3d
# model=srcnn3d.SRCNN3D()
#
# from model import ESRGAN
# model=ESRGAN.ESRGAN3D()
from model import suibian
model=suibian.ART3D()
# from model import ours
# model=ours.DAT3D()
# from model import swimir3d
# model=swimir3d.SwinIR3D()

# from model import ours
# model=ours.MRIGraphSuperResolution()
###################---------Loss_Function---------###################
l1_criterion = nn.L1Loss()
# loss_fn_alex = lpips.LPIPS(net='alex').to(device)  # 初始化LPIPS模型
###################---------Optimizer---------###################
print("===> Setting Optimizer")
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.step_size, gamma=args.gamma)

###################---------.to(Device)---------###################
print("===> Setting GPU")
if cuda:
    #print(device, gpus)
    #input()
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=gpus)
    l1_criterion = l1_criterion.to(device)
###################---------Load_model_to_resume_training---------###################
begin_epoch = 1
train_loss = {}

# if not os.path.exists(save_dir):
#    os.makedirs(save_dir)

border = args.scale



def test_tile(volume_lq, model, args, window_size):
    """3D分块处理函数
    Args:
        volume_lq (Tensor): 输入低质量体积 [B,C,D,H,W]
        model (nn.Module): 3D超分模型
        args: 包含tile(分块大小)和tile_overlap(重叠区域)参数
        window_size: 模型要求的窗口大小(如SwinIR需要8的倍数)
    Returns:
        Tensor: 超分后的3D体积 [B,C,D*scale,H*scale,W*scale]
    """
    if args.tile is None:
        # 整个体积一次性处理
        return model(volume_lq)

    # 分块处理逻辑
    b, c, d, h, w = volume_lq.shape
    tile = min(args.tile, d, h, w)
    assert tile % window_size == 0, f"tile size {tile} must be multiple of window_size {window_size}"
    tile_overlap = args.tile_overlap
    sf = args.scale

    # 计算分块步长(考虑重叠)
    stride = tile - tile_overlap

    # 生成各维度的分块索引(确保覆盖整个体积)
    d_idx_list = list(range(0, d - tile, stride)) + [d - tile]
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]

    # 初始化累加器和权重矩阵
    E = torch.zeros(b, c, d * sf, h * sf, w * sf).type_as(volume_lq)
    W = torch.zeros_like(E)

    # 分块处理
    for d_idx in d_idx_list:
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                # 提取当前块 [B,C,tile,tile,tile]
                in_patch = volume_lq[...,
                           d_idx:d_idx + tile,
                           h_idx:h_idx + tile,
                           w_idx:w_idx + tile]

                # 模型推理
                with torch.no_grad():
                    out_patch = model(in_patch)

                # 生成权重掩码(边缘区域可考虑高斯加权)
                out_patch_mask = torch.ones_like(out_patch)

                # 累加到对应位置
                E[...,
                d_idx * sf:(d_idx + tile) * sf,
                h_idx * sf:(h_idx + tile) * sf,
                w_idx * sf:(w_idx + tile) * sf].add_(out_patch)

                W[...,
                d_idx * sf:(d_idx + tile) * sf,
                h_idx * sf:(h_idx + tile) * sf,
                w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)

    # 加权平均消除边缘效应
    output = E.div_(W.clamp(min=1e-8))  # 避免除以零

    return output



print("param1:", (sum(param.numel() for param in model.parameters()) / (10 ** 6)))
loss_fn_alex = lpips.LPIPS(net='alex').to(device)
for epoch in range(48,51):
    checkpoint_file =  r'/root/VS-CODE/3D-MRI/suibian/ARTX2/checkpoint_' + str(epoch) + '.pth'

    print(checkpoint_file)

    # checkpoint=torch.load(checkpoint_file)
    # model.load_state_dict(checkpoint['state_dict'])
    if os.path.exists(checkpoint_file):
        #   print('')
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        print('loading state dict')
        model.load_state_dict(checkpoint['state_dict'])
        print('loaded state dict')
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))
    model.eval()
    with (torch.no_grad()):
        PSNR=[]
        SSIM=[]
        LPIPS=[]
        NRMSE=[]

        for mat_num in range(1,6):
            # path = r"/root/dataset-3090/data3d-test/k21-3d-bubian-2x-2-" + str(mat_num) + '.mat' # x2-k21
            # path = r"/root/dataset-3090/data3d-test/T1-3d-bubian-3x-3-" + str(mat_num) + '.mat' # x2-k21
            path = r"/root/dataset-3090/data3d-test/Brates2019-3d-bubian-2x-2-"+ str(mat_num) + '.mat' # x2-k21
            print(path)

            data = scipy.io.loadmat(path)
            lr=data['imgsdown']
            gt=data['imgs']

            # lr = data['cubic']
            # gt = data['img']
            lr=np.expand_dims(lr,0)
            gt=np.expand_dims(gt,0)
            lr = np.expand_dims(lr, 0)
            gt = np.expand_dims(gt, 0)
            gt = torch.from_numpy(gt).float().to(device)
            lr = torch.from_numpy(lr).float().to(device)
            # output = model(lr)
            output = test_tile(lr,model,args,4)

            pred = np.squeeze(output)
            gt = np.squeeze(gt)
            a = pred.cpu().detach().numpy() #170,256,256
            gt = gt.cpu().detach().numpy() #170,256,256

            psnr = metrics.peak_signal_noise_ratio(gt, a, data_range=1)
            ssim = metrics.structural_similarity(gt, a,data_range=1)
            nrmse= metrics.normalized_root_mse(gt, a)
            psnr = float(f"{psnr:.4f}")
            ssim = float(f"{ssim:.4f}")
            nrmse = float(f"{nrmse:.4f}")
            # print("Epoch:{}  第{}个头   PSNR:{} SSIM:{} NEMSE:{}".format(epoch, mat_num, psnr,ssim,nrmse))
            PSNR.append(psnr)
            SSIM.append(ssim)
            NRMSE.append(nrmse)

            lpips_values = []
            for d in range(gt.shape[2]):  # 遍历 depth 维度
                gt_slice = gt[:, :, d]  # (H, W)
                pred_slice = a[:, :, d]  # (H, W)
                # 转换为 PyTorch tensor 并添加 batch 和 channel 维度
                gt_tensor = torch.from_numpy(gt_slice).unsqueeze(0).unsqueeze(0).float().to(device)
                pred_tensor = torch.from_numpy(pred_slice).unsqueeze(0).unsqueeze(0).float().to(device)
                # 归一化到 [-1, 1]
                gt_tensor = gt_tensor * 2 - 1
                pred_tensor = pred_tensor * 2 - 1
                # 计算 LPIPS
                lpips_value = loss_fn_alex (pred_tensor, gt_tensor).item()
                lpips_values.append(lpips_value)

            lpip= np.mean(lpips_values)
            lpip= float(f"{lpip:.4f}")
            print("Epoch:{}  第{}个头   PSNR:{} SSIM:{} NEMSE:{} LIPIS:{}".format(epoch, mat_num, psnr,ssim,nrmse,lpip))
            LPIPS.append(lpip)

            from scipy.io import savemat
            new = {'data': a}
            savemat('/root/VS-CODE/3D-MRI/suibian/esrgantestx4/Brates-pred{}-{}.mat'.format(epoch, mat_num), new)
            # savemat('/root/VS-CODE/3D-MRI/suibian/esrgantestx4/T1-pred{}-{}.mat'.format(epoch, mat_num), new)
            # savemat('/root/VS-CODE/3D-MRI/suibian/mixnum/ART-pred{}-{}.mat'.format(epoch, mat_num), new)
            print('pred{}-{}.mat'.format(epoch, mat_num))

            txt_write = open(args.psnr1text, 'a')
            print("Epoch:{}  第{}个头  PSNR:{} SSIM:{} NRMSE:{}  LPIPIS:{}".format(epoch, mat_num, psnr,ssim,nrmse,lpip), file=txt_write)
            # print('----------------------------')



        mean_psnr = float(f"{sum(PSNR) / len(PSNR):.2f}")
        mean_ssim = float(f"{sum(SSIM) / len(SSIM):.4f}")
        mean_nrmse =float(f"{sum(NRMSE) / len(NRMSE):.4f}")
        mean_lpips =float(f"{sum(LPIPS) / len(LPIPS):.4f}")
        print("epoch:{} 平均PSNR:{} SSIM:{} NRMSE:{} LPIPS:{}".format(epoch,mean_psnr, mean_ssim, mean_nrmse, mean_lpips ))
        txt_write = open(args.psnr2text, 'a')
        print("Epoch:{} 平均PSNR:{} SSIM:{} NRMSE:{} LPIPS:{}".format(epoch, mean_psnr, mean_ssim, mean_nrmse, mean_lpips), file=txt_write)

print("param2:", (sum(param.numel() for param in model.parameters()) / (10 ** 6)))

print(f'=========================Done=========================')











