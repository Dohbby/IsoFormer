###################---------Import---------###################
import os 
import argparse
import torch
import testtime
import datetime
import random
import utils
import torch.nn as nn
import torch.optim as optim
import numpy as np
import skimage.color as sc

from collections import OrderedDict
from importlib import import_module
from tqdm import tqdm
from torchsummary import summary
from ptflops import get_model_complexity_info

from model import edsr
#from model_wv_abhi_gau import hat
# from Model_MFSR import MFSR
#from model_wacv import esrt
from data import DIV2K_train, DIV2K_valid, Set5_val
from torch.utils.data import DataLoader
from skimage import metrics
from scipy.io import loadmat, savemat

import  time
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.backends.cudnn.benchmark = True


###################---------Arguments---------###################
# Training settings
parser = argparse.ArgumentParser(description="CFAT")
parser.add_argument('--train-file', type=str, default="/root/dataset-3090/data-3D/train3d-32-2x-bubian.h5")  # 训练 h5文件目录
# parser.add_argument('--train-file', type=str, default="/root/dataset-3090/dataset3090/train-h5/test-64.h5")  # 训练 h5文件目录
parser.add_argument("--batch_size", type=int, default=4, help="training batch size")
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
parser.add_argument("--patch_size", type=int, default=256, help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1, help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=1, help="number of color channels to use")
parser.add_argument("--in_channels", type=int, default=72, help="number of channels for transformer")
parser.add_argument("--n_layers", type=int, default=3, help="number of FETB uits to use")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained models")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--isY", action="store_true", default=True)
parser.add_argument("--ext", type=str, default='.png')
parser.add_argument("--phase", type=str, default='train')
parser.add_argument("--model", type=str, default='ESRT')
parser.add_argument("--channel_in_DE", type=int, default=3)
parser.add_argument("--channel_base_DE", type=int, default=8)
parser.add_argument("--channel_int", type=int, default=16)
parser.add_argument("--output_dir", type=str, default="/root/VS-CODE/3D-MRI/suibian/new/DATx2/")
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--log_period", type=int, default=10)
parser.add_argument("--checkpoint_period", type=int, default=1)
parser.add_argument("--eval_period", type=int, default=2)
parser.add_argument("--validtext", type=str, default="/root/VS-CODE/3D-MRI/suibian/new/DATx2/x2-model.txt", help='text file to store validation results')
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
    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.enabled = False
else:
    seed_val = random.randint(1, 10000)
    print("Ramdom Seed: ", seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)
    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.enabled = False


###################---------Weights_Initiazation---------###################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    if m.bias is not None:
        m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    if m.bias is not None:
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('PixelShuffle') != -1:
        m.weight.data.normal_(0.0, 0.02)
    if m.bias is not None:
        m.bias.data.fill_(0)



###################---------Environment---------###################
cuda = args.cuda
device = torch.device('cuda:0' if cuda else 'cpu')
gpus=[0,1]    #[0, 1, 2, 4] for batch size of 12
def ngpu(gpus):
    """count how many gpus used"""
    gpus = [gpus] if isinstance(gpus, int) else list(gpus)
    return len(gpus)



###################---------Dataset---------###################
print("===> Loading datasets")


from dataset import H5Dataset
train_set=H5Dataset(args.train_file)
training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads * ngpu(gpus), batch_size=args.batch_size, shuffle=True, drop_last=True)







###################---------Model---------###################
print("===> Building models")
torch.cuda.empty_cache()
args.is_train = True

##Model::ESRT
#args.is_train = True
# from model import  swimir3d
# model=swimir3d.SwinIR3D()

# from model import SAFMN
# model=SAFMN.SAFMN()
#
# from model import see3d
# model=see3d.SeemoRe3D()
# from model import srformer3d
# model=srformer3d.CATANet3D()
#
# from model import bsrn
# model=bsrn.BSRN3D()

# from model import catanet3d
# model=catanet3d.CATANet3D()
#
# from model import srformer3d
# model=srformer3d.SRFormer3D()


# from model import ours
# model=ours.MRIGraphSuperResolution()

#
from model3D import dat
model=dat.DAT3D()
###################---------Loss_Function---------###################
l1_criterion = nn.L1Loss()


###################---------Optimizer---------###################
print("===> Setting Optimizer")
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.base_lr, eta_min=0.01 * args.base_lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)
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
begin_epoch=1
train_loss={}
checkpoint_file=os.path.join(args.output_dir, 'checkpoint.pth')
if os.path.exists(checkpoint_file):
      checkpoint=torch.load(checkpoint_file)
      begin_epoch = checkpoint['epoch']
      train_loss=checkpoint['train_loss']
      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      scheduler.load_state_dict(checkpoint['scheduler'])
      print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))



###################---------Define_Training_Epoch---------###################
def train(epoch):

    #utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    accum_iter=4          # Gradient accumulation

    ##Training_Iteration_Start
    start_time = time.time()
    model.train()
    loader = tqdm(training_data_loader)
    with tqdm(training_data_loader, unit="batch") as tepoch:
        for iteration, batch in enumerate(tepoch, 1):

            lr,  gt= batch[0], batch[1]
            tepoch.set_description(f"Epoch {epoch}")
            if args.cuda:
                lr_tensor = lr.to(device)  # ranges from [0, 1]
                hr_tensor = gt.to(device)  # ranges from [0, 1]
            
            sr_tensor = model(lr_tensor)
            # print(sr_tensor.shape,"HR::",hr_tensor.shape)
            loss_l1 = l1_criterion(sr_tensor, hr_tensor)
            loss_sr = loss_l1
            optimizer.zero_grad()
            loss_sr.backward()
            optimizer.step()
            loss_meter.update(loss_sr.item(), lr_tensor.shape[0])
            tepoch.set_postfix(loss=loss_sr.item())
            torch.cuda.synchronize()

            pred1 = sr_tensor.cpu().detach().numpy()
            gt1 = gt.cpu().detach().numpy()
            psnr = metrics.peak_signal_noise_ratio(gt1, pred1, data_range=1)
            tepoch.set_postfix(psnr=psnr.item())

            txt_write = open(args.validtext, 'a')
            print("Epoch: {} Iteration[{}/{}]  Learning rate: {:.5f}  Loss: {:.4f} PSNR:{}".format(epoch, iteration,len(training_data_loader),scheduler.optimizer.param_groups[0]['lr'],loss_meter.avg * accum_iter,
                                                                                                   psnr),file=txt_write)

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr))
        #For_Print
        if (iteration) % args.log_period == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.4f}".format(epoch, iteration, len(training_data_loader),loss_meter.avg))
    
    end_time = time.time()
    time_per_batch = (end_time - start_time) / (iteration)
    print("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s] Avg_Loss: {:.4f}".format(epoch, time_per_batch, training_data_loader.batch_size / time_per_batch, loss_meter.avg*accum_iter))
    ##Iteration_End
    
    # txt_write = open(args.validtext, 'a')
    # print("Epoch: {}  Learning rate: {:.5f}  Loss: {:.4f}".format(epoch, scheduler.optimizer.param_groups[0]['lr'], loss_meter.avg*accum_iter), file = txt_write)
    #txt_write.close()

    #Save_Loss
    train_loss[epoch]=loss_meter.avg*accum_iter
    
    #Save_Check-points
    states={
            'train_loss':train_loss,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler':scheduler.state_dict()
            }   
            
    torch.save(states, os.path.join(args.output_dir, 'checkpoint.pth')) 
    if epoch % args.checkpoint_period == 0 or epoch == args.nEpochs:
        torch.save(states, os.path.join(args.output_dir, 'checkpoint'+'_{}.pth'.format(epoch)))




###################---------save_checkpoint---------###################
def save_checkpoint(epoch):
    model_folder = "experiment/checkpoint_ESRT_x{}/".format(args.scale)
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))




###################---------Tot_Parameter_Count---------###################
# def print_network(model):
#     summary(model, (1,23,23,23))
#     macs, params = get_model_complexity_info(model, (1,23, 23,23), as_strings=True, print_per_layer_stat=True, verbose=True)
#     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def print_network(model):
    summary(model, (1,32, 32, 32))
    macs, params = get_model_complexity_info(model, (1, 32,32,32), as_strings=True, print_per_layer_stat=True, verbose=True)
    if 'GMac' in macs:
        gmacs = float(macs.replace(' GMac', ''))
        flops = 2 * gmacs * 1e9  # 转换为 FLOPs
        flops_str = f"{flops / 1e9:.2f} GFLOPs"
    elif 'MMac' in macs:
        mmacs = float(macs.replace(' MMac', ''))
        flops = 2 * mmacs * 1e6  # 转换为 FLOPs
        flops_str = f"{flops / 1e6:.2f} MFLOPs"
    else:
        flops_str = "N/A"

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print('{:<30}  {:<8}'.format('Computational complexity (FLOPs):', flops_str))
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


print("===> Training")

##Computational_Complexity[Parameters_&_Flops]
print_network(model)

##Start_Total_Compuational_Time
code_start_time = datetime.datetime.now()
loss_meter = AverageMeter()   
timer = utils.Timer()

optimizer.zero_grad() #We are using gradient accumulation
for epoch in range(begin_epoch, args.nEpochs + 1):
    t_epoch_start = timer.t()
    loss_meter.reset()        

    ##Training_Start
    train_start_time = datetime.datetime.now()
    train(epoch)

    train_end_time = datetime.datetime.now()
    print('Epoch cost times: %s' % str(train_end_time-train_start_time))
    ##Training_End
    
    t = timer.t()
    prog = (epoch-args.start_epoch+1)/(args.nEpochs + 1 - args.start_epoch + 1)
    t_epoch = utils.time_text(t - t_epoch_start)
    t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
    print('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
    
code_end_time = datetime.datetime.now()
print('Code cost times: %s' % str(code_end_time-code_start_time))

txt_write = open(args.validtext, 'a')
print('Code cost times: %s' % str(code_end_time-code_start_time))
txt_write.close()
##End_Total_Compuational_Time