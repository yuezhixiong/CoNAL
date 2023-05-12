import torch, time, os, json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import CoNAL
from utils import *

from tqdm import tqdm
from nocgrad import NOCLGrad

from dataset import get_loaders

# Hyperparameters etc.
TRAIN_DIR = 'E:/Dataset/nyu'
LOG_DIR = 'logs'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 3
NUM_EPOCHS = 1



def loss_fn(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == 'semantic':
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == 'depth':
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    if task_type == 'normal':
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    return loss

def search_fn(train_loader, val_loader, model, weight_optim, arch_optim, logdir):

    writer = SummaryWriter(log_dir=logdir)

    num_batchs = len(train_loader)
    avg_cost = torch.zeros([num_batchs, 24])
    alpha_arrs = []

    for epoch_idx in range(NUM_EPOCHS):
        cost = torch.zeros(24)
        model.train()
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)

        for batch_idx in tqdm(range(num_batchs)):
            train_data, train_label, train_depth, train_normal = next(train_iter)
            train_data, train_label = train_data.to(DEVICE), train_label.to(DEVICE)
            train_depth, train_normal = train_depth.to(DEVICE), train_normal.to(DEVICE)

            try:
                val_data, trainval_label, trainval_depth, trainval_normal = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_data, trainval_label, trainval_depth, trainval_normal = next(val_iter)
            val_data, trainval_label = val_data.to(DEVICE), trainval_label.to(DEVICE)
            trainval_depth, trainval_normal = trainval_depth.to(DEVICE), trainval_normal.to(DEVICE)

            # phase 1. architecture step
            model.eval()
            arch_optim.zero_grad()
            trainval_pred = model(val_data)
            trainval_loss = [loss_fn(trainval_pred[0], trainval_label, 'semantic'),
                        loss_fn(trainval_pred[1], trainval_depth, 'depth'),
                        loss_fn(trainval_pred[2], trainval_normal, 'normal')]
            loss_vals = torch.zeros(3).cuda()
            for i in range(3):
                loss_vals[i] = trainval_loss[i]
            loss_val = torch.sum(loss_vals)   
            loss_val.backward()
            arch_optim.step()

            # phase 2: network step
            weight_optim.zero_grad()
            model.train()
            train_pred = model(train_data)
            train_loss = [loss_fn(train_pred[0], train_label, 'semantic'),
                        loss_fn(train_pred[1], train_depth, 'depth'),
                        loss_fn(train_pred[2], train_normal, 'normal')]
            loss_trains = torch.zeros(3).cuda()
            for i in range(3):
                loss_trains[i] = train_loss[i]
            loss_train = torch.sum(loss_trains)  
            loss_train.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            weight_optim.step()

            cost[0] = train_loss[0].item()
            cost[3] = train_loss[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            cost[6] = train_loss[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred[2], train_normal)
            avg_cost[epoch_idx, :12] += cost[:12] / num_batchs
            break
        
        # logging loss after each epoch
        writer.add_scalar('train/loss0', avg_cost[epoch_idx, 0], epoch_idx)
        writer.add_scalar('train/loss1', avg_cost[epoch_idx, 3], epoch_idx)
        writer.add_scalar('train/loss2', avg_cost[epoch_idx, 6], epoch_idx)
        # logging metric for each task
        writer.add_scalar('train/mIOU', avg_cost[epoch_idx, 1], epoch_idx)
        writer.add_scalar('train/PixAcc', avg_cost[epoch_idx, 2], epoch_idx)
        writer.add_scalar('train/AbsErr', avg_cost[epoch_idx, 4], epoch_idx)
        writer.add_scalar('train/RelErr', avg_cost[epoch_idx, 5], epoch_idx)
        writer.add_scalar('train/ADMean', avg_cost[epoch_idx, 7], epoch_idx)
        writer.add_scalar('train/ADMedi', avg_cost[epoch_idx, 8], epoch_idx)
        writer.add_scalar('train/With11', avg_cost[epoch_idx, 9], epoch_idx)
        writer.add_scalar('train/With22', avg_cost[epoch_idx, 10], epoch_idx)
        writer.add_scalar('train/With68', avg_cost[epoch_idx, 11], epoch_idx)

    if isinstance(model.alpha, nn.ParameterList):
        alpha_arr = np.concatenate([x.detach().cpu().numpy() for x in model.alpha], -1)
    else:
        alpha_arr = model.alpha.detach().cpu().numpy()
    writer.add_text('alpha', np.array2string(alpha_arr), epoch_idx)
    writer.close()
    alpha_arrs.append(alpha_arr)
    np.save(os.path.join(logdir, 'alpha_arrs.npy'), np.array(alpha_arrs))
    return alpha_arr


def output_arch(alpha_arr, logdir):
    arch_path = os.path.join(logdir, 'arch.json')
    print('dump searched architecture to ', arch_path)
    arch_dict = {}
    for idx, alpha in enumerate(alpha_arr):
        arch_dict[idx] = int(alpha.argmax())
        print('task {} branch at {}'.format(idx, arch_dict[idx]))
    with open(arch_path, "w") as outfile:
        json.dump(arch_dict, outfile)


def main():
    # prepare logger
    logdir = create_logdir(LOG_DIR)

    # prepare dataloaders
    train_loader,  val_loader = get_loaders(TRAIN_DIR, BATCH_SIZE)

    # prepare model
    model = CoNAL().to(DEVICE)

    # prepare optimizer
    weight_optim = NOCLGrad(optim.Adam([{'params': model.private_parameters(), 'name':'private'},
                            {'params': model.layer0_parameters(), 'name':'layer0'},
                            {'params': model.layer1_parameters(), 'name':'layer1'},
                            {'params': model.layer2_parameters(), 'name':'layer2'},
                            {'params': model.layer3_parameters(), 'name':'layer3'},
                            {'params': model.layer4_parameters(), 'name':'layer4'},
                            ], lr=1e-4, weight_decay=1e-5))
    arch_optim = optim.Adam(model.arch_parameters(), lr=5e-5, weight_decay=1e-5)

    # start searching
    alpha_arr = search_fn(train_loader, val_loader, model, weight_optim, arch_optim, logdir)

    # output architecture
    output_arch(alpha_arr, logdir)

if __name__ == '__main__':
    main()