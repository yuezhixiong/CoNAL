import torch, os, argparse, shutil
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import CoNALArch
from utils import *

from tqdm import tqdm
from dataset import get_loaders


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_fn(train_loader, model, weight_optim, num_epochs, logdir):

    writer = SummaryWriter(log_dir=logdir)
    scaler = torch.cuda.amp.GradScaler()

    num_batchs = len(train_loader)
    avg_cost = torch.zeros([num_epochs, 24])

    for epoch_idx in tqdm(range(num_epochs)):
        cost = torch.zeros(24)
        model.train()
        train_iter = iter(train_loader)

        for batch_idx in range(num_batchs):
            train_data, train_label, train_depth, train_normal = next(train_iter)
            train_data, train_label = train_data.to(DEVICE), train_label.to(DEVICE)
            train_depth, train_normal = train_depth.to(DEVICE), train_normal.to(DEVICE)

            with torch.cuda.amp.autocast():  # FP16
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
                scaler.scale(loss_train).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                scaler.step(weight_optim)
                scaler.update()

                cost[0] = train_loss[0].item()
                cost[3] = train_loss[1].item()
                cost[4], cost[5] = depth_error(train_pred[1], train_depth)
                cost[6] = train_loss[2].item()
                cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred[2], train_normal)
                avg_cost[epoch_idx, :12] += cost[:12] / num_batchs
        

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


def main(args):

    # load model config
    yaml_path = 'models/{}.yml'.format(args.model)
    config = yaml_load(yaml_path)
    batch_size = config['hyper']['batch_size']
    num_epochs = config['hyper']['num_epochs']
    lr = config['hyper']['lr']
    weight_decay = config['hyper']['weight_decay']

    # prepare dataloaders
    train_loader = get_loaders(args.datadir, batch_size, stage='retrain')

    # get arch from json file
    arch_name = config['arch']['name']
    arch = config['arch']['branch_points']

    # prepare logger
    logdir = create_logdir(args.logdir, arch_name)
    shutil.copy(yaml_path, logdir)

    # prepare model
    model = CoNALArch(arch).to(DEVICE)

    # prepare optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # start training
    train_fn(train_loader, model, optimizer, num_epochs, logdir)

    # save model
    torch.save(model.state_dict(), os.path.join(logdir, '{}_e{}.pth'.format(arch_name, num_epochs)))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("train")
    parser.add_argument('--model', type=str, default='hps', help='model name')
    parser.add_argument("--logdir", type=str, default='logs')
    parser.add_argument("--datadir", type=str, default='E:/Dataset/nyu')
    args = parser.parse_args()
    
    main(args)