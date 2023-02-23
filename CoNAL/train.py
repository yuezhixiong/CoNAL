import torch, time, os, random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from backbone import CoNAL
from utils import *
import create_dataset

import argparse
from tqdm import tqdm
from nocgrad import NOCLGrad


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default='0', help='gpu_id') 
    parser.add_argument('--model', default='CoNAL', type=str, help='')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--total_index', default=100, type=int, help='training index')
    parser.add_argument('--name', default='', type=str, help='logdir name')
    parser.add_argument('--dataset', default='NYUv2', type=str, help='dataset name')
    return parser.parse_args()


params = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id

batch_size = params.batch_size
dataset = params.dataset
train_split = 0.5
model = CoNAL(dataset=dataset).cuda()

dataset_path = 'path_to_dataset'
logfolder = params.model +  params.name 
logfolder += time.strftime("_%Y_%m%d_%H%M")
logdir = os.path.join('./out', logfolder)
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(log_dir=logdir)

train_set = create_dataset(name=dataset, root=dataset_path, mode=params.train_mode, augmentation=params.aug)

num_train = len(train_set)
indices = list(range(num_train))
split = int(np.floor(train_split * num_train))
train_data = torch.utils.data.Subset(train_set, indices[:split])

trainval_data = torch.utils.data.Subset(train_set, indices[split:num_train])
trainval_loader = torch.utils.data.DataLoader(
    dataset=trainval_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True)

task_num = len(model.tasks)

optimizer = NOCLGrad(optim.Adam([{'params': model.private_parameters(), 'name':'private'},
                            {'params': model.layer0_parameters(), 'name':'layer0'},
                            {'params': model.layer1_parameters(), 'name':'layer1'},
                            {'params': model.layer2_parameters(), 'name':'layer2'},
                            {'params': model.layer3_parameters(), 'name':'layer3'},
                            {'params': model.layer4_parameters(), 'name':'layer4'},
                            ], lr=1e-4, weight_decay=1e-5))
arch_optim = optim.Adam(model.arch_parameters(), lr=5e-5, weight_decay=1e-5)

total_index = params.total_index
train_batch = len(train_loader)
avg_cost = torch.zeros([total_index, 24])
alpha_arrs = []

for index in range(total_index):
    s_t = time.time()
    cost = torch.zeros(24)
    model.train()
    train_dataset = iter(train_loader)
    conf_mat = ConfMatrix(model.class_nb)

    # search   
    if params.model == 'CoNAL':
        trainval_dataset = iter(trainval_loader)
        for k in tqdm(range(train_batch)):
            train_data, train_label, train_depth, train_normal = train_dataset.next()
            train_data, train_label = train_data.cuda(non_blocking=True), train_label.long().cuda(non_blocking=True)
            train_depth, train_normal = train_depth.cuda(non_blocking=True), train_normal.cuda(non_blocking=True)

            trainval_data, trainval_label, trainval_depth, trainval_normal = trainval_dataset.next()
            trainval_data, trainval_label = trainval_data.cuda(non_blocking=True), trainval_label.long().cuda(non_blocking=True)
            trainval_depth, trainval_normal = trainval_depth.cuda(non_blocking=True), trainval_normal.cuda(non_blocking=True)

            # phase 1. architecture step
            model.eval()
            arch_optim.zero_grad()
            trainval_pred = model(trainval_data)
            trainval_loss = [model_fit(trainval_pred[0], trainval_label, 'semantic'),
                        model_fit(trainval_pred[1], trainval_depth, 'depth'),
                        model_fit(trainval_pred[2], trainval_normal, 'normal')]
            loss_vals = torch.zeros(3).cuda()
            for i in range(3):
                loss_vals[i] = trainval_loss[i]
            loss_val = torch.sum(loss_vals)   
            loss_val.backward()
            arch_optim.step()

            # phase 2: network step
            optimizer.zero_grad()
            model.train()
            train_pred = model(train_data)
            train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                        model_fit(train_pred[1], train_depth, 'depth'),
                        model_fit(train_pred[2], train_normal, 'normal')]
            loss_trains = torch.zeros(3).cuda()
            for i in range(3):
                loss_trains[i] = train_loss[i]
            loss_train = torch.sum(loss_trains)  
            loss_train.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = train_loss[0].item()
            cost[3] = train_loss[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            cost[6] = train_loss[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred[2], train_normal)
            avg_cost[index, :12] += cost[:12] / train_batch
    else: # retrain
        for k in tqdm(range(train_batch)):
            train_data, train_label, train_depth, train_normal = train_dataset.next()
            train_data, train_label = train_data.cuda(non_blocking=True), train_label.long().cuda(non_blocking=True)
            train_depth, train_normal = train_depth.cuda(non_blocking=True), train_normal.cuda(non_blocking=True)

            train_pred = model(train_data)

            train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                        model_fit(train_pred[1], train_depth, 'depth'),
                        model_fit(train_pred[2], train_normal, 'normal')]
            
            loss_train = torch.zeros(3).cuda()
            
            for i in range(3):
                loss_train[i] = train_loss[i]

            optimizer.zero_grad()

            batch_weight = F.softmax(torch.rand(task_num).cuda(), dim=-1)
            loss_train = loss_train * batch_weight
            optimizer.step()

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = train_loss[0].item()
            cost[3] = train_loss[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            cost[6] = train_loss[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred[2], train_normal)
            avg_cost[index, :12] += cost[:12] / train_batch

    e_t = time.time()

    writer.add_scalar('train/mIOU', avg_cost[index, 1], index)
    writer.add_scalar('train/PixAcc', avg_cost[index, 2], index)
    writer.add_scalar('train/AbsErr', avg_cost[index, 4], index)
    writer.add_scalar('train/RelErr', avg_cost[index, 5], index)
    writer.add_scalar('train/ADMean', avg_cost[index, 7], index)
    writer.add_scalar('train/ADMedi', avg_cost[index, 8], index)
    writer.add_scalar('train/With11', avg_cost[index, 9], index)
    writer.add_scalar('train/With22', avg_cost[index, 10], index)
    writer.add_scalar('train/With68', avg_cost[index, 11], index)

    if isinstance(model.alpha, nn.ParameterList):
        alpha_arr = np.concatenate([x.detach().cpu().numpy() for x in model.alpha], -1)
    else:
        alpha_arr = model.alpha.detach().cpu().numpy()
    writer.add_text('alpha', np.array2string(alpha_arr), epoch)
    alpha_arrs.append(alpha_arr)
    np.save(os.path.join(logdir, 'alpha_arrs.npy'), np.array(alpha_arrs))

torch.save(model.state_dict(), os.path.join(logdir,'model.pth.tar'))