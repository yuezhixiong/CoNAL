import torch
from model import CoNALArch
from dataset import get_loaders
from utils import *


def test(model, test_loader):
    conf_mat = ConfMatrix(model.class_nb)
    cost = torch.zeros(24)
    avg_cost = torch.zeros(24)
    model.eval()
    with torch.no_grad():  # operations inside don't track history
        test_iter = iter(test_loader)
        test_batch = len(test_loader)
        for k in range(test_batch):
            test_data, test_label, test_depth, test_normal = next(test_iter)
            test_data, test_label = test_data.cuda(non_blocking=True), test_label.long().cuda(non_blocking=True)
            test_depth, test_normal = test_depth.cuda(non_blocking=True), test_normal.cuda(non_blocking=True)
            test_pred = model.predict(test_data)
            test_loss = [loss_fn(test_pred[0], test_label, 'semantic'),
                         loss_fn(test_pred[1], test_depth, 'depth'),
                         loss_fn(test_pred[2], test_normal, 'normal')]

            cost[12] = test_loss[0].item()
            cost[15] = test_loss[1].item()
            cost[18] = test_loss[2].item()

            conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())
            cost[16], cost[17] = depth_error(test_pred[1], test_depth)
            cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(test_pred[2], test_normal)
            avg_cost[12:] += cost[12:] / test_batch

        # compute mIoU and acc
        avg_cost[13], avg_cost[14] = conf_mat.get_metrics()

    task_metric = {}
    task_metric["0"] = avg_cost[13:15]
    task_metric["1"] = avg_cost[16:18]
    task_metric["2"] = avg_cost[19:24]

    return task_metric


def main():
    model_path = '../logs/arch_hps_2023_0530_1637/arch_hps_e30.pth'
    arch_path = 'arch_hps.json'

    DEVICE = 'cuda'
    TRAIN_DIR = 'E:/Dataset/nyu'

    # get arch from json file
    arch = load_arch(arch_path)

    # prepare model
    model = CoNALArch(arch).to(DEVICE)

    model.load_state_dict(torch.load(model_path))

    # prepare dataloaders
    test_loader = get_loaders(TRAIN_DIR, stage='test')
    task_metric = test(model, test_loader)
    return task_metric

if __name__ == '__main__':
    task_metric = main()
    print(task_metric)