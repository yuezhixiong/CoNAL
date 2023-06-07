import torch, argparse
from model import CoNALArch
from dataset import get_loaders
from utils import *
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test(model, test_loader):
    conf_mat = ConfMatrix(model.class_nb)
    cost = torch.zeros(24)
    avg_cost = torch.zeros(24)
    model.eval()
    with torch.no_grad():
        test_iter = iter(test_loader)
        test_batch = len(test_loader)
        for k in tqdm(range(test_batch)):
            test_data, test_label, test_depth, test_normal = next(test_iter)
            test_data, test_label = test_data.cuda(non_blocking=True), test_label.long().cuda(non_blocking=True)
            test_depth, test_normal = test_depth.cuda(non_blocking=True), test_normal.cuda(non_blocking=True)
            test_pred = model(test_data)
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
    task_metric["0"] = [round(x,6) for x in avg_cost[13:15].tolist()]
    task_metric["1"] = [round(x,6) for x in avg_cost[16:18].tolist()]
    task_metric["2"] = [round(x,6) for x in avg_cost[19:24].tolist()]

    return task_metric


def main(args):
    yaml_path = find_files(args.logdir, '.yml')[0]
    config = yaml_load(yaml_path)
    arch_name = config['arch']['name']
    arch = config['arch']['branch_points']

    model_path = find_files(args.logdir, '.pth')[0]
    print(arch_name, model_path, arch)

    # prepare model
    model = CoNALArch(arch).to(DEVICE)

    model.load_state_dict(torch.load(model_path))

    # prepare dataloaders
    test_loader = get_loaders(args.datadir, stage='test')
    task_metric = test(model, test_loader)

    config["test"] = task_metric
    config["size"] = size_model(model)

    with open(yaml_path, 'w') as file:
        yaml.dump(config, file)

    return task_metric

if __name__ == '__main__':
    parser = argparse.ArgumentParser("test")
    parser.add_argument("--logdir", type=str, default='logs')
    parser.add_argument("--datadir", type=str, default='E:/Dataset/nyu')
    args = parser.parse_args()

    task_metric = main(args)
    print(task_metric)