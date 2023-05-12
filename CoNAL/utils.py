import torch, time, os
import torch.nn as nn
import torch.nn.functional as F


class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        with torch.no_grad():
            n = self.num_classes
            if self.mat is None:
                self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
            with torch.no_grad():
                k = (target >= 0) & (target < n)
                inds = n * target[k].to(torch.int64) + pred[k]
                self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        with torch.no_grad():
            h = self.mat.float()
            acc = torch.diag(h).sum() / h.sum()
            iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
            return torch.mean(iu).item(), acc.item()


def depth_error(x_pred, x_output):
    with torch.no_grad():
        device = x_pred.device
        binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
        x_pred_true = x_pred.masked_select(binary_mask)
        x_output_true = x_output.masked_select(binary_mask)
        abs_err = torch.abs(x_pred_true - x_output_true)
        rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
        return (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item(), \
               (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()


def normal_error(x_pred, x_output):
    with torch.no_grad():
        binary_mask = (torch.sum(x_output, dim=1) != 0)
        error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1))#.detach().cpu().numpy()
    #     error = np.degrees(error)
        error = torch.rad2deg(error)
        return torch.mean(error).item(), torch.median(error).item(), \
               torch.mean((error < 11.25)*1.0).item(), torch.mean((error < 22.5)*1.0).item(), \
               torch.mean((error < 30)*1.0).item()
    
def set_param(curr_mod, name, param=None, mode='update'):
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                return set_param(mod, rest, param, mode=mode)
    else:
        if mode == 'update':
            delattr(curr_mod, name)
            setattr(curr_mod, name, param)
        elif mode == 'get':
            if hasattr(curr_mod, name):
                p = getattr(curr_mod, name)
                return p


from backbone import *

def get_fix_model(fix_json, fix_weight):
    model = Router().cuda()
    checkpoint = torch.load(fix_weight)
    model.load_state_dict(checkpoint)
    apply_fixed_architecture(model, fix_json)
    return model

def test(model, nyuv2_test_loader):
    conf_mat = ConfMatrix(model.class_nb)
    cost = torch.zeros(24)
    avg_cost = torch.zeros(24)
    model.eval()
    with torch.no_grad():  # operations inside don't track history
        test_dataset = iter(nyuv2_test_loader)
        test_batch = len(nyuv2_test_loader)
        for k in range(test_batch):
            test_data, test_label, test_depth, test_normal = test_dataset.next()
            test_data, test_label = test_data.cuda(non_blocking=True), test_label.long().cuda(non_blocking=True)
            test_depth, test_normal = test_depth.cuda(non_blocking=True), test_normal.cuda(non_blocking=True)
            test_pred = model.predict(test_data)
            test_loss = [model_fit(test_pred[0], test_label, 'semantic'),
                         model_fit(test_pred[1], test_depth, 'depth'),
                         model_fit(test_pred[2], test_normal, 'normal')]

            conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

            cost[12] = test_loss[0].item()
            cost[15] = test_loss[1].item()
            cost[16], cost[17] = depth_error(test_pred[1], test_depth)
            cost[18] = test_loss[2].item()
            cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(test_pred[2], test_normal)
            avg_cost[12:] += cost[12:] / test_batch

        # compute mIoU and acc
        avg_cost[13], avg_cost[14] = conf_mat.get_metrics()
    return avg_cost

def create_logdir(LOG_DIR):
    folder_name = time.strftime("_%Y_%m%d_%H%M")
    logdir = os.path.join(LOG_DIR, folder_name)
    os.makedirs(logdir, exist_ok=True)
    return logdir