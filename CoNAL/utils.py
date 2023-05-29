import torch, time, os, copy, json
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

class NOCLGrad():
    def __init__(self, optimizer, branch=False):
        self._optim = optimizer
        self.conflict = []
        self.c_flag = 0
        self.branch = branch
        self.noc_counter = 0
        return

    def print_noc_counter(self):
        self.noc_counter = 0

    @property
    def optimizer(self):
        return self._optim

    @property
    def param_groups(self):
        return self._optim.param_groups

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def backward(self, objectives):
        '''
        calculate the gradient of the parameters
        input:
        - objectives: a list of objectives
        '''
        num_task = len(objectives)
        self.conflict = np.zeros([num_task, num_task])
        layer_list = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4',]

        
        obj_layer_grads, shapes = self._pack_grad(objectives, layer_list)

        layer_grad = self._project_conflicting(obj_layer_grads, layer_list)
        noc_grad = self._unflatten_grad(layer_grad, shapes[0], layer_list)
        self._set_grad(noc_grad, layer_list)

        return

    def _project_conflicting(self, grads, layer_list):
        noc_grad, num_task = copy.deepcopy(grads), len(grads)

        for idx in range(len(layer_list)):
            c_flag = 0
            for idx_i,g_i in enumerate(grads[:-1]):
                for idx2,g_j in enumerate(grads[idx_i+1:]):
                    idx_j = idx_i+1 + idx2
                    g_i_g_j = torch.dot(g_i[idx], g_j[idx])
                    if g_i_g_j < 0:
                        for k, _ in enumerate(grads):
                            noc_grad[k][idx] = torch.zeros_like(noc_grad[k][idx])
                        self.noc_counter += 1
                        c_flag = 1
                    if c_flag:
                        break
                if c_flag:
                        break
                
        merged_grad = []
        for idx in range(len(layer_list)):
            layer_grad = torch.zeros_like(grads[0][idx]).to(grads[0][idx].device)
            layer_grad = torch.stack([g[idx] for g in noc_grad]).mean(dim=0)
            merged_grad.append(layer_grad)
        return merged_grad

    def _set_grad(self, grads, layer_list):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            if group['name'] in layer_list:
                for p in group['params']:
                    p.grad = grads[idx]
                    idx += 1
        return

    def _pack_grad(self, objectives, layer_list):
        '''
        pack the gradient of the parameters of the network for each objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            layer_grad = self._retrieve_grad(layer_list)
            grads_layer, shapes_layer, has_grads_layer = [], [], []
            for layer in layer_list:
                grad, shape, has_grad = layer_grad[layer]
                grads_layer.append(self._flatten_grad(grad))
                shapes_layer.append(shape)
            grads.append(grads_layer)
            shapes.append(shapes_layer)
            
        return grads, shapes

    def _unflatten_grad(self, grads, layer_shape, layer_list):
        unflatten_grad, idx = [], 0
        for i in range(len(layer_list)):
            grad = grads[i]
            # print(grad.shape)
            shapes = layer_shape[i]
            for shape in shapes:
                # print(shape)
                length = np.prod(shape)
                unflatten_grad.append(grad[idx:idx+length].view(shape).clone())
                idx += length
            idx = 0
        return unflatten_grad

    def _flatten_grad(self, grads):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self, layer_list):
        '''
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        layer_grad = {}
        for group in self._optim.param_groups:
            grad, shape, has_grad = [], [], []
            if group['name'] in layer_list:
                for p in group['params']:
                    if p.grad is None:
                        shape.append(p.shape)
                        grad.append(torch.zeros_like(p).to(p.device))
                        has_grad.append(torch.zeros_like(p).to(p.device))
                        continue
                    shape.append(p.grad.shape)
                    grad.append(p.grad.clone())
                    has_grad.append(torch.ones_like(p).to(p.device))
                layer_grad[group['name']] = [grad, shape, has_grad]
        return layer_grad


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

    return loss.to(device)

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
            test_loss = [loss_fn(test_pred[0], test_label, 'semantic'),
                         loss_fn(test_pred[1], test_depth, 'depth'),
                         loss_fn(test_pred[2], test_normal, 'normal')]

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
    print('logging at ', logdir)
    return logdir


def output_arch(alpha_arr, logdir):
    arch_path = os.path.join(logdir, 'arch.json')
    print('dump searched architecture to ', arch_path)
    arch_dict = {}
    for idx, alpha in enumerate(alpha_arr):
        arch_dict[idx] = int(alpha.argmax())
        print('task {} branch at {}'.format(idx, arch_dict[idx]))
    with open(arch_path, "w") as outfile:
        json.dump(arch_dict, outfile)


def load_arch(arch_file ='arch.json'):
    with open(arch_file) as f:
        arch = json.load(f)
    return arch