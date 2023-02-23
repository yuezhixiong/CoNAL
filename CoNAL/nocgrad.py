import torch
import torch.nn as nn
import numpy as np
import copy


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


class NOCLGrad_v1():
    def __init__(self, optimizer, branch=False):
        self._optim = optimizer
        self.conflict = []
        self.c_flag = 0
        self.branch = branch
        self.noc_counter = 0
        return

    def print_noc_counter(self):
        print(self.noc_counter)
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

    def noc_backward(self, objectives):
        '''
        calculate the gradient of the parameters
        input:
        - objectives: a list of objectives
        '''
        num_task = len(objectives)
        self.conflict = np.zeros([num_task, num_task])
        layer_list = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4',]
        if not self.branch:
            for name in layer_list:
                grads, shapes, has_grads = self._pack_grad(objectives, name)
                noc_grad = self._project_conflicting(grads, has_grads)
                noc_grad = self._unflatten_grad(noc_grad, shapes[0])
                self._set_grad(noc_grad, name)
        else:
            self.c_flag = 0
            for idx,name in enumerate(layer_list):
                # print('looking at', name)
                grads, shapes, has_grads = self._pack_grad(objectives, name)
                noc_grad = self._project_conflicting(grads, has_grads)
                noc_grad = self._unflatten_grad(noc_grad, shapes[0])
                self._set_grad(noc_grad, name)
                if self.c_flag:
                    # print('conflict break')
                    for zero_name in layer_list[idx:]:
                        self._zero_grad(zero_name)
                    break
                self.noc_counter = self.noc_counter + 1
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        noc_grad, num_task = copy.deepcopy(grads), len(grads)

        for idx_i,g_i in enumerate(grads[:-1]):
            for idx2,g_j in enumerate(grads[idx_i+1:]):
                idx_j = idx_i+1 + idx2
                if self.conflict[idx_i,idx_j]:
                    noc_grad[idx_i][shared] = 0
                    noc_grad[idx_j][shared] = 0
                    continue
                else:
                    g_i_g_j = torch.dot(g_i, g_j)
                    if g_i_g_j < 0:
                        noc_grad[idx_i][shared] = 0
                        noc_grad[idx_j][shared] = 0
                        self.conflict[idx_i,idx_j] = 1
                        self.conflict[idx_j,idx_i] = 1
                        self.c_flag = 1
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        merged_grad[shared] = torch.stack([g[shared]
                                           for g in noc_grad]).mean(dim=0)
        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in noc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads, group_name):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            if group['name'] == group_name:
                for p in group['params']:
    #                 print(p.shape)
                    # if p.grad is None: continue
                    p.grad = grads[idx]
                    idx += 1
                break
        return

    def _zero_grad(self, group_name):
        '''
        set the modified gradients to the network
        '''

        for group in self._optim.param_groups:
            if group['name'] == group_name:
                for p in group['params']:
                    p.grad = 0
            break
        return

    def _pack_grad(self, objectives, name):
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
            grad, shape, has_grad = self._retrieve_grad(name)
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self, group_name):
        '''
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            if group['name'] == group_name:
                for p in group['params']:
                    if p.grad is None:
                        shape.append(p.shape)
                        grad.append(torch.zeros_like(p).to(p.device))
                        has_grad.append(torch.zeros_like(p).to(p.device))
                        continue
                    shape.append(p.grad.shape)
                    grad.append(p.grad.clone())
                    has_grad.append(torch.ones_like(p).to(p.device))
                break
        return grad, shape, has_grad


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 4)

    def forward(self, x):
        feat = self._linear(x)
        return self._head1(feat), self._head2(feat)