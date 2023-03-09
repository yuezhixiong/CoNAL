import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.architecture.abstract_arch import AbsArchitecture


class _transform_resnet_nyu(nn.Module):
    def __init__(self, encoder_list, task_name, group, device):
        super(_transform_resnet_nyu, self).__init__()
        
        self.task_name = task_name
        self.task_num = len(task_name)
        self.device = device

        self.group = group
        self.group_num = max([len(x) for x in group])
        # print('group_num', self.group_num)
        self.resnet_conv = nn.ModuleDict({group: nn.Sequential(encoder_list[gn].conv1, encoder_list[gn].bn1, 
                                                              encoder_list[gn].relu, encoder_list[gn].maxpool) for gn, group in enumerate(self.group[0])})

        
        self.layer_num = len(self.group) - 1
        self.resnet_layer = nn.ModuleDict({})
        for i in range(self.layer_num):
            self.resnet_layer[str(i)] = nn.ModuleList([])
            for gn in range(len(self.group[i+1])):
                encoder = encoder_list[gn]
                self.resnet_layer[str(i)].append(eval('encoder.layer'+str(i+1)))
        
    def forward(self, inputs):

        ss_rep = {i: [None]*self.group_num for i in range(5)}
        for i in range(5):  # i: layer idx
            for gn, group in enumerate(self.group[i]):  # gn: group idx
                if i == 0:
                    ss_rep[i][gn] = self.resnet_conv[group](inputs)
                else:
                    ss_rep[i][gn] = self.resnet_layer[str(i-1)][gn](ss_rep[i-1][gn])
            for gn, rep in enumerate(ss_rep[i]):
                if rep is None:
                    # print('none group', gn)
                    ss_rep[i][gn] = ss_rep[i][gn-1]
        return ss_rep[4]

class CoNAL(AbsArchitecture):
    r"""
    Learning Conflict-Noticed Architecture for Multi-Task Learning
    """
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(CoNAL, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        self.group = [['012'], ['012'], ['012'], ['01', '2'], ['01', '2']]  # conv1, layer1, layer2, layer3, layer4
        group_num = max([len(x) for x in self.group])
        self.task_group = {'segmentation':'01', 'depth':'01', 'normal':'2'}
        self.encoder = nn.ModuleList([self.encoder_class() for _ in range(group_num)])
        self.encoder = _transform_resnet_nyu(self.encoder, task_name, self.group, device)

    def forward(self, inputs, task_name=None):
        r"""
        Args: 
            inputs (torch.Tensor): The input data.
            task_name (str, default=None): The task name corresponding to ``inputs`` if ``multi_input`` is ``True``.
        
        Returns:
            dict: A dictionary of name-prediction pairs of type (:class:`str`, :class:`torch.Tensor`).
        """
        out = {}
        s_rep = self.encoder(inputs)
        same_rep = True if not isinstance(s_rep, list) and not self.multi_input else False
        for tn, task in enumerate(self.task_name):
            if task_name is not None and task != task_name:
                continue

            key = self.task_group[task]
            gn = 0
            for idx, group in enumerate(self.group[-1]):
                if key == group:
                    gn = idx
                    break
            # print(task, key, gn)
            ss_rep = s_rep[gn] if isinstance(s_rep, list) else s_rep
            ss_rep = self._prepare_rep(ss_rep, gn, same_rep)
            out[task] = self.decoders[task](ss_rep)
        return out