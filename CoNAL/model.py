import torch
import torch.nn.functional as F
import torch.nn as nn
from backbone import NyuModel


class STL(NyuModel):
    def __init__(self):
        super().__init__()
        self.backbones = nn.ModuleList([self.new_bb() for _ in self.tasks])
        
    def forward(self, x):
        img_size  = x.size()[-2:]

        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](self.backbones[i](x)), img_size, mode='bilinear', align_corners=True)
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out


class HPS(NyuModel):
    def __init__(self):
        super().__init__()
        
        self.backbone = self.new_bb()
        
    def forward(self, x):
        img_size  = x.size()[-2:]
        x = self.backbone(x)
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](x), img_size, mode='bilinear', align_corners=True)
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out
    
    def get_share_params(self):
        return self.backbone.parameters()
    
class CoNAL(NyuModel):
    def __init__(self, gumble=False):
        super().__init__()

        self.gumble = gumble
        alpha = torch.ones(self.task_num, 6)
        self.alpha = nn.Parameter(alpha)

        self.backbone = self.new_bb()
        self.backbones = nn.ModuleList([self.new_bb() for _ in self.tasks])

        self.task_conv = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu1, self.backbone.maxpool)
        self.sl0 = nn.ModuleList([nn.Sequential(
            bb.layer1, bb.layer2, bb.layer3, bb.layer4) for bb in self.backbones])
        self.sl1 = nn.ModuleList([nn.Sequential(
            bb.layer2, bb.layer3, bb.layer4) for bb in self.backbones])
        self.sl2 = nn.ModuleList([nn.Sequential(
            bb.layer3, bb.layer4) for bb in self.backbones])
        self.sl3 = nn.ModuleList([nn.Sequential(
            bb.layer4) for bb in self.backbones])
        self.layer_list = [self.task_conv, self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]
        self.sl_list = [self.backbones, self.sl0, self.sl1, self.sl2, self.sl3]
    
    def forward(self, x, tau=1):
        img_size  = x.size()[-2:]

        t_outs = [0,0,0]
        t_alpha = F.softmax(self.alpha, dim=-1) if not self.gumble else F.gumbel_softmax(self.alpha, tau=tau, hard=True, dim=-1)
        
        for i,layer in enumerate(self.layer_list):
            for tn in range(self.task_num):
                t_outs[tn] += self.sl_list[i][tn](x) * t_alpha[tn, i]
            x = layer(x)
        for tn in range(self.task_num):
            t_outs[tn] += x * t_alpha[tn,-1]

        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](t_outs[i]), img_size, mode='bilinear', align_corners=True)
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out

    def weight_parameters(self):
        for name, param in self.named_weight_parameters():
            yield param

    def named_weight_parameters(self):
        return filter(lambda x: not x[0].startswith('alpha'),
                      self.named_parameters())

    def arch_parameters(self):
        for name, param in self.named_arch_parameters():
            print('arch param:', name)
            yield param

    def named_arch_parameters(self):
        return filter(lambda x: x[0].startswith('alpha'),
                      self.named_parameters())

    def layer0_parameters(self):
        name_params = filter(lambda x: x[0].startswith('backbone.conv1') 
                             or x[0].startswith('backbone.bn1'), 
                             self.named_parameters())
        for name, param in name_params:
            yield param

    def layer1_parameters(self):
        name_params = filter(lambda x: x[0].startswith('backbone.layer1'),
                             self.named_parameters())
        for name, param in name_params:
            yield param

    def layer2_parameters(self):
        name_params = filter(lambda x: x[0].startswith('backbone.layer2'),
                             self.named_parameters())
        for name, param in name_params:
            yield param

    def layer3_parameters(self):
        name_params = filter(lambda x: x[0].startswith('backbone.layer3'),
                             self.named_parameters())
        for name, param in name_params:
            yield param

    def layer4_parameters(self):
        name_params = filter(lambda x: x[0].startswith('backbone.layer4'),
                             self.named_parameters())
        for name, param in name_params:
            yield param

    def private_parameters(self):
        name_params = filter(lambda x: not x[0].startswith('backbone.') 
                             and not x[0].startswith('alpha'), 
                             self.named_parameters())
        for name, param in name_params:
            yield param


class CoNALArch(NyuModel):
    def __init__(self, arch={}):
        super().__init__()
        
        group = [[] for _ in range(self.num_branch_points)]
        for i in range(self.task_num):
            branch_point = arch[str(i)]
            if branch_point == self.num_branch_points -1:
                group[branch_point].append(i)
            else:
                group[branch_point].append([i])
        self.group = group
        print('branchs:', group)

        self.backbone = self.new_bb()

        self.task_conv = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu1,
            self.backbone.maxpool)        
        self.backbones = nn.ModuleList([self.new_bb() for _ in group[0]])
        self.sl0 = nn.ModuleList([nn.Sequential(
            self.new_bb().layer1, 
            self.new_bb().layer2,
            self.new_bb().layer3, 
            self.new_bb().layer4) for _ in group[1]])
        self.sl1 = nn.ModuleList([nn.Sequential(
            self.new_bb().layer2,
            self.new_bb().layer3, 
            self.new_bb().layer4) for _ in group[2]])
        self.sl2 = nn.ModuleList([nn.Sequential(
            self.new_bb().layer3, 
            self.new_bb().layer4) for _ in group[3]])
        self.sl3 = nn.ModuleList([nn.Sequential(
            self.new_bb().layer4) for _ in group[4]])
        self.layer_list = [self.task_conv, self.backbone.layer1, self.backbone.layer2, 
            self.backbone.layer3, self.backbone.layer4]
        self.sl_list = [self.backbones, self.sl0, self.sl1, self.sl2, self.sl3]

    def forward(self, x):
        img_size  = x.size()[-2:]
        t_outs = [0] * self.task_num
        for i,layer in enumerate(self.layer_list):
            for tn in range(self.task_num):
                for gn,group in enumerate(self.group[i]):
                    if tn in group:
                        t_outs[tn] = self.sl_list[i][gn](x)
            x = layer(x)

        for tn in range(self.task_num):
            if tn in self.group[5]:
                t_outs[tn] = x

        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](t_outs[i]), img_size, mode='bilinear', align_corners=True)
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out


def test():
    x = torch.randn((3, 1, 161, 161))
    model = CoNAL()
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()