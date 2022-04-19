import torch
import torch.nn as nn
import copy
import time
import numpy as np
import torch.optim as optim
from .meta_pruner import MetaPruner

import torch.nn.functional as F
import types
from pdb import set_trace as st
import network
import loss as loss_package

def snip_forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias, self.stride, self.padding, self.dilation, self.groups)

def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)

class Pruner(MetaPruner):
    def __init__(self, model, args):
        self.args = args    

    def prune(self, netF, netB, netC, keep_ratio, train_dataloader, device):
        data = next(iter(train_dataloader))
        inputs = data[0]
        targets = data[1]
        inputs = inputs.to(device)
        targets = targets.to(device)

        netF = copy.deepcopy(netF)
        # netB = copy.deepcopy(netB)
        # netC = copy.deepcopy(netC)

        for layer in netF.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                # nn.init.xavier_normal_(layer.weight)
                layer.weight.requires_grad = False

            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(snip_forward_conv2d, layer)

            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(snip_forward_linear, layer)

        # for layer in netB.modules():
        #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #         layer.weight.requires_grad = False

        # for layer in netC.modules():
        #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #         layer.weight.requires_grad = False

        netF.zero_grad()
        outputs = netF.forward(inputs)
        outputs = netB.forward(outputs)
        outputs = netC.forward(outputs)
        if self.args.dd_loss == 'label':
            print('==> SNIP using label loss')
            loss = F.nll_loss(outputs, targets)
        elif self.args.dd_loss == 'ent':
            print('==> SNIP using entropy loss (non-label)')
            softmax_out = nn.Softmax(dim=1)(outputs)
            entropy_loss = torch.mean(loss_package.Entropy(softmax_out))
            if self.args.dd_gent == True:
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
            loss = entropy_loss
        elif self.args.dd_loss == 'mix':
            print('==> SNIP using mix loss')
            label_loss = F.nll_loss(outputs, targets)
            softmax_out = nn.Softmax(dim=1)(outputs)
            entropy_loss = torch.mean(loss_package.Entropy(softmax_out))
            if self.args.dd_gent == True:
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
            loss = label_loss + entropy_loss
        else:
            raise NotImplementedError

        loss.backward()
        # st()

        grads_abs = []
        for layer in netF.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                grads_abs.append(torch.abs(layer.weight.data))

        all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)

        num_params_to_keep = int(len(all_scores) * keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        keep_masks = []
        for g in grads_abs:
            keep_masks.append(((g / norm_factor) >= acceptable_score).float())

        self.mask = {}
        name_ = []
        mask_ = []
        idx = 0
        for name, module in netF.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                name_.append(name)
                mask_.append(keep_masks[idx])
                idx = idx + 1

        for i in range(idx):
            self.mask[name_[i]] = mask_[i]
        print('mask obtained')

