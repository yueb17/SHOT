import torch
import torch.nn as nn
import copy
import time
import numpy as np
from math import ceil, sqrt
from collections import OrderedDict

from pdb import set_trace as st

def parse_prune_ratio_vgg(sstr, num_layers=20):
    # example: [0-4:0.5, 5:0.6, 8-10:0.2]
    out = np.zeros(num_layers)
    if '[' in sstr:
        sstr = sstr.split("[")[1].split("]")[0]
    else:
        sstr = sstr.strip()
    for x in sstr.split(','):
        k = x.split(":")[0].strip()
        v = x.split(":")[1].strip()
        if k.isdigit():
            out[int(k)] = float(v)
        else:
            begin = int(k.split('-')[0].strip())
            end = int(k.split('-')[1].strip())
            out[begin : end+1] = float(v)
    return list(out)

class MetaPruner:
    def __init__(self, model, args):
        self.model = model
        self.args = args

        self.learnable_layers = (nn.Conv2d, nn.Linear) # Note: for now, we only focus on weights in Conv and FC modules, no BN.
        self.layers = OrderedDict() # learnable layers

        self.kept_wg = {}
        self.pruned_wg = {}
        self.get_pr() # set up pr for each layer
        
    def _pick_pruned(self, w_abs, pr, mode="min", name=None):
        if pr == 0:
            return []
        w_abs_list = w_abs # .flatten()
        n_wg = len(w_abs_list)
        # st()
        n_pruned = min(ceil(pr * n_wg), n_wg - 1) # do not prune all
        if mode == "rand":
            out = np.random.permutation(n_wg)[:n_pruned]
        elif mode == "min":
            out = w_abs_list.sort()[1][:n_pruned]
            out = out.data.cpu().numpy()
        elif mode == "max":
            out = w_abs_list.sort()[1][-n_pruned:]
            out = out.data.cpu().numpy()
        
        return out
    
    def get_pr(self):
        self.pr = {}
        num_learnable_layers = 0
        for name, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers):
                num_learnable_layers += 1
        self.args.stage_pr = parse_prune_ratio_vgg(self.args.stage_pr, num_layers=num_learnable_layers)
        print('Given pr:', self.args.stage_pr)

        idx_learnable_layers = 0
        for name, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers):
                self.pr[name] = self.args.stage_pr[idx_learnable_layers]
                idx_learnable_layers += 1
        print('pr details:', self.pr)

    def _get_kept_wg_L1(self):    
        for name, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers):
                shape = m.weight.data.shape
                score = m.weight.abs().flatten()
                    
                self.pruned_wg[name] = self._pick_pruned(score, self.pr[name], self.args.pick_pruned, name)
                self.kept_wg[name] = list(set(range(len(score))) - set(self.pruned_wg[name]))   

    def _prune_and_build_new_model(self):
        self._get_masks()
        return

    def _get_masks(self):
        '''Get masks for unstructured pruning
        '''
        self.mask = {}
        for name, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers):
                mask = torch.ones_like(m.weight.data).cuda().flatten()
                pruned = self.pruned_wg[name]
                mask[pruned] = 0
                self.mask[name] = mask.view_as(m.weight.data)
        print('==> Mask obtained')


