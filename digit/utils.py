import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle
from data_load import mnist, svhn, usps

from pruner import pruner_dict
from pdb import set_trace as st
import pathlib

def apply_mask_forward(model, mask):
            for name, m in model.named_modules():
                if name in mask:
                    m.weight.data.mul_(mask[name])

def check_sparsity(model):
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                if hasattr(module.weight, 'data'):
                    zero_pos = torch.nonzero(module.weight.data == 0)
                    print(name, 'sparsity:', len(zero_pos)/module.weight.data.numel())

def write_result_to_csv(args, **kwargs):
    results = pathlib.Path(args.save_file)

    if not results.exists():
        results.write_text(
            "seed, "
            "dataset, "
            "pruner_s, "
            "pruner_t, "
            "stage_pr, "
            "global_pr, "
            "dd_loss, "
            "best_acc\n "
        )

    with open(results, "a+") as f:
        f.write(
            (
                "{seed}, "
                "{dataset}, "
                "{pruner_s}, "
                "{pruner_t}, "
                "{stage_pr}, "
                "{global_pr}, "
                "{dd_loss}, "
                "{best_acc:.02f}\n"
            ).format(**kwargs)
        )


