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
            "best_acc, "
            "num_shot\n "
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
                "{best_acc:.02f}, "
                "{num_shot}\n"
            ).format(**kwargs)
        )

def cal_acc(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy*100, mean_ent

import numpy as np
from sklearn import svm
def extract_feat(netF, netB, loader, num_batch):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(num_batch):
            data = iter_test.next()
            inputs = data[0].cuda()
            feats = netB(netF(inputs))
            if start_test:
                all_feats = feats.float().cpu()
                start_test = False
            else:
                all_feats = torch.cat((all_feats, feats.float().cpu()), 0)

    return all_feats

def cal_a_dis(netF, netB, loader_a, loader_b):
    print('==> Get A-distance')

    netF.eval()
    netB.eval()
    num_batch=20

    feats_a = extract_feat(netF, netB, loader_a, num_batch)
    feats_b = extract_feat(netF, netB, loader_b, num_batch)
    print('Dataset a sampled size:', feats_a.size())
    print('Dataset b sampled size:', feats_b.size())

    netF.train()
    netB.train()

    a_dis = proxy_a_distance(feats_a.numpy(), feats_b.numpy(), verbose=True)
    print('A-distance:', a_dis)
    return a_dis


def proxy_a_distance(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    C_list = np.logspace(-5, 4, 10)
    C_list = np.logspace(-5, 2, 8)

    half_source, half_target = int(nb_source/2), int(nb_target/2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    return 2 * (1. - 2 * best_risk)











