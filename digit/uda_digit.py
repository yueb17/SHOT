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

from utils import apply_mask_forward, check_sparsity
from utils import write_result_to_csv
from utils import proxy_a_distance, cal_a_dis
from utils import cal_acc, cal_loss
import copy

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def digit_load(args): 
    train_bs = args.batch_size
    if args.dset == 's2m':
        train_source = svhn.SVHN('./data/svhn/', split='train', download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
        test_source = svhn.SVHN('./data/svhn/', split='test', download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))  
        train_target = mnist.MNIST_idx('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))      
        test_target = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
    elif args.dset == 'u2m':
        train_source = usps.USPS('./data/usps/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        test_source = usps.USPS('./data/usps/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))    
        train_target = mnist.MNIST_idx('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))    
        test_target = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
    elif args.dset == 'm2u':
        train_source = mnist.MNIST('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        test_source = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))

        train_target = usps.USPS_idx('./data/usps/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        test_target = usps.USPS('./data/usps/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(train_source, batch_size=train_bs, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["source_te"] = DataLoader(test_source, batch_size=train_bs*2, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["target"] = DataLoader(train_target, batch_size=train_bs, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["target_te"] = DataLoader(train_target, batch_size=train_bs, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(test_target, batch_size=train_bs*2, shuffle=False, 
        num_workers=args.worker, drop_last=False)

    train_target_copy = copy.deepcopy(train_target)
    ob_train_target, _ = torch.utils.data.random_split(train_target_copy, [args.num_shot, len(train_target_copy)-args.num_shot])
    dset_loaders['ob_target'] = DataLoader(ob_train_target, batch_size=train_bs, shuffle=True,
    	num_workers=args.worker, drop_last=False)
    # st()


    return dset_loaders

def train_source(args):
    print('==> Start train source')
    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()  
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = netC(netB(netF(inputs_source)))
        classifier_loss = loss.CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)            
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            acc_s_tr, _ = cal_acc(dset_loaders['source_tr'], netF, netB, netC)
            acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%/ {:.2f}%'.format(args.dset, iter_num, max_iter, acc_s_tr, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()
            
            netF.train()
            netB.train()
            netC.train()

    torch.save(best_netF, osp.join(args.output_dir, "source_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir, "source_C.pt"))

    return netF, netB, netC

def test_target(args):
    print('==> Start test target')
    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()  
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
    log_str = 'Task: {}, Accuracy = {:.2f}%'.format(args.dset, acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def train_target(args):
    print('==> Start train target')
    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()  
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C.pt'    
    netC.load_state_dict(torch.load(args.modelpath))

    print('==> Calculate S-T A-dis: S pretrained model')
    cal_a_dis(netF, netB, dset_loaders['source_tr'], dset_loaders['target'])
    
    # add potential prune pretrained source model
    if args.pruner_s == 'non':
    	print('NOT prune S pretrained model before T finetune')
    	mask = 'no mask'
    else:
        print("==> Start prune S pretrained model using:", args.pruner_s)

        print("==> Check acc before pruning")
        vanilla_test(args, dset_loaders['source_te'], netF, netB, netC)
        vanilla_test(args, dset_loaders['test'], netF, netB, netC)

        print('==> Obtain pruner for source model')
        netF_s = copy.deepcopy(netF)
        netF_t = copy.deepcopy(netF)
        netF_l = copy.deepcopy(netF)

        mask_s = pruner_to_prune(args, netF_s, netB, netC, 's_snip', dset_loaders)
        mask_t = pruner_to_prune(args, netF_t, netB, netC, 't_snip', dset_loaders)
        mask_l = pruner_to_prune(args, netF_l, netB, netC, 'l1', dset_loaders)

        print("==> Prune once")
        st()
        check_sparsity(netF)
        print('Full S model', cal_loss(netF, netB, netC, dset_loaders, args))
        apply_mask_forward(netF_s, mask_s)
        apply_mask_forward(netF_t, mask_t)
        apply_mask_forward(netF_l, mask_l)
        check_sparsity(netF)
        print('Pruned S model', cal_loss(netF_s, netB, netC, dset_loaders, args))
        print('Pruned S model', cal_loss(netF_t, netB, netC, dset_loaders, args))
        print('Pruned S model', cal_loss(netF_l, netB, netC, dset_loaders, args))
        st()

        print("==> Check acc just after pruning")
        vanilla_test(args, dset_loaders['source_te'], netF, netB, netC)
        vanilla_test(args, dset_loaders['test'], netF, netB, netC)

        print('==> Calculate S-T A-dis: S pretrained pruned model')
        cal_a_dis(netF, netB, dset_loaders['source_tr'], dset_loaders['target'])

    print('==> Finetune T')
    netF, netB, netC, best_test_acc = vanilla_train_target(args, netF, netB, netC, mask, dset_loaders['ob_target'], dset_loaders['test'])

    vanilla_test(args, dset_loaders['source_te'], netF, netB, netC)
    vanilla_test(args, dset_loaders['test'], netF, netB, netC)

    print('==> S-T A-dis: T pretrained model')
    cal_a_dis(netF, netB, dset_loaders['source_tr'], dset_loaders['target'])

    if args.pruner_t == 'non':
        print('NO prune T model and no further finetune')
    else:
        print('==> Start prune pre-pretrained T model')

        print('==> Check acc before pruning')
        vanilla_test(args, dset_loaders['test'], netF, netB, netC)

        print("==> Obtain pruner for T model")
        mask = pruner_to_prune(args, netF, netB, netC, args.pruner_t, dset_loaders)

        print("==> Prune once")
        check_sparsity(netF)
        apply_mask_forward(netF, mask)
        check_sparsity(netF)

        print("==> Check acc just after pruning")
        vanilla_test(args, dset_loaders['test'], netF, netB, netC)

        print('==> Start T pruned finetuning')
        netF, netB, netC = vanilla_train_target(args, netF, netB, netC, mask, dset_loaders['target'], dset_loaders['test'])


    if args.save_acc:
        print('==> Saving results')
        write_result_to_csv(args,
            seed=args.seed,
            dataset=args.dset,
            pruner_s=args.pruner_s,
            pruner_t=args.pruner_t,
            stage_pr=args.stage_pr,
            global_pr=args.global_pr,
            dd_loss=args.dd_loss,
            best_acc=best_test_acc,
            num_shot=args.num_shot,
            )

    return netF, netB, netC

def pruner_to_prune(args, netF, netB, netC, pruner_name, dset_loaders):
	pruner = pruner_dict[pruner_name].Pruner(netF, args)
	if pruner_name == 'l1':
		print('==> Using l1')
		pruner.prune()
	elif pruner_name == 't_snip':
		print('Using target snip')
		pruner.prune(netF, netB, netC, 1-args.global_pr, dset_loaders['ob_target'], torch.device("cuda:0"))
	elif pruner_name == 's_snip':
		print('Using source snip')
		pruner.prune(netF, netB, netC, 1-args.global_pr, dset_loaders['source_tr'], torch.device("cuda:0"))
	else:
		raise NotImplementedError

	return pruner.mask

def vanilla_test(args, loader, netF, netB, netC):
	netF.eval()
	netB.eval()
	netC.eval()
	acc, _ = cal_acc(loader, netF, netB, netC)
	print('Acc:', acc)
	netF.train()
	netB.train()
	netC.train()

def vanilla_train_target(args, netF, netB, netC, mask, loader, loader_val):
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(loader)
    interval_iter = len(loader)
    iter_num = 0
    best_test_acc = 0

    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            inputs_test, labels_test, tar_idx = iter_test.next()
        except:
            iter_test = iter(loader)
            inputs_test, labels_test, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_test = inputs_test.cuda()
        labels_test = labels_test.cuda()

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        total_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            im_loss = entropy_loss * args.ent_par
            total_loss += im_loss

        if args.supervised_target:
            label_loss = loss.CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_test, labels_test)
            total_loss += label_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if mask != 'no mask':
            apply_mask_forward(netF, mask)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            acc, _ = cal_acc(loader_val, netF, netB, netC)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.dset, iter_num, max_iter, acc)
            print(log_str+'\n')
            netF.train()
            netB.train()

            if acc >= best_test_acc:
                best_test_acc = acc

    print('Best test acc:', best_test_acc)

    return netF, netB, netC, best_test_acc

def obtain_label(loader, netF, netB, netC, args, c=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')
    return pred_label.astype('int')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=30, help="maximum epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='s2m', choices=['u2m', 'm2u','s2m'])
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--issave', type=bool, default=True)

    parser.add_argument('--pruner_s', type=str, default='full', choices=['non', 'l1', 't_snip', 's_snip'])
    parser.add_argument('--stage_pr', type=str, default="")
    parser.add_argument('--pick_pruned', type=str, default='min', choices=['min', 'max', 'rand'])

    parser.add_argument('--global_pr', type=float, default=0.0)
    parser.add_argument('--dd_loss', type=str, default='', choices=['', 'label', 'ent', 'mix'])
    parser.add_argument('--dd_gent', type=bool, default=True)

    parser.add_argument('--save_acc', action='store_true')
    parser.add_argument('--save_file', type=str)

    parser.add_argument('--pruner_t', type=str, default='non', choices=['non', 'l1', 't_snip', 's_snip'])
    parser.add_argument('--supervised_target', action='store_true')
    parser.add_argument('--num_shot', type=int, default=10)

    args = parser.parse_args()
    args.class_num = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # SEED = args.seed
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)
    # np.random.seed(SEED)
    # random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    args.output_dir = osp.join(args.output, 'seed' + str(args.seed), args.dset)
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if not osp.exists(osp.join(args.output_dir + '/source_F.pt')):
        args.out_file = open(osp.join(args.output_dir, 'log_src.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train_source(args)
        test_target(args)

    args.savename = 'par_' + str(args.cls_par)
    args.out_file = open(osp.join(args.output_dir, 'log_tar_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    test_target(args)
    train_target(args)
