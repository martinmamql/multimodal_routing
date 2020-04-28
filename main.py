import argparse
import os
import random
import shutil
import time
import warnings
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.utils import get_data
from src.capsule_model import CapsModel 
from src.transformer_model import Transformer 
from src.eval_metrics import eval_mosei_senti, eval_iemocap, eval_mosei_emo
parser = argparse.ArgumentParser(description='PyTorch Capsule Learner')

"""Argument Parsing
"""

# Train
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosei_senti',
                    help='dataset to use (default: mosei_senti)')
parser.add_argument('--data-path', type=str, default='data', help='path for storing the dataset')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-a', '--arch', metavar='ARCH', default='multimodal_capsule', 
                    help='model architecture: (default: multimodal_capsule)')
parser.add_argument('--config-path', default='./configs/full_rank_2C1F_matrix_for_iterations.json', 
                    type=str, help='path of the config')
#parser.add_argument('--optim', type=str, default='Adam', help='optimizer to use (default: Adam)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clip value (default: 1)')
parser.add_argument('--loss-type', type=str, default="CrossEntropy",
                            help='type of loss (default: CrossEntropy)')
parser.add_argument('--act-type', default='EM', type=str, 
                    help='type of activation. ONES or EM or Hubert (default: EM)')
parser.add_argument('--num-routing', default=1, type=int, help='number of routings performed each layer (default: 2)')
parser.add_argument('--dp', default=0.5, type=float, help='dropout rate inside capsule layers')
parser.add_argument('--patience', default=5, type=int, help='patience for learning rate decay')

# Multimodal Capsule Arguments
parser.add_argument('--layer_norm', action='store_true', help='layer normalization in capsule model (default: False)')
parser.add_argument('--d_mult', default=200, type=int, help='hidden size in multimodal transformer layer')
parser.add_argument('--transformer_layers', default=7, type=int, help='number of layers in the transformer model')
parser.add_argument('--self_transformer_layers', default=3, type=int, help='number of self attention layers in the transformer model')
parser.add_argument('--num_heads', default=20, type=int, help='number of heads in multimodal transformer')
parser.add_argument('--attn_dropout', default=0.0, type=float, help='attention dropout in multimodal transformer')
parser.add_argument('--attn_dropout_a', default=0.0, type=float, help='attention dropout in multimodal transformer for modality audio')
parser.add_argument('--attn_dropout_v', default=0.0, type=float, help='attention dropout in multimodal transformer for modality vision')
parser.add_argument('--relu_dropout', default=0.0, type=float, help='dropout rate after relu in multimodal transformer before prediction')
parser.add_argument('--res_dropout', default=0.0, type=float, help='dropout rate for residue in multimodal transformer before prediction')
parser.add_argument('--out_dropout', default=0.0, type=float, help='output dropout for multimodal transformer')
parser.add_argument('--embed_dropout', default=0.0, type=float, help='embedding dropout for multimodal transformer (also serve as attention dropout for modality text)')
parser.add_argument('--pc_dim', default=64, type=int, help='dimension for primary capsule')
parser.add_argument('--mc_caps_dim', default=64, type=int, help='dimension for main capsule')
parser.add_argument('--dim_pose_to_vote', default=100, type=int, help='dimension from pose to vote')

# Logistics
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-p', '--print-freq', default=30, type=int,
                    metavar='N', help='print frequency (default: 30)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use. (default: 0)')

best_acc1 = 0

"""Set up for multiprocessing distributed data parallel training

"""

def main():

    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 10000)
    print("seed: ", args.seed) 
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True

    if args.dataset == 'mosei_senti':
        args.output_dim = 7
    elif args.dataset == 'mosei_emo':
        args.output_dim = 6
    elif args.dataset == 'iemocap':
        args.output_dim = 4

    ngpus_per_node = torch.cuda.device_count() # 4
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch == "multimodal_capsule":
            model = CapsModel(dataset=args.dataset,
                            backbone=None,
                            loss_type=args.loss_type,
                            act_type=args.act_type,    
                            num_routing=args.num_routing,
                            dp=args.dp,
                            layer_norm=args.layer_norm,
                            d_mult=args.d_mult,
                            num_heads=args.num_heads,
                            transformer_layers=args.transformer_layers,
                            self_transformer_layers=args.self_transformer_layers,
                            multimodal_transformer_layer=True,
                            attn_dropout=args.attn_dropout,
                            attn_dropout_a=args.attn_dropout_a,
                            attn_dropout_v=args.attn_dropout_v,
                            relu_dropout=args.relu_dropout,
                            res_dropout=args.res_dropout,
                            out_dropout=args.out_dropout,
                            embed_dropout=args.embed_dropout,
                            pc_dim=args.pc_dim,
                            mc_caps_dim=args.mc_caps_dim,
                            dim_pose_to_vote=args.dim_pose_to_vote)
        elif args.arch == "vanilla_capsule":
            model = CapsModel(image_dim_size=None,
                            backbone=None,
                            loss_type=args.loss_type,
                            act_type=args.act_type,    
                            num_routing=args.num_routing,
                            dp=args.dp,
                            small_std=True,
                            with_recon=False,
                            sequential_routing=True,
                            multimodal_transformer_layer=False)
        elif args.arch == "multimodal_transformer":
            model = Transformer(early_fusion=False,
                    d_model=100,
                    n_head=5, 
                    dim_feedforward=2048, 
                    dropout=0.1, 
                    num_layers=1, 
                    layer_norm=False, #'Implement layer norm later', 
                    embed_dropout=0.1,
                    output_dim=7,
                    out_dropout=0.5,
                    multimodal_transformer=True) 
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    dataset = args.dataset
    if args.loss_type != "CrossEntropy":
        print("Not supported with other losses")
        assert False
    if dataset == "mosei_senti":
        criterion = nn.CrossEntropyLoss().cuda(args.gpu) 
    elif dataset == "mosei_emo":
        criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)
    elif dataset == "iemocap":
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        print("No dataset found or wrong dataset name")
        assert False
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, factor=0.1, verbose=True)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    # print hyper parameters
    print(args)

    # start loading dataset
    print("Start loading the data....")

    train_data = get_data(args, dataset, 'train')
    valid_data = get_data(args, dataset, 'valid')
    test_data = get_data(args, dataset, 'test')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    print('Finish loading the data....')
    if not args.aligned:
        print("### Note: You are running in unaligned mode.")

    if args.evaluate:
        test_loss, test_results, test_truth, test_results_weighted, test_truths_rounded = validate(test_loader, model, criterion, args)
        if args.dataset == "mosei_senti":
            test_acc, _ = eval_mosei_senti(test_results, test_truth, test_results_weighted, test_truths_rounded, True)
            print("Test Acc {:5.4f}".format(test_acc))
        elif args.dataset == 'iemocap':
            test_acc, test_f1 = eval_mosei_senti(test_results, test_truth, True)
            print('Epoch {:2d} Acc | Test Neutral {:5.4f} | Test Happy {:5.4f} | Test Sad {:5.4f} | Test Angry {:5.4f}'.format(epoch, test_acc[0], test_acc[1], test_acc[2], test_acc[3]))
            print('Epoch {:2d} F1  | Test Neutral {:5.4f} | Test Happy {:5.4f} | Test Sad {:5.4f} | Test Angry {:5.4f}'.format(epoch, test_f1[0], test_f1[1], test_f1[2], test_f1[3]))
        elif args.dataset == "mosei_emo":
            test_acc, test_f1 = eval_mosei_emo(test_results, test_truth, True)
            print('Acc | Test 1 {:5.4f} | Test 2 {:5.4f} | Test 3 {:5.4f} | Test 4 {:5.4f} | Test 5 {:5.4f} | Test 6 {:5.4f}'\
                    .format(test_acc[0], test_acc[1], test_acc[2], test_acc[3], test_acc[4], test_acc[5]))
            print('F1  | Test 1 {:5.4f} | Test 2 {:5.4f} | Test 3 {:5.4f} | Test 4 {:5.4f} | Test 5 {:5.4f} | Test 6 {:5.4f}'\
                    .format(test_f1[0], test_f1[1], test_f1[2], test_f1[3], test_f1[4], test_f1[5]))
        else:
            raise NotImplementedError
        return
   
    # starting training epochs
    best_acc = -1
    best_epoch = -1
    if args.dataset == "iemocap":
        best_acc_neutral = best_acc_happy = best_acc_sad = best_acc_angry = \
                best_f1_neutral = best_f1_happy = best_f1_sad = best_f1_angry = -1
    for epoch in range(args.start_epoch, args.epochs):
        #if args.distributed:
        #    train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_loss, train_results, train_truth, train_results_weighted, train_truths_rounded = train(train_loader, model, criterion, optimizer, epoch, args)
        # evaluate on validation set
        valid_loss, valid_results, valid_truth, valid_results_weighted, valid_truths_rounded = validate(valid_loader, model, criterion, args)
        test_loss, test_results, test_truth, test_results_weighted, test_truths_rounded = validate(test_loader, model, criterion, args)
        scheduler.step(valid_loss)
        if args.dataset == "mosei_senti":
            train_acc, _ = eval_mosei_senti(train_results, train_truth, train_results_weighted, train_truths_rounded, True)
            valid_acc, _ = eval_mosei_senti(valid_results, valid_truth, valid_results_weighted, valid_truths_rounded, True)
            test_acc, _ = eval_mosei_senti(test_results, test_truth, test_results_weighted, test_truths_rounded, True)
            print('Epoch {:2d} | Train Loss {:5.4f} | Valid Loss {:5.4f} | Test Loss {:5.4f} | Train Acc {:5.4f} | Valid Acc {:5.4f} | Test Acc {:5.4f}'
                    .format(epoch, train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc))
            # remember best acc@1 and save checkpoint
            is_best = test_acc > best_acc
            #if is_best:
            #    best_epoch = epoch
            #    best_acc = max(test_acc, best_acc)
            #    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            #            and args.rank % ngpus_per_node == 0):
            #        save_checkpoint({
            #            'epoch': epoch + 1,
            #            'arch': args.arch,
            #            'state_dict': model.state_dict(),
            #            'best_acc': best_acc,
            #            'optimizer' : optimizer.state_dict(),
            #        }, is_best, "mosei_senti_checkpoint_{}.pth.tar".format(random.random()), args.dataset)
        elif args.dataset == "iemocap":
            train_acc, train_f1 = eval_iemocap(train_results, train_truth)
            valid_acc, valid_f1 = eval_iemocap(valid_results, valid_truth)
            test_acc, test_f1 = eval_iemocap(test_results, test_truth)
            #print('Epoch {:2d} Loss| Train Loss {:5.4f} | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, train_loss, valid_loss, test_loss))
            #print('Epoch {:2d} Acc | Train Neutral {:5.4f} | Train Happy {:5.4f} | Train Sad {:5.4f} | Train Angry {:5.4f}'.format(epoch, train_acc[0], train_acc[1], train_acc[2], train_acc[3]))
            #print('Epoch {:2d} Acc | Valid Neutral {:5.4f} | Valid Happy {:5.4f} | Valid Sad {:5.4f} | Valid Angry {:5.4f}'.format(epoch, valid_acc[0], valid_acc[1], valid_acc[2], valid_acc[3]))
            print('Epoch {:2d} Acc | Test Neutral {:5.4f} | Test Happy {:5.4f} | Test Sad {:5.4f} | Test Angry {:5.4f}'.format(epoch, test_acc[0], test_acc[1], test_acc[2], test_acc[3]))
            #print('Epoch {:2d} F1  | Train Neutral {:5.4f} | Train Happy {:5.4f} | Train Sad {:5.4f} | Train Angry {:5.4f}'.format(epoch, train_f1[0], train_f1[1], train_f1[2], train_f1[3]))
            #print('Epoch {:2d} F1  | Valid Neutral {:5.4f} | Valid Happy {:5.4f} | Valid Sad {:5.4f} | Valid Angry {:5.4f}'.format(epoch, valid_f1[0], valid_f1[1], valid_f1[2], valid_f1[3]))
            print('Epoch {:2d} F1  | Test Neutral {:5.4f} | Test Happy {:5.4f} | Test Sad {:5.4f} | Test Angry {:5.4f}'.format(epoch, test_f1[0], test_f1[1], test_f1[2], test_f1[3]))
            print('-' * 50)
            best_acc_neutral = max(best_acc_neutral, test_acc[0])
            best_acc_happy = max(best_acc_happy, test_acc[1])
            best_acc_sad = max(best_acc_sad, test_acc[2])
            best_acc_angry = max(best_acc_angry, test_acc[3])
            best_f1_neutral = max(best_f1_neutral, test_f1[0])
            best_f1_happy = max(best_f1_happy, test_f1[1])
            best_f1_sad = max(best_f1_sad, test_f1[2])
            best_f1_angry = max(best_f1_angry, test_f1[3])
            test_acc = (best_acc_neutral + best_acc_happy + best_acc_sad + best_acc_angry + best_f1_neutral + best_f1_happy + best_f1_sad + best_f1_angry) / 8
            is_best = test_acc > best_acc
            #if is_best:
            #    best_epoch = epoch
            #    best_acc = max(test_acc, best_acc)
            #    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            #            and args.rank % ngpus_per_node == 0):
            #        save_checkpoint({
            #            'epoch': epoch + 1,
            #            'arch': args.arch,
            #            'state_dict': model.state_dict(),
            #            'best_acc': best_acc,
            #            'optimizer' : optimizer.state_dict(),
            #        }, is_best, "iemocap_checkpoint.pth.tar", args.dataset)

        elif args.dataset == "mosei_emo":
            train_acc, train_f1 = eval_mosei_emo(train_results, train_truth)
            valid_acc, valid_f1 = eval_mosei_emo(valid_results, valid_truth)
            test_acc, test_f1 = eval_mosei_emo(test_results, test_truth)
            print('Epoch {:2d} Acc | Test 1 {:5.4f} | Test 2 {:5.4f} | Test 3 {:5.4f} | Test 4 {:5.4f} | Test 5 {:5.4f} | Test 6 {:5.4f}'\
                    .format(epoch, test_acc[0], test_acc[1], test_acc[2], test_acc[3], test_acc[4], test_acc[5]))
            #print('Epoch {:2d} F1  | Train Neutral {:5.4f} | Train Happy {:5.4f} | Train Sad {:5.4f} | Train Angry {:5.4f}'.format(epoch, train_f1[0], train_f1[1], train_f1[2], train_f1[3]))
            #print('Epoch {:2d} F1  | Valid Neutral {:5.4f} | Valid Happy {:5.4f} | Valid Sad {:5.4f} | Valid Angry {:5.4f}'.format(epoch, valid_f1[0], valid_f1[1], valid_f1[2], valid_f1[3]))
            print('Epoch {:2d} F1  | Test 1 {:5.4f} | Test 2 {:5.4f} | Test 3 {:5.4f} | Test 4 {:5.4f} | Test 5 {:5.4f} | Test 6 {:5.4f}'\
                    .format(epoch, test_f1[0], test_f1[1], test_f1[2], test_f1[3], test_f1[4], test_f1[5]))
            print('-' * 50)
            test_acc = (sum(test_acc) + sum(test_f1)) / 12
            is_best = test_acc > best_acc
            #if is_best:
            #    best_epoch = epoch
            #    best_acc = max(test_acc, best_acc)
            #    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            #            and args.rank % ngpus_per_node == 0):
            #        save_checkpoint({
            #            'epoch': epoch + 1,
            #            'arch': args.arch,
            #            'state_dict': model.state_dict(),
            #            'best_acc': best_acc,
            #            'optimizer' : optimizer.state_dict(),
            #        }, is_best, "mosei_emo_checkpoint_{}.pth.tar".format(random.random()), args.dataset)
        else:
            print("evalutation for other datasets coming soon")
            assert False

    if args.dataset == "mosei_senti":
        print("Best Test Acc {:5.4f} | at Epoch {:2d}".format(best_acc, best_epoch))
    elif args.dataset == "iemocap":
        print("Best Acc: Neutral{:5.4f}, Happy{:5.4f}, Sad{:5.4f}, Angry{:5.4f})".format(best_acc_neutral, best_acc_happy, best_acc_sad, best_acc_angry))
        print("Best F1 : Neutral{:5.4f}, Happy{:5.4f}, Sad{:5.4f}, Angry{:5.4f})".format(best_f1_neutral, best_f1_happy, best_f1_sad, best_f1_angry))
    else:
        assert False


"""Train the network
"""
def train(train_loader, model, criterion, optimizer, epoch, args):
    results = []
    truths = []
    results_weighted = []
    truths_rounded = []
    # switch to train mode
    model.train()
    total_loss = 0.0
    total_batch_size = 0

    for ind, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
        # measure data loading time

        if args.gpu is not None:
            sample_ind, text, audio, vision = batch_X
            text, audio, vision = text.cuda(args.gpu, non_blocking=True), \
            audio.cuda(args.gpu, non_blocking=True), vision.cuda(args.gpu, non_blocking=True)
        batch_Y = batch_Y.cuda(args.gpu, non_blocking=True)
        eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
        batch_size = text.size(0)
        total_batch_size += batch_size

        # compute output
        combined_loss = 0
        preds, _, _ = model(text, audio, vision)
        if args.dataset == "iemocap":
            preds = preds.reshape(-1, 2)
            eval_attr = eval_attr.long().reshape(-1)
            raw_loss = criterion(preds, eval_attr)
            results.append(preds)
            truths.append(eval_attr)
        elif args.dataset == "mosei_senti":
            eval_attr_original = eval_attr.clone().detach()
            eval_attr = torch.squeeze((torch.round(eval_attr) + 3).type("torch.cuda.LongTensor"))
            preds_in_digit = torch.argmax(preds, 1) - 3
            raw_loss = criterion(preds, eval_attr)
            truths.append(eval_attr_original)
            results.append(preds_in_digit)
            results_weighted.append(-3 * preds[:,0] + (-2) * preds[:,1] + (-1) * preds[:,2] + 
                    1 * preds[:,4] + 2 * preds[:,5] + 3 * preds[:,6])
            truths_rounded.append(torch.round(eval_attr_original))

        elif args.dataset == "mosei_emo":
            emotion_cutoff = 0 # target value greater than this will be labeled as 1, else 0
            sigmoid_cutoff = 0.5 # prediction value greater than this will be labeled as 1, else 0
            # For each emotion, if >=1 we label the emotion as exists (1), else 0
            eval_attr = (eval_attr > emotion_cutoff).type(torch.FloatTensor).to(args.gpu)
            raw_loss = criterion(preds, eval_attr)
            preds_in_digit = (torch.sigmoid(preds) > sigmoid_cutoff).type(torch.FloatTensor).to(args.gpu)
            results.append(preds_in_digit)
            truths.append(eval_attr)
        total_loss += raw_loss.item() * batch_size
        combined_loss = raw_loss
        optimizer.zero_grad()
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        
    avg_loss = total_loss / total_batch_size
    results = torch.cat(results)
    truths = torch.cat(truths)
    if args.dataset == "mosei_senti":
        results_weighted = torch.cat(results_weighted)
        truths_rounded = torch.cat(truths_rounded)
    return avg_loss, results, truths, results_weighted, truths_rounded


"""Validate the network
"""

def validate(loader, model, criterion, args, ctc_a2l_module=None):
    if args.dataset == 'mosei_senti':
        a_matrix_all = torch.zeros(1, 7)
        r_matrix_all = torch.zeros(1, 7, args.output_dim)
        beta_matrix_all = torch.zeros(1, 7, args.output_dim)
        meta_info = [] 
    elif args.dataset == "iemocap":
        a_matrix_all = torch.zeros(1, 7)
        r_matrix_all = torch.zeros(1, 7, args.output_dim)
        beta_matrix_all = torch.zeros(1, 7, args.output_dim)
        correct_matrix = torch.zeros(1, 1) # record whether a prediction is correct or not
    elif args.dataset == "mosei_emo":
        a_matrix_all = torch.zeros(1, 7)
        r_matrix_all = torch.zeros(1, 7, args.output_dim)
        beta_matrix_all = torch.zeros(1, 7, args.output_dim)
        meta_info = [] 
    else:
        raise NotImplementedError
    # switch to evaluate mode
    model.eval()
    results = []
    truths = []
    results_weighted = []
    truths_rounded = []
    total_loss = 0.0
    total_batch_size = 0
    with torch.no_grad():
        for ind, (batch_X, batch_Y, batch_META) in enumerate(loader):
            if args.gpu is not None:
                sample_ind, text, audio, vision = batch_X
                text, audio, vision = text.cuda(args.gpu, non_blocking=True), \
                audio.cuda(args.gpu, non_blocking=True), vision.cuda(args.gpu, non_blocking=True)
            batch_Y = batch_Y.cuda(args.gpu, non_blocking=True)
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
            batch_size = text.size(0)
            total_batch_size += batch_size
            if args.arch == "multimodal_capsule":
                preds, activation, routing_coefficient = model(text, audio, vision)
                beta = torch.einsum('bn, bnm->bnm', activation, routing_coefficient)
            else:
                preds, _, _ = model(text, audio, vision)
            if args.dataset == 'iemocap':
                preds = preds.reshape(-1, 2)
                eval_attr = eval_attr.long().reshape(-1)
                raw_loss = criterion(preds, eval_attr)
                results.append(preds)
                truths.append(eval_attr)
            elif args.dataset == "mosei_senti":
                eval_attr_original = eval_attr.clone().detach()
                eval_attr = torch.squeeze((torch.round(eval_attr) + 3).type("torch.cuda.LongTensor"))
                preds_in_digit = torch.argmax(preds, 1) - 3
                raw_loss = criterion(preds, eval_attr)
                truths.append(eval_attr_original)
                results.append(preds_in_digit)
                results_weighted.append(-3 * preds[:,0] + (-2) * preds[:,1] + (-1) * preds[:,2] + 
                        1 * preds[:,4] + 2 * preds[:,5] + 3 * preds[:,6])
                truths_rounded.append(torch.round(eval_attr_original))
                #if args.arch == "multimodal_capsule" and args.evaluate == True:
                    #for b in range(beta.shape[0]):
                    #    beta_matrix_all = torch.cat((beta_matrix_all, beta[b].unsqueeze(0)), dim=0)
                    #    a_matrix_all = torch.cat((a_matrix_all, activation[b].unsqueeze(0)), dim=0)
                    #    r_matrix_all = torch.cat((r_matrix_all, routing_coefficient[b].unsqueeze(0)), dim=0)
                        #correct_matrix = torch.cat((correct_matrix, torch.tensor(float(preds_in_digit[b] + 3 == eval_attr[b])).view(1, 1)), dim=0)
                    #    meta_info.append(batch_META[0][b])
            elif args.dataset == "mosei_emo":
                emotion_cutoff = 0 # target value greater than this will be labeled as 1, else 0
                sigmoid_cutoff = 0.5 # prediction value greater than this will be labeled as 1, else 0
                # For each emotion, if >=1 we label the emotion as exists (1), else 0
                eval_attr = (eval_attr > emotion_cutoff).type(torch.FloatTensor).to(args.gpu)
                raw_loss = criterion(preds, eval_attr)
                preds_in_digit = (torch.sigmoid(preds) > sigmoid_cutoff).type(torch.FloatTensor).to(args.gpu)
                results.append(preds_in_digit)
                truths.append(eval_attr)
                #if args.arch == "multimodal_capsule" and args.evaluate == True:
                #    for b in range(beta.shape[0]):
                #        beta_matrix_all = torch.cat((beta_matrix_all, beta[b].unsqueeze(0)), dim=0)
                #        a_matrix_all = torch.cat((a_matrix_all, activation[b].unsqueeze(0)), dim=0)
                #        r_matrix_all = torch.cat((r_matrix_all, routing_coefficient[b].unsqueeze(0)), dim=0)
                #        meta_info.append(batch_META)
                    #correct_matrix = torch.cat((correct_matrix, torch.tensor(float(preds_in_digit[b] + 3 == eval_attr[b])).view(1, 1)), dim=0)
            total_loss += raw_loss.item() * batch_size
    avg_loss = total_loss / total_batch_size
    results = torch.cat(results)
    truths = torch.cat(truths)
    if args.dataset == "mosei_senti":
        results_weighted = torch.cat(results_weighted)
        truths_rounded = torch.cat(truths_rounded)
    if args.dataset == "iemocap":
        pass
    else:
        pass
        #if args.arch == "multimodal_capsule" and args.evaluate == True:
            #beta_matrix_all = beta_matrix_all[1:]
            #a_matrix_all = a_matrix_all[1:]
            #r_matrix_all = r_matrix_all[1:]
            ##correct_matrix = correct_matrix[1:]

            #np.save("beta_matrix_all_{}.npy".format(args.dataset), beta_matrix_all.detach().cpu().numpy())
            #np.save("a_matrix_all_{}.npy".format(args.dataset), a_matrix_all.detach().cpu().numpy())
            #np.save("r_matrix_all_{}.npy".format(args.dataset), r_matrix_all.detach().cpu().numpy())
            ##np.save("correct_matrix_{}.npy".format(args.dataset), correct_matrix.detach().cpu().numpy())
            #filename = "meta_info_{}.pkl".format(args.dataset)
            #with open(filename, "wb") as f:
            #    pickle.dump(meta_info, f)
    return avg_loss, results, truths, results_weighted, truths_rounded


def save_checkpoint(state, is_best, filename, dataset):
    torch.save(state, filename)
    if is_best:
        if dataset == "iemocap":
            shutil.copyfile(filename, 'iemocap_model_best_{}.pth.tar'.format(filename))
        elif dataset == "mosei_senti":
            shutil.copyfile(filename, 'mosei_senti_model_best_{}.pth.tar'.format(filename))
        elif dataset == "mosei_emo":
            shutil.copyfile(filename, 'mosei_emo_model_best_{}.pth.tar'.format(filename))
        else:
            assert False


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
