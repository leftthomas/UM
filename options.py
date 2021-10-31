import argparse
import os
import random

import numpy as np
import torch
from torch.backends import cudnn


def parse_args():
    description = 'Pytorch Implementation of \'Weakly-supervised Temporal Action Localization by Uncertainty Modeling\''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_path', type=str, default='/data')
    parser.add_argument('--save_path', type=str, default='result')
    parser.add_argument('--alpha', type=float, default=0.0005)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--magnitude', type=int, default=100)
    parser.add_argument('--r_act', type=float, default=9)
    parser.add_argument('--r_bkg', type=float, default=4)
    parser.add_argument('--class_th', type=float, default=0.2, help='threshold for class score')
    parser.add_argument('--data_name', type=str, default='thumos14',
                        choices=['thumos14', 'activitynet1.2', 'activitynet1.3'])
    parser.add_argument('--num_segments', type=int, default=750)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_iters', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 for no manual seed)')
    parser.add_argument('--model_file', type=str, default=None, help='the path of pre-trained model file')

    return init_args(parser.parse_args())


def init_args(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        args.worker_init_fn = np.random.seed(args.seed)
    else:
        args.worker_init_fn = None

    args.seg_th = np.arange(0.0, 0.25, 0.025)
    return vars(args)
