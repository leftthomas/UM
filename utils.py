import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
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

    args.scale = 24
    args.fps = 25
    args.seg_th = np.arange(0.0, 0.25, 0.025)
    args.act_thresh_magnitudes = np.arange(0.4, 0.625, 0.025)
    return args


def upgrade_resolution(arr, scale):
    x = np.arange(0, arr.shape[0])
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale


def get_proposal_oic(tList, wtcam, final_score, c_pred, scale, v_len, sampling_frames, num_segments, _lambda=0.25,
                     gamma=0.2):
    t_factor = (16 * v_len) / (scale * num_segments * sampling_frames)
    temp = []
    for i in range(len(tList)):
        c_temp = []
        temp_list = np.array(tList[i])[0]
        if temp_list.any():
            grouped_temp_list = grouping(temp_list)
            for j in range(len(grouped_temp_list)):
                if len(grouped_temp_list[j]) < 2:
                    continue

                inner_score = np.mean(wtcam[grouped_temp_list[j], i, 0])

                len_proposal = len(grouped_temp_list[j])
                outer_s = max(0, int(grouped_temp_list[j][0] - _lambda * len_proposal))
                outer_e = min(int(wtcam.shape[0] - 1), int(grouped_temp_list[j][-1] + _lambda * len_proposal))

                outer_temp_list = list(range(outer_s, int(grouped_temp_list[j][0]))) + list(
                    range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))

                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(wtcam[outer_temp_list, i, 0])

                c_score = inner_score - outer_score + gamma * final_score[c_pred[i]]
                t_start = grouped_temp_list[j][0] * t_factor
                t_end = (grouped_temp_list[j][-1] + 1) * t_factor
                c_temp.append([c_pred[i], c_score, t_start, t_end])
            temp.append(c_temp)
    return temp


def result2json(result, class_dict):
    result_file = []
    for i in range(len(result)):
        for j in range(len(result[i])):
            line = {'label': class_dict[result[i][j][0]], 'score': result[i][j][1],
                    'segment': [result[i][j][2], result[i][j][3]]}
            result_file.append(line)
    return result_file


def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


def save_best_record(test_info, file_path):
    with open(file_path, 'w') as fo:
        fo.write("Test Acc: {:.3f}\n".format(test_info["test_acc"][-1]))
        fo.write("mAP@AVG: {:.3f}\n".format(test_info["mAP@AVG"][-1]))
        tIoU_thresh = np.linspace(0.1, 0.7, 7)
        for i in range(len(tIoU_thresh)):
            fo.write("mAP@{:.1f}: {:.3f}\n".format(tIoU_thresh[i], test_info["mAP@{:.1f}".format(tIoU_thresh[i])][-1]))


def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = nn.ReLU()
        max_val = relu(torch.max(act_map, dim=1)[0])
        min_val = relu(torch.min(act_map, dim=1)[0])

    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta

    ret[ret > 1] = 1
    ret[ret < 0] = 0

    return ret



