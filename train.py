import os

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataset import VideoDataset
from model import Model
from test import test_loop


class UMLoss(nn.Module):
    def __init__(self, magnitude):
        super(UMLoss, self).__init__()
        self.magnitude = magnitude

    def forward(self, feat_act, feat_bkg):
        loss_act = torch.relu(self.magnitude - torch.norm(torch.mean(feat_act, dim=-1), dim=-1))
        loss_bkg = torch.norm(torch.mean(feat_bkg, dim=-1), dim=-1)
        loss_um = torch.mean((loss_act + loss_bkg) ** 2)
        return loss_um


def train_loop(network, data_loader, train_optimizer, n_iter):
    network.train()
    data, label = next(data_loader)
    data, label = data.cuda(), label.cuda()
    label_act = label / torch.sum(label, dim=-1, keepdim=True)
    label_bkg = torch.ones_like(label)
    label_bkg /= torch.sum(label_bkg, dim=-1, keepdim=True)

    train_optimizer.zero_grad()
    score_act, score_bkg, _, feat_act, feat_bkg, _ = network(data)
    cls_loss = bce_criterion(score_act, label_act)
    um_loss = um_criterion(feat_act, feat_bkg)
    be_loss = bce_criterion(score_bkg, label_bkg)
    loss = cls_loss + args.alpha * um_loss + args.beta * be_loss
    loss.backward()
    train_optimizer.step()

    print('Train Step: [{}/{}] Total Loss: {:.4f} CLS Loss: {:.4f} UM Loss: {:.4f} BE Loss: {:.4f}'
          .format(n_iter, args.num_iters, loss.item(), cls_loss.item(), um_loss.item(), be_loss.item()))


if __name__ == "__main__":
    args, test_info = utils.parse_args()
    train_loader = DataLoader(VideoDataset(args.data_path, args.data_name, 'train', args.num_segments),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              worker_init_fn=args.worker_init_fn)
    test_loader = DataLoader(VideoDataset(args.data_path, args.data_name, 'test', args.num_segments), batch_size=1,
                             shuffle=False, num_workers=args.num_workers, worker_init_fn=args.worker_init_fn)

    net = Model(args.r_act, args.r_bkg, len(train_loader.dataset.class_name_to_idx)).cuda()

    best_mAP, um_criterion, bce_criterion = -1, UMLoss(args.magnitude), nn.BCELoss()
    optimizer = Adam(net.parameters(), lr=args.lr, weight_decay=args.decay)
    desc_bar = tqdm(range(1, args.num_iters + 1), total=args.num_iters, dynamic_ncols=True)

    for step in desc_bar:
        if (step - 1) % len(train_loader) == 0:
            loader_iter = iter(train_loader)

        train_loop(net, loader_iter, optimizer, step)
        test_loop(net, args, test_loader, test_info, step)

        if test_info['mAP@AVG'][-1] > best_mAP:
            best_mAP = test_info['mAP@AVG'][-1]
            utils.save_best_record(test_info, os.path.join(args.save_path, '{}_record.txt'.format(args.data_name)))
            torch.save(net.state_dict(), os.path.join(args.model_path, '{}_model.pth'.format(args.data_name)))
