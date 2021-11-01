from tensorboard_logger import Logger

from test import *
from utils import *


class UM_loss(nn.Module):
    def __init__(self, alpha, beta, margin):
        super(UM_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.ce_criterion = nn.BCELoss()

    def forward(self, score_act, score_bkg, feat_act, feat_bkg, label):
        loss = {}

        label = label / torch.sum(label, dim=1, keepdim=True)

        loss_cls = self.ce_criterion(score_act, label)

        label_bkg = torch.ones_like(label).cuda()
        label_bkg /= torch.sum(label_bkg, dim=1, keepdim=True)
        loss_be = self.ce_criterion(score_bkg, label_bkg)

        loss_act = self.margin - torch.norm(torch.mean(feat_act, dim=1), p=2, dim=1)
        loss_act[loss_act < 0] = 0
        loss_bkg = torch.norm(torch.mean(feat_bkg, dim=1), p=2, dim=1)

        loss_um = torch.mean((loss_act + loss_bkg) ** 2)

        loss_total = loss_cls + self.alpha * loss_um + self.beta * loss_be

        loss["loss_cls"] = loss_cls
        loss["loss_be"] = loss_be
        loss["loss_um"] = loss_um
        loss["loss_total"] = loss_total

        return loss_total, loss


def train(net, loader_iter, optimizer, criterion, logger, step):
    net.train()

    _data, _label, _, _, _ = next(loader_iter)

    _data = _data.cuda()
    _label = _label.cuda()

    optimizer.zero_grad()

    score_act, score_bkg, feat_act, feat_bkg, _, _ = net(_data)

    cost, loss = criterion(score_act, score_bkg, feat_act, feat_bkg, _label)

    cost.backward()
    optimizer.step()

    for key in loss.keys():
        logger.log_value(key, loss[key].cpu().item(), step)


if __name__ == "__main__":
    args = parse_args()

    config = Config(args)
    worker_init_fn = None

    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)

    utils.save_config(config, os.path.join(config.output_path, "config.txt"))

    net = Model(config.len_feature, config.num_classes, config.r_act, config.r_bkg)
    net = net.cuda()

    train_loader = data.DataLoader(
        VideoDataset(data_path=config.data_path, mode='train',
                     modal=config.modal, fps=config.feature_fps,
                     num_segments=config.num_segments, supervision='weak',
                     seed=config.seed, sampling='random'),
        batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        worker_init_fn=worker_init_fn)

    test_loader = data.DataLoader(
        VideoDataset(data_path=config.data_path, mode='test',
                     modal=config.modal, fps=config.feature_fps,
                     num_segments=config.num_segments, supervision='weak',
                     seed=config.seed, sampling='uniform'),
        batch_size=1,
        shuffle=False, num_workers=config.num_workers,
        worker_init_fn=worker_init_fn)

    test_info = {"step": [], "test_acc": [], "average_mAP": [],
                 "mAP@0.1": [], "mAP@0.2": [], "mAP@0.3": [],
                 "mAP@0.4": [], "mAP@0.5": [], "mAP@0.6": [], "mAP@0.7": []}

    best_mAP = -1

    criterion = UM_loss(config.alpha, config.beta, config.margin)

    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr[0],
                                 betas=(0.9, 0.999), weight_decay=0.0005)

    logger = Logger(config.log_path)

    for step in tqdm(
            range(1, config.num_iters + 1),
            total=config.num_iters,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(train_loader) == 0:
            loader_iter = iter(train_loader)

        train(net, loader_iter, optimizer, criterion, logger, step)

        test(net, config, logger, test_loader, test_info, step)

        if test_info["average_mAP"][-1] > best_mAP:
            best_mAP = test_info["average_mAP"][-1]

            utils.save_best_record(test_info,
                                   os.path.join(config.output_path,
                                                "best_record_seed_{}.txt".format(config.seed)))

            torch.save(net.state_dict(), os.path.join(args.model_path, \
                                                      "model_seed_{}.pkl".format(config.seed)))
