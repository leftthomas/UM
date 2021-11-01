import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataset import VideoDataset, LocalizationEvaluation
from model import Model


def test(network, config, data_loader, metric_info):
    with torch.no_grad():
        network.load_state_dict(torch.load(config.model_file, 'cpu'))
        network = network.cuda()
        network.eval()

        results, num_correct, num_total = {}, 0, 0
        for feat, label, video_name, num_seg, annotation in tqdm(data_loader):
            feat, label, video_name = feat.cuda(), label.cuda(), video_name[0]
            num_seg, num_segments = num_seg.item(), feat.shape[1]
            score_act, _, feat_act, feat_bkg, features, cas_softmax = network(feat)

            feat_magnitudes_act = torch.mean(torch.norm(feat_act, dim=2), dim=1)
            feat_magnitudes_bkg = torch.mean(torch.norm(feat_bkg, dim=2), dim=1)

            label_np = label.detach().cpu().numpy()
            score_np = score_act[0].detach().cpu().numpy()

            pred_np = np.zeros_like(score_np)
            pred_np[np.where(score_np < config.class_th)] = 0
            pred_np[np.where(score_np >= config.class_th)] = 1

            correct_pred = np.sum(label_np == pred_np, axis=1)

            num_correct += np.sum((correct_pred == len(data_loader.dataset.class_name_to_idx)).astype(np.float32))
            num_total += correct_pred.shape[0]

            feat_magnitudes = torch.norm(features, p=2, dim=2)

            feat_magnitudes = utils.minmax_norm(feat_magnitudes, max_val=feat_magnitudes_act,
                                                min_val=feat_magnitudes_bkg)
            feat_magnitudes = feat_magnitudes.repeat((len(data_loader.dataset.class_name_to_idx), 1, 1)).permute(1, 2,
                                                                                                                 0)

            cas = utils.minmax_norm(cas_softmax * feat_magnitudes)

            pred = np.where(score_np >= config.class_th)[0]

            if len(pred) == 0:
                pred = np.array([np.argmax(score_np)])

            cas_pred = cas[0].cpu().numpy()[:, pred]
            cas_pred = np.reshape(cas_pred, (num_segments, -1, 1))

            cas_pred = utils.upgrade_resolution(cas_pred, config.scale)

            proposal_dict = {}

            feat_magnitudes_np = feat_magnitudes[0].detach().cpu().numpy()[:, pred]
            feat_magnitudes_np = np.reshape(feat_magnitudes_np, (num_segments, -1, 1))
            feat_magnitudes_np = utils.upgrade_resolution(feat_magnitudes_np, config.scale)

            for i in range(len(config.seg_th)):
                cas_temp = cas_pred.copy()

                zero_location = np.where(cas_temp[:, :, 0] < config.seg_th[i])
                cas_temp[zero_location] = 0

                seg_list = []
                for c in range(len(pred)):
                    pos = np.where(cas_temp[:, c, 0] > 0)
                    seg_list.append(pos)

                proposals = utils.get_proposal_oic(seg_list, cas_temp, score_np, pred, config.scale, num_seg,
                                                   config.fps, num_segments)

                for j in range(len(proposals)):
                    class_id = proposals[j][0][0]

                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []

                    proposal_dict[class_id] += proposals[j]

            for i in range(len(config.act_thresh_magnitudes)):
                cas_temp = cas_pred.copy()

                feat_magnitudes_np_temp = feat_magnitudes_np.copy()

                zero_location = np.where(feat_magnitudes_np_temp[:, :, 0] < config.act_thresh_magnitudes[i])
                feat_magnitudes_np_temp[zero_location] = 0

                seg_list = []
                for c in range(len(pred)):
                    pos = np.where(feat_magnitudes_np_temp[:, c, 0] > 0)
                    seg_list.append(pos)

                proposals = utils.get_proposal_oic(seg_list, cas_temp, score_np, pred, config.scale,
                                                   num_seg, config.fps, num_segments)

                for j in range(len(proposals)):
                    class_id = proposals[j][0][0]

                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []

                    proposal_dict[class_id] += proposals[j]

            final_proposals = []
            for class_id in proposal_dict.keys():
                final_proposals.append(utils.nms(proposal_dict[class_id], 0.6))

            results[video_name] = utils.result2json(final_proposals, data_loader.dataset.idx_to_class_name)

        test_acc = num_correct / num_total

        gt_path = os.path.join(config.save_path, '{}_gt.json'.format(config.data_name))
        with open(gt_path, 'w') as f:
            json.dump(data_loader.dataset.annotations, f)
        pred_path = os.path.join(config.save_path, '{}_pred.json'.format(config.data_name))
        with open(pred_path, 'w') as f:
            json.dump(results, f)

        iou_thresh = np.linspace(0.1, 0.7, 7)
        anet_detection = LocalizationEvaluation(gt_path, pred_path, tiou_thresholds=iou_thresh, verbose=False)
        mAP, average_mAP = anet_detection.evaluate()

        print('Test accuracy: {}'.format(test_acc))

        for i in range(iou_thresh.shape[0]):
            print('mAP@{:.1f}: {}'.format(iou_thresh[i], mAP[i]))

        print('mAP@AVG: {}'.format(average_mAP))

        metric_info['test_acc'].append(test_acc)
        metric_info['mAP@AVG'].append(average_mAP)

        for i in range(iou_thresh.shape[0]):
            metric_info['mAP@{:.1f}'.format(iou_thresh[i])].append(mAP[i])


if __name__ == "__main__":
    args = utils.parse_args()
    test_loader = DataLoader(VideoDataset(args.data_path, args.data_name, 'test', args.num_segments), batch_size=1,
                             shuffle=False, num_workers=args.num_workers, worker_init_fn=args.worker_init_fn)

    net = Model(args.r_act, args.r_bkg, len(test_loader.dataset.class_name_to_idx))

    if args.data_name == 'thumos14':
        test_info = {'test_acc': [], 'mAP@AVG': [], 'mAP@0.1': [], 'mAP@0.2': [], 'mAP@0.3': [],
                     'mAP@0.4': [], 'mAP@0.5': [], 'mAP@0.6': [], 'mAP@0.7': []}
    else:
        test_info = {'test_acc': [], 'mAP@AVG': [], 'mAP@0.5': [], 'mAP@0.75': [], 'mAP@0.95': []}

    test(net, args, test_loader, test_info)
    utils.save_best_record_thumos(test_info, os.path.join(args.save_path, 'best_record.txt'))
