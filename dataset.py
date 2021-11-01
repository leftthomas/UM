import glob
import json
import os

import numpy as np
import torch
from mmaction.core.evaluation import ActivityNetLocalization
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, data_path, data_name, mode, num_segments):

        self.data_name, self.mode, self.num_segments = data_name, mode, num_segments

        # prepare features
        if data_name == 'thumos14':
            mode = 'val' if mode == 'train' else 'test'
            self.rgb = glob.glob(os.path.join(data_path, data_name, 'features', mode, 'rgb', '*'))
            self.flow = glob.glob(os.path.join(data_path, data_name, 'features', mode, 'flow', '*'))
            with open(os.path.join(data_path, data_name, 'annotations.json')) as f:
                annotations = json.load(f)['database']
        else:
            mode = 'train' if mode == 'train' else 'val'
            data_name, suffix = data_name[:-3], data_name[-3:]
            self.rgb = glob.glob(os.path.join(data_path, data_name, 'features_{}'.format(suffix), mode, 'rgb', '*'))
            self.flow = glob.glob(os.path.join(data_path, data_name, 'features_{}'.format(suffix), mode, 'flow', '*'))
            with open(os.path.join(data_path, data_name, 'annotations_{}.json'.format(suffix))) as f:
                annotations = json.load(f)['database']

        # prepare labels
        assert len(self.rgb) == len(self.flow)
        self.annotations, classes, self.class_name_to_idx, self.idx_to_class_name = {}, set(), {}, {}
        for key in self.rgb:
            video_name = os.path.basename(key).split('.')[0]
            value = annotations[video_name]['annotations']
            self.annotations[video_name] = value
            for annotation in value:
                classes.add(annotation['label'])
        for i, key in enumerate(sorted(classes)):
            self.class_name_to_idx[key] = i
            self.idx_to_class_name[i] = key

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, index):
        rgb, flow = np.load(self.rgb[index]), np.load(self.flow[index])
        video_name, num_seg = os.path.basename(self.rgb[index]).split('.')[0], rgb.shape[0]
        annotation = self.annotations[video_name]
        sample_idx = self.random_sampling(num_seg) if self.mode == 'train' else self.uniform_sampling(num_seg)
        rgb, flow = torch.from_numpy(rgb[sample_idx]), torch.from_numpy(flow[sample_idx])

        label = torch.zeros(len(self.class_name_to_idx))
        for item in annotation:
            label[self.class_name_to_idx[item['label']]] = 1
        feat = torch.cat((rgb, flow), dim=-1)
        return feat, label, video_name, num_seg, annotation

    def random_sampling(self, length):
        if self.num_segments == length:
            return np.arange(length).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)

    def uniform_sampling(self, length):
        # because the length may different as these two line codes, make sure batch size == 1 in test mode
        if length <= self.num_segments:
            return np.arange(length).astype(int)
        else:
            return np.floor(np.arange(self.num_segments) * length / self.num_segments).astype(int)


class LocalizationEvaluation(ActivityNetLocalization):
    @staticmethod
    def _import_ground_truth(ground_truth_filename):
        """Read ground truth file and return the ground truth instances and the
        activity classes.

        Args:
            ground_truth_filename (str): Full path to the ground truth json file.

        Returns:
            tuple[list, dict]: (ground_truth, activity_index).
                ground_truth contains the ground truth instances, which is in a
                    dict format.
                activity_index contains classes index.
        """
        with open(ground_truth_filename, 'r') as f:
            data = json.load(f)
        activity_index, class_idx, ground_truth = {}, 0, []
        for video_id, video_info in data.items():
            for anno in video_info:
                if anno['label'] not in activity_index:
                    activity_index[anno['label']] = class_idx
                    class_idx += 1
                ground_truth_item = {'video-id': video_id, 'label': activity_index[anno['label']],
                                     't-start': float(anno['segment'][0]), 't-end': float(anno['segment'][1])}
                ground_truth.append(ground_truth_item)

        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Read prediction file and return the prediction instances.

        Args:
            prediction_filename (str): Full path to the prediction json file.

        Returns:
            List: List containing the prediction instances (dictionaries).
        """
        with open(prediction_filename, 'r') as f:
            data = json.load(f)
        prediction = []
        for video_id, video_info in data.items():
            for result in video_info:
                prediction_item = {'video-id': video_id, 'label': self.activity_index[result['label']],
                                   't-start': float(result['segment'][0]), 't-end': float(result['segment'][1]),
                                   'score': result['score']}
                prediction.append(prediction_item)

        return prediction
