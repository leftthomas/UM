import glob
import json
import os

import numpy as np
import torch
import torch.utils.data as data


class VideoDataset(data.Dataset):
    def __init__(self, data_path, data_name, mode, num_segments):

        self.data_name, self.num_segments = data_name, num_segments

        # prepare features
        if data_name == 'thumos14':
            self.rgb = glob.glob(os.path.join(data_path, data_name, 'features', mode, 'rgb', '*'))
            self.flow = glob.glob(os.path.join(data_path, data_name, 'features', mode, 'flow', '*'))
            self.mode = 'train' if mode == 'val' else 'test'
            with open(os.path.join(data_path, data_name, 'annotations.json')) as f:
                annotations = json.load(f)['database']
        else:
            data_name, suffix = data_name[:-3], data_name[-3:]
            self.rgb = glob.glob(os.path.join(data_path, data_name, 'features_{}'.format(suffix), mode, 'rgb', '*'))
            self.flow = glob.glob(os.path.join(data_path, data_name, 'features_{}'.format(suffix), mode, 'flow', '*'))
            self.mode = 'train' if mode == 'train' else 'test'
            with open(os.path.join(data_path, data_name, 'annotations_{}.json'.format(suffix))) as f:
                annotations = json.load(f)['database']

        # prepare labels
        assert len(self.rgb) == len(self.flow)
        self.annotations, classes, self.class_name_to_idx, self.idx_to_class_name = [], set(), {}, {}
        for key in self.rgb:
            value = annotations[os.path.basename(key).split('.')[0]]['annotations']
            self.annotations.append(value)
            for annotation in value:
                classes.add(annotation['label'])
        for i, key in enumerate(sorted(classes)):
            self.class_name_to_idx[key] = i
            self.idx_to_class_name[i] = key

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, index):
        rgb, flow, annotation = np.load(self.rgb[index]), np.load(self.flow[index]), self.annotations[index]
        num_seg = rgb.shape[0]
        sample_idx = self.random_sampling(num_seg) if self.mode == 'train' else self.uniform_sampling(num_seg)
        rgb, flow = torch.from_numpy(rgb[sample_idx]), torch.from_numpy(flow[sample_idx])

        label = torch.zeros(len(self.class_name_to_idx))
        for item in annotation:
            label[self.class_name_to_idx[item['label']]] = 1
        return rgb, flow, label, annotation

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
