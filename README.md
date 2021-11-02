# UM

A PyTorch implementation of UM based on AAAI 2021 paper
[Weakly-supervised Temporal Action Localization by Uncertainty Modeling](https://arxiv.org/abs/2006.07006).

![Network Architecture](result/structure.png)

## Requirements

- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)

```
conda install pytorch=1.10.0 torchvision cudatoolkit=11.3 -c pytorch
```

- [MMAction2](https://mmaction2.readthedocs.io)

```
pip install git+https://github.com/open-mmlab/mim.git
mim install mmaction2
```

## Dataset

[THUMOS 14](http://crcv.ucf.edu/THUMOS14/download.html) and
[ActivityNet](http://activity-net.org/download.html) datasets are used in this repo, you could download these datasets
from official websites. The I3D features of `THUMOS 14` dataset can be downloaded from
[Google Drive](https://drive.google.com/file/d/1NqaDRo782bGZKo662I0rI_cvpDT67VQU/view), I3D features
of `ActivityNet 1.2` dataset can be downloaded from
[OneDrive](https://emailucr-my.sharepoint.com/personal/sujoy_paul_email_ucr_edu/_layouts/15/onedrive.aspx?originalPath=aHR0cHM6Ly9lbWFpbHVjci1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC9zdWpveV9wYXVsX2VtYWlsX3Vjcl9lZHUvRXMxemJIUVk0UHhLaFVrZGd2V0h0VTBCSy1feXVnYVNqWEs4NGtXc0IwWEQwdz9ydGltZT1vVlREWlhLUjJVZw&id=%2Fpersonal%2Fsujoy%5Fpaul%5Femail%5Fucr%5Fedu%2FDocuments%2Fwtalc%2Dfeatures)
, I3D features of `ActivityNet 1.3` dataset can be downloaded
from [Google Drive](https://drive.google.com/drive/folders/1W2t4UKUkV_9duAsAFWU0HHYWbav2CZXp). The data directory
structure is shown as follows:

 ```
├── thumos14                                    |  ├── activitynet
   ├── features                                  |    ├── features_1.2
       ├── val                                   |        ├── train 
           ├── flow                              |            ├── flow    
               ├── video_validation_0000051.npy  |                ├── v___dXUJsj3yo.npy
               └── ...                           |                └── ...
           ├── rgb (same structure as flow)      |            ├── rgb
       ├── test                                  |                ├── v___dXUJsj3yo.npy
           ├── flow                              |                └── ...
               ├── video_test_0000004.npy        |        ├── val (same structure as tain)
               └── ...                           |    ├── features_1.3 (same structure as features_1.2)
           ├── rgb (same structure as flow)      |    ├── videos
   ├── videos                                    |        ├── train
       ├── val                                   |            ├── v___c8enCfzqw.mp4
           ├── video_validation_0000051.mp4      |            └──... 
           └──...                                |         ├── val           
       ├── test                                  |            ├── v__1vYKA7mNLI.mp4
           ├──video_test_0000004.mp4             |            └──...   
           └──...                                | annotations_1.2.json
   annotations.json                              | annotations_1.3.json
```

## Usage

You can easily train and test the model by running the script below. If you want to try other options, please refer to
`utils.py`.
### Train Model

```
python train.py --data_name activitynet1.2 --num_segments 50 --seed 0
```

### Test Model

```
python test.py --model_file --data_name thumos14 --model_file result/thumos14_model.pth
```

## Benchmarks

The models are trained on one NVIDIA GeForce GTX 1080 Ti GPU (11G). All the hyper-parameters are the default values
according to the papers.

<table>
<thead>
  <tr>
    <th rowspan="3">Backbone</th>
    <th rowspan="3">Dim</th>
    <th colspan="4">Sketchy Extended</th>
    <th colspan="4">TU-Berlin Extended</th>
    <th rowspan="3">Download</th>
  </tr>
  <tr>
    <td align="center">mAP@200</td>
    <td align="center">mAP@all</td>
    <td align="center">P@100</td>
    <td align="center">P@200</td>
    <td align="center">mAP@200</td>
    <td align="center">mAP@all</td>
    <td align="center">P@100</td>
    <td align="center">P@200</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">VGG16</td>
    <td align="center">64</td>
    <td align="center">53.0</td>
    <td align="center">38.0</td>
    <td align="center">50.1</td>
    <td align="center">46.0</td>
    <td align="center">54.7</td>
    <td align="center">37.4</td>
    <td align="center">52.2</td>
    <td align="center">49.4</td>
    <td align="center"><a href="https://pan.baidu.com/s/14lJMIRCMJIIM4QrP_Gbqfg">e8db</a></td>
  </tr>
  <tr>
    <td align="center">VGG16</td>
    <td align="center">512</td>
    <td align="center">57.5</td>
    <td align="center">42.6</td>
    <td align="center">54.6</td>
    <td align="center">50.6</td>
    <td align="center">62.3</td>
    <td align="center">44.6</td>
    <td align="center">60.1</td>
    <td align="center">57.1</td>
    <td align="center"><a href="https://pan.baidu.com/s/1rdyX8S4J7hHrDk33QHip1A">uiv4</a></td>
  </tr>
  <tr>
    <td align="center">VGG16</td>
    <td align="center">4096</td>
    <td align="center">58.6</td>
    <td align="center">44.4</td>
    <td align="center">56.0</td>
    <td align="center">51.9</td>
    <td align="center">64.3</td>
    <td align="center">47.6</td>
    <td align="center">62.5</td>
    <td align="center">59.7</td>
    <td align="center"><a href="https://pan.baidu.com/s/1z30aDG-ra0owr2P59SnpZA">mb9f</a></td>
  </tr>
  <tr>
    <td align="center">ResNet50</td>
    <td align="center">128</td>
    <td align="center">62.6</td>
    <td align="center">48.7</td>
    <td align="center">60.4</td>
    <td align="center">56.4</td>
    <td align="center">61.2</td>
    <td align="center">46.2</td>
    <td align="center">59.4</td>
    <td align="center">57.6</td>
    <td align="center"><a href="https://pan.baidu.com/s/1aK2xiSPZRPXuORoH-8-aoQ">c7h4</a></td>
  </tr>
  <tr>
    <td align="center">ResNet50</td>
    <td align="center">512</td>
    <td align="center">66.2</td>
    <td align="center">53.2</td>
    <td align="center">63.9</td>
    <td align="center">60.1</td>
    <td align="center">64.8</td>
    <td align="center">50.3</td>
    <td align="center">63.0</td>
    <td align="center">61.1</td>
    <td align="center"><a href="https://pan.baidu.com/s/1N7iYhbj6GBQ3byRPoekFHA">mhmm</a></td>
  </tr>
  <tr>
    <td align="center">ResNet50</td>
    <td align="center">2048</td>
    <td align="center">66.6</td>
    <td align="center">53.6</td>
    <td align="center">64.5</td>
    <td align="center">60.7</td>
    <td align="center">65.6</td>
    <td align="center">53.7</td>
    <td align="center">64.2</td>
    <td align="center">62.7</td>
    <td align="center"><a href="https://pan.baidu.com/s/1unfflapyOiRvqEbYMZH-gg">5vcy</a></td>
  </tr>
</tbody>
</table>

