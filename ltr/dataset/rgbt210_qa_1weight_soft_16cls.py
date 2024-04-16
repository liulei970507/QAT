import torch
import os
import os.path
import numpy as np
import pandas
import random
from collections import OrderedDict

from ltr.data.image_loader import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset
from ltr.admin.environment import env_settings

class RGBT210_qa_1weight_soft_16cls(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None):
        self.root = env_settings().rgbt210_dir if root is None else root
        super().__init__('RGBT210_qa_1weight_soft_16cls', root, image_loader)

        # video_name for each sequence
        self.sequence_list = os.listdir(self.root)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))
        
    def get_name(self):
        return 'rgbt210_qa_1weight_soft_16cls'

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'init.txt')
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_name):
        seq_path = os.path.join(self.root, seq_name)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_v(self, seq_path, frame_id):
        frame_path_v = os.path.join(seq_path, 'visible', sorted([p for p in os.listdir(os.path.join(seq_path, 'visible')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return self.image_loader(frame_path_v)
        
    def _get_frame_i(self, seq_path, frame_id):
        frame_path_i = os.path.join(seq_path, 'infrared', sorted([p for p in os.listdir(os.path.join(seq_path, 'infrared')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return self.image_loader(frame_path_i)

    def _get_qa_label(self, seq_name, frame_id):
        rgb_t_label_file  = np.loadtxt(os.path.join('/data/liulei/pytracking/pytracking/tracking_results/dimp/dimp50_rgbt_three_branches_20211120/rgbt210', seq_name+'_iou.txt'))[frame_id]   
        # 得分阈值
        thr = torch.nn.functional.softmax(torch.Tensor([rgb_t_label_file[2],rgb_t_label_file[3]]))[0]
        # 得分阈值下标
        index = sum(i/16<thr for i in range(16))
        # 生成标签 np.mean(label)
        label = [ 1 if i<index else 0 for i in range(16) ] 
        return label#torch.nn.functional.softmax(torch.Tensor([rgb_t_label_file[2],rgb_t_label_file[3]]))[0]
    
    def _get_qa_label_2weight(self, seq_name, frame_id):
        rgb_t_label_file  = np.loadtxt(os.path.join('/data/liulei/pytracking/pytracking/tracking_results/dimp/dimp50_rgbt_three_branches_20211120/rgbt210', seq_name+'_iou.txt'))[frame_id]   
        if rgb_t_label_file[2]-rgb_t_label_file[3] >= 0.3:
            label = [1., 0.]
        elif rgb_t_label_file[3]-rgb_t_label_file[2] >= 0.3:
            label = [0., 1.]
        elif rgb_t_label_file[2] >= 0.5 and rgb_t_label_file[3] >= 0.5:
            label = [1., 1.]
        else:
            label = [0., 0.]
        return label
        # if rgb_t_label_file[2]>=0.7 and rgb_t_label_file[3]<=0.3:
        #     label = [1., 0.]
        # elif rgb_t_label_file[3]>=0.7 and rgb_t_label_file[2]<=0.3:
        #     label = [0., 1.]
        # elif rgb_t_label_file[2]>=0.7 and rgb_t_label_file[3]>=0.7:
        #     label = [1., 1.]
        # else:
        #     label = [0., 0.]
        # return label

    def get_frames(self, seq_name, frame_ids, anno=None):
        seq_path = os.path.join(self.root, seq_name)
        frame_list_v = [self._get_frame_v(seq_path, f) for f in frame_ids]
        frame_list_i = [self._get_frame_i(seq_path, f) for f in frame_ids]
        frame_list  = frame_list_v + frame_list_i # 6
        qa_label = [self._get_qa_label(seq_name, f) for f in frame_ids]
        if seq_name not in self.sequence_list:
            print('warning!!!'*100)
        if anno is None:
            anno = self.get_sequence_info(seq_path)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        #return frame_list_v, frame_list_i, anno_frames, object_meta
        return frame_list, anno_frames, object_meta, torch.from_numpy(np.array(qa_label)).float()