import torch
from torch.utils.data import Dataset
import numpy as np


class VideoPatchData(Dataset):
    def __init__(self, database='LIVE', dis_type_list=list(range(1, 5)), ref_idx_list=list(range(1, 11)),
                 file_name='p_112_80_8_30'):
        super(VideoPatchData, self).__init__()
        if database == 'LIVE':
            video_info_file = 'data/'+ database + '_' + file_name + '.txt'
            self.video_patch_path = 'Set your video patch path'
            img_height, img_width = (432, 768)

        video_info_id = open(video_info_file, mode='r')
        video_info_list = video_info_id.readlines()

        chosen_video_info_list = []
        dis_idx_list = []
        dis_type_idx_list = []
        label_list = []
        for k in range(len(video_info_list)):
            line_content = (video_info_list[k]).strip()
            line_content = line_content.split(' ')
            dis_type_idx = int(line_content[2])
            ref_idx = int(line_content[4])
            dis_idx = int(line_content[1])
            label = np.float32(line_content[5])
            if (dis_type_idx in dis_type_list) and (ref_idx in ref_idx_list):
                chosen_video_info_list.append(line_content)
                dis_idx_list.append(dis_idx)
                dis_type_idx_list.append(dis_type_idx)
                label_list.append(label)

        self.chosen_video_info_list = chosen_video_info_list

        dis_idx_list = np.array(dis_idx_list)
        dis_type_idx_list = np.array(dis_type_idx_list)
        label_list = np.array(label_list)

        dis_idx_unique_list = np.unique(np.array(dis_idx_list))

        dis_type_idx_2_dis_idx_unique = []
        label_2_dis_idx_unique = []

        for k in range(len(dis_idx_unique_list)):
            dis_type_idx_2_dis_idx_unique.append((dis_type_idx_list[dis_idx_list == dis_idx_unique_list[k]])[0])
            label_2_dis_idx_unique.append((label_list[dis_idx_list == dis_idx_unique_list[k]])[0])
        dis_type_idx_2_dis_idx_unique = np.array(dis_type_idx_2_dis_idx_unique)
        label_2_dis_idx_unique = np.array(label_2_dis_idx_unique)

        self.dis_idx_list = dis_idx_unique_list
        self.dis_type_idx_list = dis_type_idx_2_dis_idx_unique
        self.label_list = label_2_dis_idx_unique

        self.stride = 80
        self.patch_size = 112
        self.patch_y_num = (img_height - self.patch_size) // self.stride + 1
        self.patch_x_num = (img_width - self.patch_size) // self.stride + 1

    def __len__(self):
        return len(self.chosen_video_info_list)

    def __getitem__(self, item):
        item_info = self.chosen_video_info_list[item]
        dis_idx = int(item_info[1])
        dis_type_idx = int(item_info[2])
        mos = np.float32(item_info[5])
        patch_idx = item_info[7]

        dis_file_name = self.video_patch_path + 'dis_' + str(dis_idx) + '_' + patch_idx + '.npy'
        err_file_name = self.video_patch_path + 'err_' + str(dis_idx) + '_' + patch_idx + '.npy'
        err4_file_name = self.video_patch_path + 'err4_' + str(dis_idx) + '_' + patch_idx + '.npy'

        dis_patch = np.load(dis_file_name)
        err_patch = np.load(err_file_name)
        err4_patch = np.load(err4_file_name)
        dis_patch = torch.from_numpy(dis_patch)
        err_patch = torch.from_numpy(err_patch)
        err4_patch = torch.from_numpy(err4_patch)

        return {'dis': dis_patch, 'err': err_patch, 'err_4': err4_patch, 'mos': mos,
                'dis_idx': dis_idx, 'dis_type_idx': dis_type_idx}
