import torch
import os
import numpy as np
from tool import log_diff, ImageDownSample, VideoTemporalSample, ReadY
import pandas as pd


database = 'LIVE'
device = torch.device('cuda:0')


if database == 'LIVE':
    video_database_path = 'your video database path/'
    video_info_file = 'data/LIVE_Video_Quality_Assessment_Database_videos_y.txt'
    database_info_file = 'data/LIVE_info.xlsx'
    img_height, img_width = (432, 768)

patch_size = 112
patch_stride = 80
clip_size = 8
clip_len = 30


test_index = -1

# video patches path
patch_clip_name = 'p_' + str(patch_size) + '_' + str(patch_stride) + '_' + str(clip_size) + '_' + str(clip_len)
VideoDataSet_path = video_database_path + patch_clip_name + '/'
if not os.path.exists(VideoDataSet_path):
    os.makedirs(VideoDataSet_path)

video_info_id = open(video_info_file, mode='r')
video_info_list = video_info_id.readlines()

video_patch_database_file = 'data/' + database + '_' + patch_clip_name + '.txt'

if os.path.exists(video_patch_database_file):
    raise ValueError('file has exist !!!')
else:
    os.mknod(video_patch_database_file)

fp = open(video_patch_database_file, 'w')

read_yuv_y = ReadY(path=video_database_path, width=img_width, height=img_height)

temproal_clip = VideoTemporalSample(clip_len=clip_len, clip_size=clip_size)

down_sample = ImageDownSample(level=2)
down_sample.to(device)

database_info = pd.read_excel(database_info_file)

neg_pix_y = (img_height-patch_size) % patch_stride
neg_pix_x = (img_width-patch_size) % patch_stride
y_start = neg_pix_y // 2
x_start = neg_pix_x // 2
y_start_list = list(range(y_start, img_height-patch_size+1, patch_stride))
x_start_list = list(range(x_start, img_width-patch_size+1, patch_stride))
y_start_list_4 = [(k - y_start) // 4 for k in y_start_list]
x_start_list_4 = [(k - x_start) // 4 for k in x_start_list]

ref_num = 1
for k_line in video_info_list:
    k_content = k_line.strip()
    k_line = k_line.split(' ')
    dis_idx = k_line[1]
    dis_type_idx = k_line[2]
    ref_idx = k_line[4]
    mos = k_line[5]

    dis_video_file = database_info['dis'].values[int(dis_idx)-1]
    ref_video_file = database_info['ref'].values[int(dis_idx)-1]

    dis_video, _ = read_yuv_y(dis_video_file)
    ref_video, _ = read_yuv_y(ref_video_file)

    dis_video = torch.from_numpy(dis_video)/255.0
    ref_video = torch.from_numpy(ref_video)/255.0

    dis_video = dis_video.unsqueeze(1)
    ref_video = ref_video.unsqueeze(1)

    #  (seq_len, 1, height, width) --> (seq_len, 1, depth, height, width)
    dis_video_clip = temproal_clip(dis_video)
    ref_video_clip = temproal_clip(ref_video)

    # dis_video
    dis_video_clip_patch = torch.stack([dis_video_clip[:, :, :, y_idx:(y_idx + patch_size), x_idx:(x_idx + patch_size)]
                                        for y_idx in y_start_list for x_idx in x_start_list], dim=0)

    # err_video
    err_video_clip = log_diff(dis_video_clip, ref_video_clip)
    err_video_clip_patch = torch.stack([err_video_clip[:, :, :, y_idx:(y_idx + patch_size), x_idx:(x_idx + patch_size)]
                                        for y_idx in y_start_list for x_idx in x_start_list], dim=0)

    err_video_clip_crop = err_video_clip[:, :, test_index, y_start_list[0]:(y_start_list[-1]+patch_size),
                          x_start_list[0]:(x_start_list[-1]+patch_size)]

    with torch.no_grad():
        err_video_clip_crop_4 = down_sample(err_video_clip_crop.to(device))

    err_video_clip_crop_4 = err_video_clip_crop_4.cpu()
    err_video_clip_patch_4 = torch.stack([err_video_clip_crop_4[:, :, y_idx:(y_idx + patch_size//4), x_idx:(x_idx + patch_size//4)]
                                          for y_idx in y_start_list_4 for x_idx in x_start_list_4], dim=0)

    for kk in range(err_video_clip_patch_4.size(0)):
        dis_video_clip_patch_file_name = 'dis_' + dis_idx + '_' + str(kk) + '.npy'
        err_video_clip_patch_file_name = 'err_' + dis_idx + '_' + str(kk) + '.npy'
        err_video_clip_patch_4_file_name = 'err4_' + dis_idx + '_' + str(kk) + '.npy'

        np.save(VideoDataSet_path + dis_video_clip_patch_file_name, dis_video_clip_patch[kk, ].numpy())
        np.save(VideoDataSet_path + err_video_clip_patch_file_name, err_video_clip_patch[kk, ].numpy())
        np.save(VideoDataSet_path + err_video_clip_patch_4_file_name, err_video_clip_patch_4[kk, ].numpy())

        kk_line = k_content + ' ' + str(kk) + '\n'
        fp.write(kk_line)
        fp.flush()
fp.close()
