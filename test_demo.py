import torch
from QAN import SQA, TQA
import pandas as pd
import numpy as np
from DatasetPatch import VideoPatchData
from tool import plcc_srcc


train_test_ref_idx = np.loadtxt('data/train_test_LIVE_VQA.txt')
train_test_ref_idx = train_test_ref_idx.astype(np.int)
train_ref_idx_list = train_test_ref_idx[:8]
test_ref_idx_list = train_test_ref_idx[8:]

database = 'LIVE'
test_batch_size = 9
device_single = torch.device('cuda:0')

test_data = VideoPatchData(database=database, ref_idx_list=test_ref_idx_list)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, num_workers=8)

trained_SQA_model_file = 'model/SQA.pt'
SQA_model = SQA()
SQA_model.load_state_dict(torch.load(trained_SQA_model_file))
SQA_model.to(device_single)

trained_TQA_model_file = 'model/TQA.pt'
TQA_model = TQA()
TQA_model.load_state_dict(torch.load(trained_TQA_model_file))
TQA_model.to(device_single)

dis_idx_list = []
score_list = []
predict_list = []

for k, data in enumerate(test_loader):
    print(k, 'train')

    dis = data['dis']
    err = data['err']
    err_4 = data['err_4']
    score_label = data['mos']
    score_list.extend(score_label.tolist())

    dis = dis.to(device_single)
    err = err.to(device_single)
    err_4 = err_4.to(device_single)
    score_label.unsqueeze_(1)
    score_label = score_label.to(device_single)

    dis_idx = data['dis_idx']
    dis_idx_list.extend(dis_idx.tolist())

    SQA_model.eval()
    with torch.no_grad():
        _, out = SQA_model(dis, err, err_4)
        out = out * err_4
        out = torch.mean(out[:, :, 0, 4:-4, 4:-4], dim=[-2, -1])
        predict_list.append(out.cpu())

predict_list = torch.cat(predict_list, dim=0)
predict_list = predict_list.numpy()

dis_idx_list_uniqe = np.unique(dis_idx_list)
score_list = np.array(score_list)
dis_idx_list = np.array(dis_idx_list)

predict_list = [predict_list[dis_idx_list == dis_idx, :] for dis_idx in dis_idx_list_uniqe]
predict_list = np.stack(predict_list, axis=0)
# batch_size, channel_size, seq_len
predict_list = np.mean(predict_list, axis=1, keepdims=True)
# batch_size, channel_size, seq_len --> # batch_size, seq_len, channel_size
predict_list = np.transpose(predict_list, (0, 2, 1))

score_list = [score_list[dis_idx_list == dis_idx] for dis_idx in dis_idx_list_uniqe]
score_list = np.stack(score_list, axis=0)
score_list = np.mean(score_list, axis=1, keepdims=True)

predict_list = torch.from_numpy(predict_list)

with torch.no_grad():
    score_predict = TQA_model(predict_list.to(device_single))

score_predict = score_predict.cpu().numpy()
score_predict = np.squeeze(score_predict)
score_list = np.squeeze(score_list)

plcc_srcc_all = plcc_srcc(score_predict, score_list)

result_excel = 'result/test_result.xlsx'
result = pd.DataFrame()
result['PLCC'] = [plcc_srcc_all[0]]
result['SROCC'] = [plcc_srcc_all[1]]
result.to_excel(result_excel)



