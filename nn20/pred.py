import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from data.dataset import to_tensor, Norm, TinyMind
from models.ResNet import resnet50
import csv
import numpy as np
import cv2

mean=[0.485]
std=[0.229]

device = torch.device('cuda')
model_path='checkpoints/019.pt'

test_set = TinyMind(mode='test',root='data/',fpath="test2.txt",trans=T.Compose([Norm(mean,std),to_tensor()]))
test_loader = DataLoader(dataset=test_set, batch_size=256, shuffle=False, num_workers=0)

net=resnet50().to(device)
state = torch.load(model_path)
net.load_state_dict(state['state_dict'])


classes = ['且', '世', '东', '九', '亭', '今', '从', '令', '作', '使', '侯', '元', '光', '利', '印', '去', '受', '右',
    '司', '合', '名', '周', '命', '和', '唯', '堂', '士', '多', '夜', '奉', '女', '好', '始', '字', '孝',
    '守', '宗', '官', '定', '宜', '室', '家', '寒', '左', '常', '建', '徐', '御', '必', '思', '意', '我',
    '敬', '新', '易', '春', '更', '朝', '李', '来', '林', '正', '武', '氏', '永', '流', '海', '深', '清',
    '游', '父', '物', '玉', '用', '申', '白', '皇', '益', '福', '秋', '立', '章', '老', '臣', '良', '莫',
    '虎', '衣', '西', '起', '足', '身', '通', '遂', '重', '陵', '雨', '高', '黄', '鼎']
results_csv = csv.writer(open('./data/result.csv','w', encoding='utf-8',newline=''))
results_csv.writerow(['filename', 'label'])

net.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):

        img_names, images = data
        images = images.to(device)
        outs = net(images)

        outs = outs.detach_().cpu().numpy()
        preds = np.argsort(outs,1)[:,::-1]
        preds = preds[:,0:5]
        for idx, p in enumerate(preds):
            results = [classes[p[i]] for i in range(1)]
            results = ''.join(results)

            results = [img_names[idx], results]

            results_csv.writerow(results)

print('done')