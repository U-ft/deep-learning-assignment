import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision.transforms as T
import numpy as np

def showTensor(tensor,mean,std):
    img=tensor.numpy()
    img=img.transpose([1,2,0])#C H W->H W C
    img=(img*0.5+0.5)
    cv2.imshow('tensor2img',img)
    cv2.waitKey(0)

def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    return cv_img

class TinyMind(Dataset):

    def __init__(self, mode, root, fpath, trans):
        super().__init__()
        self.transform = trans
        self.root=root
        self.mode=mode
        img = []
        label = []

        if mode == "train" or mode == 'val':
            with open(root+fpath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split()
                    img.append(line[0])
                    label.append(line[1])

        elif mode == 'test':
            with open(root+fpath, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split()
                    img.append(line[0])
        
        self.img = img
        self.label = label

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img = cv_imread(self.root + self.img[idx])

        if self.transform is not None: 
            img = self.transform(img)
        if self.mode =='train' or self.mode =='val':
            return img, int(self.label[idx])
        elif self.mode =='test':
            return self.img[idx].split('/')[-1], img


class Norm():

    def __init__(self, mean, std):
        self.mean=mean
        self.std=std

    def __call__(self, img):
       
        img = img / 255.0

        img = (img - self.mean) / self.std
        img=cv2.resize(img,(112,112),interpolation=cv2.INTER_NEAREST)

        return img[:,:,None]


class to_tensor():

    def __call__(self, img):
        # H x W x C ---> C x H x W
        img = torch.from_numpy(img.transpose([2, 0, 1])).type(torch.FloatTensor)

        return img


if __name__ == "__main__":
    
    mean = [0.5]
    std = [0.5]
    # a = cv2.imread("1.jpg")
    # trans = Norm(mean, std)
    # a = trans(a)
    # cv2.imshow('2',a)
    # cv2.waitKey()
    train_set = TinyMind(mode='train', root='data/', fpath="train.txt", trans=T.Compose([Norm(mean, std), to_tensor()]))
    train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True, num_workers=0)

    it = iter(train_loader)
    c, b = it.next()
    showTensor(c[0],mean,std)
    print(b[0])
