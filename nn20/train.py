import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2
from data.dataset import to_tensor, Norm, TinyMind
import numpy as np
from models.ResNet import resnet50
from models.res2next import res2next50
import torch.optim as optim

from utils import *

mean=[0.485]
std=[0.229]

device = torch.device('cuda')
epoch=60
#model_path='checkpoints/020.pt'
model_path=None
saveDir='checkpoints/'
usingloss='LableSmoothing'


def train(model, train_loader, optimizer, loss_func, step):
    model.train()
    aver_loss = 0.0
    total=correct=0
    step += 1
    for i , data in enumerate(train_loader):
        optimizer.zero_grad()

        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        outs = model(imgs)

        loss = loss_func(outs,labels) 
       
        loss.backward()
        optimizer.step()
        
        aver_loss += loss.item()
        _, preds = outs.max(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        if i % 100 == 99:    # print every 100 mini-batches
            accuracy = 100.0 * correct / total
            print('epoch{}  {}batches: loss1={}, acc={}%'.format(step , i + 1, aver_loss / 100, accuracy))
            aver_loss = 0.0   

    return step

def val(model, val_loader, best_accuracy):
    model.eval()
    total=0
    correct=0
    with torch.no_grad():
        for i , data in enumerate(val_loader):

            img,labels=data
            img,labels=img.to(device),labels.to(device)
            out=model(img)

            _,pred=out.max(dim=1)   
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            accuracy = 100.0 * correct / total
        print('Accuracy of val: {}%'.format(accuracy))

        if accuracy > best_accuracy:
            best_accuracy = accuracy

    return best_accuracy,accuracy


def main():
  
    train_set = TinyMind(mode='train',root='data/',fpath="train.txt",trans=T.Compose([Norm(mean,std),to_tensor()]))
    val_set = TinyMind(mode='val',root='data/',fpath="val.txt",trans=T.Compose([Norm(mean,std),to_tensor()]))
    train_loader = DataLoader(dataset=train_set, batch_size=48, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_set, batch_size=112, shuffle=False, num_workers=0)

    net=res2next50()
    #net = init_weights(net)
    net=net.to(device)

    if usingloss == 'CrossEntropy':
        loss_func = nn.CrossEntropyLoss()
    elif usingloss == 'LableSmoothing':
        from models.LabelSmoothing import LSR
        loss_func = LSR()

    #apply no weight decay on bias
    params = split_weights(net)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    step = 0
    accuracy = 0.0
    best_accuracy = 0.0

    if model_path is not None:
        state = torch.load(model_path)
        net.load_state_dict(state['state_dict'])
        step = state['step']
        best_accuracy = state['accuracy']
        for p in optimizer.param_groups:
            p['lr']=state['lr']
            #p['lr']=0.001
        print('\nload model from {}\n'.format(model_path))

    for i in range(epoch):
        
        step = train(model=net, train_loader=train_loader, optimizer=optimizer, loss_func=loss_func, step=step)
        adjust_lr(step=step, warm_step=3, optimizer=optimizer)
        if step>=3:      
            best_accuracy, accuracy = val(model=net, val_loader=val_loader, best_accuracy=best_accuracy)
            print('best_accuracy: {}%\n'.format(best_accuracy))
            save(model=net, optimizer=optimizer, step=step, accuracy=accuracy, best_accuracy=best_accuracy, saveDir=saveDir)

if __name__=="__main__":

    main()