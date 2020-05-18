import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2
from data.dataset import to_tensor, Norm, TinyMind
import numpy as np
from models.ResNet import resnet50
import torch.optim as optim
import shutil  

def save(model, optimizer, step, accuracy, best_accuracy, saveDir):
    path = saveDir + '{:03}.pt'.format(step)

    state = {}
    state['state_dict'] = model.state_dict()
    state['accuracy'] = accuracy
    state['step'] = step
    state['lr']=optimizer.param_groups[0]['lr']
    torch.save(state, path)
    print('save model at epoch{}'.format(step))

    if accuracy == best_accuracy:
        best_accuracy = accuracy
        best_path = saveDir + 'best_model.pt'
        shutil.copyfile(path, best_path)#clone 'path' named 'best_path'
        print('best model in step {}'.format(step))

    return best_accuracy

def adjust_lr(step, warm_step, optimizer):
    lr = optimizer.param_groups[0]['lr']
    if step < warm_step:
        lr*=10
    elif (step-warm_step)%4==3:
        lr*=0.7
    for p in optimizer.param_groups:
        p['lr']=lr
    return lr 

def init_weights(net):
    """the weights of conv layer and fully connected layers 
    are both initilized with Xavier algorithm, In particular,
    we set the parameters to random values uniformly drawn from [-a, a]
    where a = sqrt(6 * (din + dout)), for batch normalization 
    layers, y=1, b=0, all bias initialized to 0.
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    return net

def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias, 
    bn weights, bn bias, linear bias)

    Args:
        net: network architecture
    
    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)
        
        else: 
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)
        
    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]