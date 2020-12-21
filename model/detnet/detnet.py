'''
detnet based on PyTorch
version: 1.0
author: lingteng qiu 
email: qiulingteng@link.cuhk.edu.cn
'''
import torch
import sys
sys.path.append("./")
from torch import nn, einsum
from einops import rearrange,repeat
import torch.nn.functional as F
import torchvision
from model.helper import resnet50,conv3x3
import numpy as np

def pos_tile():
    retv = torch.from_numpy(np.expand_dims(
        np.stack(
          [
            np.tile(np.linspace(-1, 1, 32).reshape([1, 32]), [32, 1]),
            np.tile(np.linspace(-1, 1, 32).reshape([32, 1]), [1, 32])
          ], -1
        ), 0
        )).float()
    return rearrange(retv,'b h w c -> b c h w')



class net_2d(nn.Module):
    def __init__(self,input_features,output_features,stride,joints=21):
        super().__init__()
        self.project = nn.Sequential(conv3x3(input_features,output_features,stride),nn.BatchNorm2d(output_features),nn.ReLU())
        
        self.prediction = nn.Conv2d(output_features,joints,1,1,0)
    def forward(self,x):
        x = self.project(x)
        x = self.prediction(x).sigmoid()
        return x

class net_3d(nn.Module):
    def __init__(self,input_features,output_features,stride,joints=21,need_norm =False):
        super().__init__()
        self.need_norm = need_norm
        self.project = nn.Sequential(conv3x3(input_features,output_features,stride),nn.BatchNorm2d(output_features),nn.ReLU())
        self.prediction = nn.Conv2d(output_features,joints*3,1,1,0)
    def forward(self,x):
        x = self.project(x)
        x = self.prediction(x)
        
        dmap = rearrange(x,'b (j l) h w -> b j l h w',l=3)

        return dmap



class detnet(nn.Module):
    def __init__(self,stacks =1):
        super().__init__()
        self.resnet50= resnet50()


        self.hmap_0 = net_2d(258,256,1)
        self.dmap_0 = net_3d(279,256,1)
        self.lmap_0 = net_3d(342,256,1)
        self.__pos_tile = pos_tile()
        self.stacks = 1
    
    def forward(self,x):
        device = x.device
        features = self.resnet50(x)
        pos_tile = self.pos.to(device)

        x =torch.cat([features,pos_tile],dim=1)

        hmaps = []
        dmaps = []
        lmaps = []


        for _ in range(self.stacks):
            heat_map = self.hmap_0(x)
            hmaps.append(heat_map)
            x = torch.cat([x,heat_map],dim=1)

            dmap = self.dmap_0(x)
            dmaps.append(dmap)

            x = torch.cat([x,rearrange(dmap,'b j l h w -> b (j l) h w')],dim =1)

            lmap = self.lmap_0(x)
            lmaps.append(lmap)
        hmap,dmap,lmap = hmaps[-1],dmaps[-1],lmaps[-1]


        uv,argmax = self.map_to_uv(hmap)

        delta = self.dmap_to_delta(dmap,argmax)
        xyz = self.lmap_to_xyz(lmap,argmax)

        return uv,xyz

    
    @property
    def pos(self):
        return self.__pos_tile
    @staticmethod
    def map_to_uv(hmap):

        b, j, h, w = hmap.shape
        hmap = rearrange(hmap,'b j h w -> b j (h w)')
        argmax = torch.argmax(hmap,-1,keepdim=True)
        u = argmax//w
        v = argmax %w
        uv = torch.cat([u,v],dim =-1)

        return uv,argmax

    @staticmethod
    def dmap_to_delta(dmap,argmax):
        return detnet.lmap_to_xyz(dmap,argmax)
    @staticmethod
    def lmap_to_xyz(lmap,argmax):        
        #output: batch, joints ,3
        lmap = rearrange(lmap,'b j l h w -> b j (h w) l')
        index = repeat(argmax,'b j i -> b j i c',c=3)
        xyz = torch.gather(lmap,dim=2,index=index).squeeze()
        return xyz


def load_model_from_tensorflow():
    

    import tensorflow as tf
    from tensorflow.python import pywrap_tensorflow

    model_reader = pywrap_tensorflow.NewCheckpointReader("../weights/minimal_hand/model/detnet/detnet.ckpt")

    value = var_dict = model_reader.get_variable_to_shape_map()
    keys = value.keys()
    keys = sorted(keys)

    need_weights = {}
    for key in keys:
        if 'Adam' not in key:
            if 'resnet' in key:
                continue
            if 'train/' in key:
                continue
            transfer_key = key.split('prior_based_hand/')[-1]
            transfer_key = transfer_key.replace('/','.')
            transfer_key = transfer_key.replace('moving_mean','running_mean').replace('moving_variance','running_var')
            transfer_key = transfer_key.replace('gamma','weight').replace('beta','bias').replace('batch_normalization','bn')

            transfer_key = transfer_key.replace("kernel",'weight').replace('bn','1').replace("prediction.conv2d.",'prediction.').replace('project.conv2d.','project.0.')
            need_weights[transfer_key] = model_reader.get_tensor(key)
    return need_weights





if __name__ == '__main__':



    detnet = detnet()
    inp  = np.load("./input.npy")
    output = np.load('./output.npy')

    hmap = np.load("./hmap.npy")
    dmap = np.load("./dmap.npy")

    lmap = np.load("./lmap.npy")
    hmap = torch.from_numpy(hmap)
    hmap = rearrange(hmap,'b h w c -> b c h w')


    #[1, 21, 3, 32, 32]
    dmap = torch.from_numpy(dmap)
    dmap = rearrange(dmap,'b h w j l -> b j l h w')

    lmap = torch.from_numpy(lmap)
    lmap = rearrange(lmap,'b h w j l -> b j l h w')


    output = torch.from_numpy(output)
    output = rearrange(output,'b h w c -> b c h w')

    
    
    inp = torch.from_numpy(inp)
    inp = rearrange(inp,'b h w c -> b c h w')  

    detnet.eval()

    # r, h,d,l = detnet(inp)
    # print(torch.sum(r-output))
    # print(torch.sum(l-lmap))

    detnet.load_state_dict(torch.load("./weights/detnet.pth"))
    
    uv,xyz = detnet(inp)
    print(uv)
    print(xyz)

    # print(torch.sum(features-output))
    # print(torch.sum(l-lmap),torch.max(l-lmap))

    
        
