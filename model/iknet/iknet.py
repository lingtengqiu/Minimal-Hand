'''
iknet based on PyTorch
version: 1.0
author: lingteng qiu 
email: qiulingteng@link.cuhk.edu.cn
'''
import torch
import torch.nn as nn
from torch import nn, einsum
from einops import rearrange,repeat
import torch.nn.functional as F
import numpy as np

import warnings
warnings.filterwarnings("ignore")





# def dense(layer, n_units):
#       layer = tf.layers.dense(
#     layer, n_units, activation=None,
#     kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
#     kernel_initializer=tf.initializers.truncated_normal(stddev=0.01)
#   )
#   return layer


def get_iknet_weights():

    import tensorflow as tf
    from tensorflow.python import pywrap_tensorflow
    model_reader = pywrap_tensorflow.NewCheckpointReader("../weights/minimal_hand/model/iknet/iknet.ckpt")

    value = var_dict = model_reader.get_variable_to_shape_map()
    keys = value.keys()
    keys = sorted(keys)

    remain_weights = {}
    for key in keys:
        if 'Adam' not in key:
            if 'train' in key:
                continue

            transfer_key = key.replace('/','.')
            transfer_key = transfer_key.replace('moving_mean','running_mean').replace('moving_variance','running_var')
            transfer_key = transfer_key.replace('gamma','weight').replace('beta','bias').replace('batch_normalization','bn')
            
            if 'bn' in transfer_key:
                bn_key = transfer_key.split('.')[0]
                if len(bn_key) ==2:
                    transfer_key = transfer_key.replace(bn_key,'dense.dense.1')
                else:
                    num = bn_key.split('_')[-1]
                    transfer_key = transfer_key.replace(bn_key,'dense_{}.dense.1'.format(num))
            else:
                transfer_key = transfer_key.replace('kernel','weight')
                if '6' not in transfer_key:
                    transfer_key = transfer_key.replace('.','.dense.0.')
            remain_weights[transfer_key] = model_reader.get_tensor(key)

    return remain_weights
    #inc ouc

    # for e in ele:
    #     print(e)


class dense_bn(nn.Module):
    def __init__(self,inc,ouc):
        super().__init__()
        self.dense = nn.Sequential(nn.Linear(inc,ouc,True),nn.BatchNorm1d(ouc),nn.Sigmoid())
    def forward(self,x):
        return self.dense(x)
    

class iknet(nn.Module):
    def __init__(self,inc,depth,width,joints = 21):
        super().__init__()

        self.depth = depth
        self.width = width
        self.__eps =  torch.tensor(np.finfo(np.float32).eps).float()
        
        #to match easily, I write it so stupidly.
        self.dense = dense_bn(inc,width)
        self.dense_1 = dense_bn(width,width)
        self.dense_2 = dense_bn(width,width)
        self.dense_3 = dense_bn(width,width)
        self.dense_4 = dense_bn(width,width)
        self.dense_5 = dense_bn(width,width)

        self.dense_6 = nn.Linear(width,joints*4)
    def forward(self,x):
        '''
        joints : 21*4
        x :(batch 84 ,3) --> (batch 84*3)
        '''
        x = rearrange(x,'b j c -> b (j c)',c= 3)
        device =x.device

        x = self.dense(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.dense_5(x)

        theta_raw = self.dense_6(x)
        theta_raw = rearrange(theta_raw,'b (j n) -> b j n', n=4)

        norm = torch.maximum(torch.norm(theta_raw,dim=-1,keepdim=True),self.eps.to(device))

        theta_pos = theta_raw /norm

        theta_neg = theta_pos*-1

        flag = repeat(theta_pos[:,:,0]>0,'b j ->b j c',c = 4)
        theta = torch.where(flag,theta_pos,theta_neg)

        return theta ,norm 
    @property
    def eps(self):
        return  self.__eps




if __name__ == '__main__':

    x = np.load("iknet_inputs.npy")
    gt = np.load("theta.npy")

    x = torch.from_numpy(x).float()
    gt = torch.from_numpy(gt).float()

    iknet = iknet(84*3,6,1024)
    iknet.eval()

    
    # torch_keys = sorted(model_params.keys())  
    # remain_torch_keys = []
    # for key in torch_keys:
    #     if 'num_batches_tracked' not in key:
    #         remain_torch_keys.append(key)

    
    # tensor_weights = get_iknet_weights()

    # keys = sorted(tensor_weights.keys())
    # for key in keys:
    #     tensor_weights[key] = torch.from_numpy(tensor_weights[key])
    #     if len(tensor_weights[key].shape)>1:
    #         tensor_weights[key] = rearrange(tensor_weights[key],'ic oc -> oc ic')
    
    # flag = iknet.load_state_dict(tensor_weights)
    iknet.load_state_dict(torch.load("./weights/iknet.pth"))
    theta, norm = iknet(x)

    print(torch.sum(theta-gt))



    
    