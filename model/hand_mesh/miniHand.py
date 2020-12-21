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
import numpy as np
import pickle
from model.hand_mesh.kinematics import *
from model.detnet import detnet
from model.iknet import iknet


class minimal_hand(nn.Module):
    def __init__(self,mano_path,detnet_path =None,iknet_path=None):
        super().__init__()
        self.para_init(mano_path)
        self.detnet = detnet(stacks = 1)
        self.iknet = iknet(inc = 84*3,depth = 6, width = 1024)

        self.model_init(detnet_path,iknet_path)

    def para_init(self,mano_path):
        '''
        para_init
        mpii_ref_delta means the longth of bone between children joints and parent joints note that root is equal to itself.
        '''
        self.__ik = 0.09473151311686484
        mano_ref_xyz = self.__load_pkl(mano_path)['joints']
        mpii_ref_xyz = mano_to_mpii(mano_ref_xyz) / self.__ik
        mpii_ref_xyz -= mpii_ref_xyz[9:10]
        mpii_ref_delta, mpii_ref_length = xyz_to_delta(mpii_ref_xyz, MPIIHandJoints)
        mpii_ref_delta = mpii_ref_delta * mpii_ref_length

        self.__mpii_ref_xyz = torch.from_numpy(mpii_ref_xyz).float()
        self.__mpii_ref_delta = torch.from_numpy(mpii_ref_delta).float()
    def forward(self,x):
        # b c h w == 128
        uv,xyz = self.detnet(x)
        device = xyz.device


        #this 11-12 delta have some mistake, need to check why the different is so high.
        delta, length = xyz_to_delta_tensor(xyz, MPIIHandJoints,device=device)



        delta *= length
        pack = torch.cat([xyz, delta, self.mpii_ref_xyz.to(device), self.mpii_ref_delta.to(device)],dim=0).unsqueeze(0)
        



        theta = self.iknet(pack)[0]
        
        return xyz,theta[0]

        
    
    @property
    def ik_unit_length(self):
        return self.__ik
    @property
    def mpii_ref_xyz(self):
        return self.__mpii_ref_xyz
    @property
    def mpii_ref_delta(self):
        return self.__mpii_ref_delta  

    def model_init(self,detnet,iknet):
        if detnet == None:
            raise NotImplementedError
        if iknet == None:
            raise NotImplementedError
        self.detnet.load_state_dict(torch.load(detnet))
        self.iknet.load_state_dict(torch.load(iknet))
        
    def __load_pkl(self,path):
        """
        Load pickle data.
        Parameter
        ---------
        path: Path to pickle file.
        Return
        ------
        Data in pickle file.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

if __name__ == '__main__':
    hand_machine = minimal_hand('../weights/minimal_hand/model/hand_mesh/hand_mesh_model.pkl','./weights/detnet.pth','./weights/iknet.pth')
    hand_machine.eval()
    inp  = np.load("./input.npy")
    output = np.load('./output.npy')
    output = torch.from_numpy(output)
    output = rearrange(output,'b h w c -> b c h w')
    inp = torch.from_numpy(inp)
    inp = rearrange(inp,'b h w c -> b c h w')  

    x = np.load("iknet_inputs.npy")
    x = torch.from_numpy(x).float()

    gt = np.load("theta.npy")
    gt = torch.from_numpy(gt).float()
    hand_machine.cuda()
    hand_machine(inp.cuda())
    