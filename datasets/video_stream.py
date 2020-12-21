from torch.utils.data import Dataset
import os
import cv2 
import numpy as np
class video_stream(Dataset):
    def __init__(self,video_path):
        names = sorted(os.listdir(video_path))
        self.names = [os.path.join(video_path,name) for name in names]
        self.__cnt = 0
    def __getitem__(self,item):
        img_name = self.names[item]
        img = cv2.imread(img_name)
        #transfer BGR 2 RGB 
        return np.flip(img, -1).copy()
    
    @property
    def cnt(self):
        return self.__cnt
    @cnt.setter
    def cnt(self,value):
        self.__cnt = value
    
    def empty(self):
        return self.__cnt >=self.__len__()
    def read(self):
        if not self.empty():
            img = self.__getitem__(self.cnt)
            self.cnt+=1
            return True,img
        return False, None


    def __len__(self):
        return len(self.names)

        