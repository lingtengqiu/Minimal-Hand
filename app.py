import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import cv2
import warnings
warnings.filterwarnings("ignore")
import keyboard
import numpy as np
import open3d as o3d
import pygame
from transforms3d.axangles import axangle2mat
import config
from hand_mesh import HandMesh
from kinematics import mpii_to_mano
from utils import OneEuroFilter, imresize
from utils import *
import time
from datasets import video_stream
from model.hand_mesh import minimal_hand
import torch
from einops import rearrange
from open3d import io as io

import cv2
import os 
from render import o3d_render



def live_application(capture):
  """
  Launch an application that reads from a webcam and estimates hand pose at
  real-time.

  The captured hand must be the right hand, but will be flipped internally
  and rendered.

  Parameters
  ----------
  capture : object
    An object from `capture.py` to read capture stream from.
  """
  
  ############render setting############ 
  render = o3d_render(config.HAND_MESH_MODEL_PATH)
  extrinsic = render.extrinsic
  extrinsic[0:3, 3] = 0
  render.extrinsic = extrinsic
  render.intrinsic = [config.CAM_FX,config.CAM_FY]
  render.updata_params()
  render.environments('./render_option.json',1000)

  extrinsic = render.extrinsic.copy()
  intrinsic = render.intrinsic

  ############ misc ############
  mesh_smoother = OneEuroFilter(4.0, 0.0)
  clock = pygame.time.Clock()
  hand_machine = minimal_hand('../weights/minimal_hand/model/hand_mesh/hand_mesh_model.pkl','./weights/detnet.pth','./weights/iknet.pth')


  hand_machine.cuda()
  hand_machine.eval()
  cnt = 0

  while True:
    flag,frame_large = capture.read()
    cnt+=1
    if not flag:
      break
    if frame_large is None:
      continue
    if frame_large.shape[0] > frame_large.shape[1]:
      margin = int((frame_large.shape[0] - frame_large.shape[1]) / 2)
      frame_large = frame_large[margin:-margin]
    else:
      margin = int((frame_large.shape[1] - frame_large.shape[0]) / 2)
      frame_large = frame_large[:, margin:-margin]

    frame_large = np.flip(frame_large, axis=1).copy()
    frame = imresize(frame_large, (128, 128))
    original_img = frame.copy()[...,::-1]
    original_img = cv2.resize(original_img,(256,256))
    frame = torch.from_numpy(frame)
    frame = rearrange(frame,'h w c -> 1 c h w')
    frame = frame.float()/255
    with torch.no_grad():
      frame = frame.cuda()
      _, theta_mpii = hand_machine(frame)
    theta_mpii = theta_mpii.detach().cpu().numpy()
    theta_mano = mpii_to_mano(theta_mpii)


    v = render.hand_mesh.set_abs_quat(theta_mano)
    v *= 2 # for better visualization
    v = v * 1000 + np.array([0, 0, 400])
    v = mesh_smoother.process(v)

    render.rendering(v,config.HAND_COLOR)    
    render_img = render.capture_img()
    xy  = render.projector()

    for joint in xy.astype(np.int).tolist():
      render_img = cv2.circle(render_img,tuple(joint[:2]),3,(0,0,255),3)



    render_img = cv2.resize(render_img,(256,256))
    save_img = np.concatenate([original_img,render_img],axis=1)

    cv2.imshow("fuck",save_img)
    cv2.imwrite("render_results/%06d.png"%cnt ,save_img)


if __name__ == '__main__':
  live_application(video_stream("../test_video/img_3/"))
