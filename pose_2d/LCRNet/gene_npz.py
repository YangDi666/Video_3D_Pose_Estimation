import torch
#  import ipdb;ipdb.set_trace()
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms



import pickle
from pose_2d.LCRNet.detect_pose import detect_pose
from pose_2d.LCRNet.LCRNet_2D_3D_Pose_Estimation.tools.lcr_net_ppi import LCRNet_PPI
import pose_2d.LCRNet.LCRNet_2D_3D_Pose_Estimation.tools.scene as scene

import torch.nn as nn
import torch.utils.data
import numpy as np
import os
import sys

main_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(main_path)

import ntpath
from tqdm import tqdm
import time
import cv2

# posemap
coco=['nose', 'Leye' ,'Reye' ,'Lear' ,'Rear' ,'Lsho' ,'Rsho', 'Lelb', 'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank' ,'Rank']
lcrnet={'nose': 12, 'Leye':12 ,'Reye':12 ,'Lear':12 ,'Rear':12 ,'Lsho':11 ,'Rsho':10, 'Lelb':9, 'Relb':8, 'Lwri':7, 'Rwri':6, 'Lhip':5, 'Rhip':4, 'Lkne':3, 'Rkne':2, 'Lank':1 ,'Rank':0}


def get_pose( imagename, modelname, gpuid):

    fname = os.path.join(os.path.dirname(__file__), 'models', modelname+'.pkl')
    if not os.path.isfile(fname):
        # Download the files 
        dirname = os.path.dirname(fname)
        if not os.path.isdir(os.path.dirname(fname)):
            os.system('mkdir -p "{:s}"'.format(dirname))
        os.system('wget http://pascal.inrialpes.fr/data2/grogez/LCR-Net/pthmodels/{:s} -P {:s}'.format(modelname+'.pkl', dirname))        
        if not os.path.isfile(fname):
            raise Exception("ERROR: download incomplete")

    with open(fname, 'rb') as fid:
      model = pickle.load(fid)
            
    anchor_poses = model['anchor_poses']

    print(anchor_poses.shape)
    K = anchor_poses.shape[0]
    njts = anchor_poses.shape[1]//5 # 5 = 2D + 3D
    
    if imagename[-4:] == '.mp4':
        img_output_list = []
        video_reader = cv2.VideoCapture(imagename)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        #for i in tqdm(range(nb_frames-1)):
        for i in tqdm(range(100,110)):
            _, image = video_reader.read()
            # image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            img_output_list += [(image, None)]

        video_reader.release()

    else:
        image=cv2.imread(imagename)
        # image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        # image = np.asarray(Image.open(imagename))
        img_output_list = [(image, None)]

    print(len(img_output_list))    
    
    output = {}
    output['frames'] = []

    # run lcrnet on a list of images
    print("Pose detection starting")
    res = detect_pose( img_output_list, model['model'], model['cfg'], anchor_poses, njts, gpuid=gpuid)
    print("Pose detection finished")
    for i,(image,_) in enumerate(img_output_list): # for each image
        
        output['frames'] += [[]]


        resolution = image.shape[:2]

        # perform postprocessing
        print('postprocessing (PPI) on image ', str(i))
        detections = LCRNet_PPI(res[i], K, resolution, J=njts, **model['ppi_params'])             
        
        # move 3d pose into scene coordinates
        print('3D scene coordinates regression on image ', str(i))
        for detection in detections:
          det = {}
          det['cumscore'] = detection['cumscore'].item()
          det['pose2d'] = detection['pose2d'].tolist()
          det['pose3d'] = detection['pose3d'].tolist()
          output['frames'][i].append(det)
	
    return output 

def model_load():
    return model

#def image_interface(model, image):

   
def handle_video(videofile):
    #'InTheWild-ResNet50'
    pose_pre=get_pose(videofile, 'InTheWild-ResNet50', 1)

    if not len(videofile):
        raise IOError('Error: must contain --video')


    kpts = []
    for i in range(len(pose_pre['frames'])):
        if pose_pre['frames'][i]==[]:
            kpt0=[0]*26
            
        else:    
            kpt0 = pose_pre['frames'][i][0]['pose2d']
        kpt=[[kpt0[lcrnet[j]], kpt0[lcrnet[j]+13]] for j in coco]
        kpts.append(np.array(kpt))
       
    filename = os.path.basename(videofile).split('.')[0]
    name = filename + '_LCRNet2d.npz'
    kpts = np.array(kpts).astype(np.float32)
    print('kpts npz save in ', name)
    np.savez_compressed(name, kpts=kpts)
    return kpts

if __name__ == "__main__":
    handle_video()
