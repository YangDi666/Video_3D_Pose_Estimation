""" LCR-Net: Localization-Classification-Regression for Human Pose
Copyright (C) 2017 Gregory Rogez & Philippe Weinzaepfel

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>"""

from __future__ import print_function
import os, sys
import numpy as np
from scipy.io import savemat
import h5py 
import cv2


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader
sys.path.insert(0, os.path.join( os.path.dirname(__file__),"LCRNet_2D_3D_Pose_Estimation/lib"))
from core.config import cfg, cfg_from_file, assert_and_infer_cfg, _merge_a_into_b
import utils.net as net_utils
import utils.blob as blob_utils
import nn as mynn

from modeling.model_builder import LCRNet


NT = 5       # 2D + 3D 


def _get_blobs(im, rois, target_scale, target_max_size):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale, blobs['im_info'] = \
        blob_utils.get_image_blob(im, target_scale, target_max_size)
    if rois is not None:
        blobs['rois'] = _get_rois_blob(rois, im_scale)
    return blobs, im_scale

def detect_pose( img_output_list, ckpt, cfg_dict, anchor_poses, njts, gpuid=-1):
    """
    detect poses in a list of image
    img_output_list: list of couple (path_to_image, path_to_outputfile)
    ckpt_fname: path to the model weights
    cfg_dict: directory of configuration
    anchor_poses: file containing the anchor_poses or directly the anchor poses
    njts: number of joints in the model
    gpuid: -1 for using cpu mode, otherwise device_id
    """
        
    # load the anchor poses and the network
    if gpuid>=0:
        assert torch.cuda.is_available(), "You should launch the script on cpu if cuda is not available"
        torch.device('cuda:0')
    else:
        torch.device('cpu')

    # load config and network
    print('loading the model')
    _merge_a_into_b(cfg_dict, cfg)
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False
    cfg.CUDA = gpuid>=0
    assert_and_infer_cfg()
    model = LCRNet(njts)
    if cfg.CUDA: model.cuda()
    net_utils.load_ckpt(model, ckpt)
    model = mynn.DataParallel(model, cpu_keywords=['im_info', 'roidb'], minibatch=True, device_ids=[0])
    model.eval()

    
    output = []
    # iterate over image 
    for i, img_output in enumerate(img_output_list):
        print('Processing frame '+str(i))
        im = img_output[0]
        outputname = img_output[1]
        # load the images and prepare the blob
        # im = cv2.imread( imgname )        
        inputs, im_scale = _get_blobs(im, None, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE) # prepare blobs    

        # forward
        if cfg.FPN.MULTILEVEL_ROIS and not cfg.MODEL.FASTER_RCNN: 
            _add_multilevel_rois_for_test(inputs, 'rois') # Add multi-level rois for FPN
        if cfg.PYTORCH_VERSION_LESS_THAN_040: # forward
            inputs['data'] = [Variable(torch.from_numpy(inputs['data']), volatile=True)]
            inputs['im_info'] = [Variable(torch.from_numpy(inputs['im_info']), volatile=True)]
            return_dict = model(**inputs)
        else:
            inputs['data'] = [torch.from_numpy(inputs['data'])]
            inputs['im_info'] = [torch.from_numpy(inputs['im_info'])]
            with torch.no_grad():
              return_dict = model(**inputs)
        # get boxes
        rois = return_dict['rois'].data.cpu().numpy()
        boxes = rois[:, 1:5] / im_scale
        # get scores
        scores = return_dict['cls_score'].data.cpu().numpy().squeeze()
        scores = scores.reshape([-1, scores.shape[-1]]) # In case there is 1 proposal
        # get pose_deltas
        pose_deltas = return_dict['pose_pred'].data.cpu().numpy().squeeze()
        # project poses on boxes
        boxes_size = boxes[:,2:4]-boxes[:,0:2]
        offset = np.concatenate( ( boxes[:,:2], np.zeros((boxes.shape[0],3),dtype=np.float32)), axis=1) # x,y top-left corner for each box
        scale = np.concatenate( ( boxes_size[:,:2], np.ones((boxes.shape[0],3),dtype=np.float32)), axis=1) # width, height for each box
        offset_poses = np.tile( np.concatenate( [np.tile( offset[:,k:k+1], (1,njts)) for k in range(NT)], axis=1), (1,anchor_poses.shape[0])) # x,y top-left corner for each pose 
        scale_poses = np.tile( np.concatenate( [np.tile( scale[:,k:k+1], (1,njts)) for k in range(NT)], axis=1), (1,anchor_poses.shape[0]))
            # x- y- scale for each pose 
        pred_poses = offset_poses + np.tile( anchor_poses.reshape(1,-1), (boxes.shape[0],1) ) * scale_poses # put anchor poses into the boxes
        pred_poses += scale_poses * pose_deltas[:,njts*NT:] # apply regression (do not consider the one for the background class)
        
        # we save only the poses with score over th with at minimum 500 ones 
        th = 0.1/(scores.shape[1]-1)
        Nmin = min(500, scores[:,1:].size-1)
        if np.sum( scores[:,1:]>th ) < Nmin: # set thresholds to keep at least Nmin boxes
            th = - np.sort( -scores[:,1:].ravel() )[Nmin-1]
        where = list(zip(*np.where(scores[:,1:]>=th ))) # which one to save 
        nPP = len(where) # number to save 
        regpose2d = np.empty((nPP,njts*2), dtype=np.float32) # regressed 2D pose 
        regpose3d = np.empty((nPP,njts*3), dtype=np.float32) # regressed 3D pose 
        regscore = np.empty((nPP,1), dtype=np.float32) # score of the regressed pose 
        regprop = np.empty((nPP,1), dtype=np.float32) # index of the proposal among the candidate boxes 
        regclass = np.empty((nPP,1), dtype=np.float32) # index of the anchor pose class 
        for ii, (i,j) in enumerate(where):
            regpose2d[ii,:] = pred_poses[i, j*njts*5:j*njts*5+njts*2]
            regpose3d[ii,:] = pred_poses[i, j*njts*5+njts*2:j*njts*5+njts*5]
            regscore[ii,0] = scores[i,1+j]
            regprop[ii,0] = i+1
            regclass[ii,0] = j+1
        tosave = {'regpose2d': regpose2d, 
                  'regpose3d': regpose3d, 
                  'regscore': regscore, 
                  'regprop': regprop, 
                  'regclass': regclass, 
                  'rois': boxes,
                 }
        output.append( tosave )
        if outputname is not None:
            outputdir = os.path.dirname(outputname)
            if len(outputdir)>0 and not os.path.isdir(outputdir): os.system('mkdir -p '+os.path.dirname(outputname))
            savemat(outputname, tosave, do_compression=True)
            
    return output
        
