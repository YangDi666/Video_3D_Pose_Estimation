# Video_3D_Pose_Estimation
3D pose estimation for video : [LCRNet2D](https://thoth.inrialpes.fr/src/LCR-Net/)+[VideoPose3D](https://github.com/facebookresearch/VideoPose3D)

![ad](https://github.com/YangDi666/LCRNet-2D-3D-Pose-Estimation/blob/master/demo/result_010817395615_0real_im_1_1.png)

## Introduction  
- [LCR-Net Multi-person 2D and 3D Pose Detection](https://thoth.inrialpes.fr/src/LCR-Net/) is realised by INRIA THOTH team and this code is for training the model.

### Quick start

```
git clone https://github.com/YangDi666/Video_3D_Pose_Estimation.git
```
#### I have 2D pose

```
python tools/lcrnet_video.py --viz-output output.gif --viz-video {path to your input video} --input-npz testvideo_lcrnet2d.npz
```
#### I don't have 2D pose
```
git clone https://github.com/YangDi666/LCRNet_2D_3D_Pose_Estimation.git
```
Move it to pose_2d/LCRNet :
```
mv LCRNet_2D_3D_Pose_Estimation pose_2d/LCRNet
```

```
python tools/lcrnet_video.py --viz-output output.gif --viz-video {path to your input video} 
```
