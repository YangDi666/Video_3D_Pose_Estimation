# Video_3D_Pose_Estimation
3D pose estimation for video : [LCRNet2D](https://thoth.inrialpes.fr/src/LCR-Net/)+[VideoPose3D](https://github.com/facebookresearch/VideoPose3D)

![ad](https://github.com/YangDi666/Video_3D_Pose_Estimation/blob/master/output.gif)


### Quick start

```
git clone https://github.com/YangDi666/Video_3D_Pose_Estimation.git
```

Download the pretrained videopose model
```
mkdir checkpoint
cd checkpoint
```
videopose model [address](https://dl.fbaipublicfiles.com/video-pose-3d/cpn-pt-243.bin)

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
