# Video_3D_Pose_Estimation
3D pose estimation for video : LCRNet2D+VideoPose3D

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
