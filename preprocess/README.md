# Data preparation
关于pre_trained normal predicting network
- 官方SNU使用的应该是scannet-frames的train/test划分
- TiltedSN应该是从scannet中按10%抽取，官方是使用scanent的train/test划分
作者这里去除了confs/path的几个场景

```
python exp_preprocess.py --data_type scannet
```



Data organization:
```
<scene_name>
|-- cameras_sphere.npz   # camera parameters
|-- image
    |--train
        |-- 0000.png        # target image for each view
        |-- 0001.png
    |--test
    ...
|-- depth
    |--train
        |-- 0000.png        # target depth for each view
        |-- 0001.png
    |--test
    ...
|-- depth_vis
    |--train
        |-- 0000.png        # target depth for each view
        |-- 0001.png
    |--test
    ...
|-- pose
    |--train
        |-- 0000.txt        # camera pose for each view
        |-- 0001.txt
    |--test
    ...
|-- normal
    |--train
        |-- pred_normal
            |-- 0000.npz        # predicted normal for each view
            |-- 0001.npz
    ...
|-- semantic
    |--train
        |-- semantic_GT     # target GT semantic for each view
            |-- 0000.png
        |-- semantic_GT_vis
            |-- 0000.png
        |-- predicted semantic         # target predicted semantics for each view
            |-- 0000.png
        |-- predicted semantic_vis
            |-- 0000.png  
        |-- predicted semantic_logits # target predicted logits for each view
            |-- 0000.npz   
    ...
|-- grids
    |-- instance     # target instance for each view
        |-- 0000.png
    |-- instance_vis
        |-- 0000.png
    |-- spp_seg     # target superpixel segments for each view
        |-- 0000.png
    |-- spp_seg_vis
        |-- 0000.png     
    ...  
|-- xxx.ply		# GT mesh or point cloud from MVS
|-- xxx.labels.ply # GT semantic mesh
|-- trans_n2w.txt       # transformation matrix from normalized coordinates to world coordinates
```