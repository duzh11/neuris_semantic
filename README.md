# NeuRIS_Semantic

## Data preparation
Scene data used in NeuRIS can be downloaded from [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/jiepeng_connect_hku_hk/ElKcK1sus9pLnARZ_e9l-IcBS6cE-6w8xt34bMsvMAiuIQ?e=0z1eka) and extract the scene data into folder `../Data/dataset/indoor`. And the scene data used in [ManhattanSDF](https://github.com/zju3dv/manhattan_sdf) are also included for convenient comparisons.
The data is organized as follows:
```
<scene_name>
|-- cameras_sphere.npz   # camera parameters
|-- image
    |-- 0000.png        # target image for each view
    |-- 0001.png
    ...
|-- depth
    |-- 0000.png        # target depth for each view
    |-- 0001.png
    ...
|-- pose
    |-- 0000.txt        # camera pose for each view
    |-- 0001.txt
    ...
|-- pred_normal
    |-- 0000.npz        # predicted normal for each view
    |-- 0001.npz
    ...
|-- semantic
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

## Setup
```
conda create -n neuris python=3.8
conda activate neuris
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r neuris.txt
```

## Training

```
python ./exp_runner.py --mode train --conf ./confs/neuris_server.conf --server server2 --gpu 0 --scene_name scene0050_00
```

## Mesh extraction
```
python exp_runner.py --mode validate_mesh --conf ./confs/neuris.conf --server server2 --gpu 0 --is_continue
```
## validate image
```
python exp_runner.py --mode validate_image --conf ./confs/neuris_server.conf --server server2 --gpu 0 --scene_name scene0050_00 --is_continue
```
## Evaluation
```
python ./exp_evaluation.py --mode eval_3D_mesh_metrics --is_continue
```

## Citation

```
@article{wang2022neuris,
      	title={NeuRIS: Neural Reconstruction of Indoor Scenes Using Normal Priors}, 
      	author={Wang, Jiepeng and Wang, Peng and Long, Xiaoxiao and Theobalt, Christian and Komura, Taku and Liu, Lingjie and Wang, Wenping},
	publisher = {arXiv},
      	year={2022}
}
```