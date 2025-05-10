# MOSE

## Data preparation
Scene data used in NeuRIS can be downloaded from [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/jiepeng_connect_hku_hk/ElKcK1sus9pLnARZ_e9l-IcBS6cE-6w8xt34bMsvMAiuIQ?e=0z1eka) and extract the scene data into folder `./Data/dataset/indoor`. And the scene data used in [ManhattanSDF](https://github.com/zju3dv/manhattan_sdf) are also included for convenient comparisons.

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
## validate train image
```
python exp_runner.py --mode validate_image --conf ./confs/neuris_server.conf --server server2 --gpu 0 --scene_name scene0050_00 --is_continue
```
## validate test image
```
python exp_runner.py --mode test --conf ./confs/neuris_server.conf --server server2 --gpu 0 --scene_name scene0050_00 --is_continue
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