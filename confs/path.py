import os

BIN_DIR = "/home/ethan/Research/NeuRIS"
DIR_MVG_BUILD = BIN_DIR + "/openMVG_Build"
DIR_MVS_BUILD = BIN_DIR + "/openMVS_build"

# normal path
dir_snu_code = '/home/du/Proj/3Dv_Reconstruction/NeuRIS/Data/path/snucode/surface_normal_uncertainty' # directory of code
path_snu_pth = '/home/du/Proj/3Dv_Reconstruction/NeuRIS/Data/path/snucode_ckpt/scannet_neuris_retrain.pt'
assert os.path.exists(path_snu_pth)

dir_tiltedsn_code = '/home/du/Proj/3Dv_Reconstruction/NeuRIS/Data/path/tiltedsn_code'
dir_tiltedsn_ckpt = '/home/du/Proj/3Dv_Reconstruction/NeuRIS/Data/path/tiltedsn_ckpt' # directory of pretrained model
path_tiltedsn_pth_pfpn = f"{dir_tiltedsn_ckpt}/PFPN_SR_full/model-best.ckpt"
path_tiltedsn_pth_sr = f"{dir_tiltedsn_ckpt}/SR_only/model-latest.ckpt"
assert os.path.exists(path_tiltedsn_pth_sr)

# update training/test split
names_scenes_neuris_remove = ['scene0009_01','scene0050_00','scene0084_00','scene0580_00','scene0616_00','scene0625_00']
names_scenes_manhattansdf_remove = []
lis_name_scenes_remove = names_scenes_neuris_remove + names_scenes_manhattansdf_remove

lis_name_scenes = ['scene0378_00']


    