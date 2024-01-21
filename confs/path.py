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
names_scenes_neuris_remove = ['scene0009', 'scene0085', 'scene0114',
                        'scene0603', 'scene0617', 'scene0625',
                        'scene0721', 'scene0771']
names_scenes_manhattansdf_remove = ['scene0050', 'scene0084', 
                                'scene0580', 'scene0616']
lis_name_scenes_remove = names_scenes_neuris_remove + names_scenes_manhattansdf_remove

lis_name_scenes = ['scene0435_02', 'scene0474_01']
# lis_name_scenes = ['scene0435_02', 'scene0084_00']
# lis_name_scenes = ['scene0378_00', 'scene0435_02']
# lis_name_scenes += ['scene0050_00','scene0616_00'] # 简单
# lis_name_scenes += ['scene0084_00', 'scene0648_00', 'scene0474_01', 'scene0030_00'] # 难


    