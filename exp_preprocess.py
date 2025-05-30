import os, argparse, logging
from datetime import datetime
import numpy as np

import preprocess.neuris_data  as neuris_data
import evaluation.EvalScanNet as EvalScanNet
from evaluation.renderer import render_depthmaps_pyrender

import utils.utils_geometry as GeoUtils
import utils.utils_image  as ImageUtils
import utils.utils_io as IOUtils
import utils.utils_normal as NormalUtils

from confs.path import lis_name_scenes
np.random.seed(42)

if __name__ == '__main__':
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    np.random.seed(42)

    FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='scannet')
    args = parser.parse_args()
    
    dataset_type = args.data_type
    
    if dataset_type == 'scannet':
        dir_root_scannet = '/home/du/Proj/Dataset/ScanNet/scans'
        dir_root_neus = '/home/du/Proj/3Dv_Reconstruction/NeRF-Reconstruction/MOSE/Data/dataset/indoor'

        for scene_name in lis_name_scenes:
            logging.info(f'***process scene: {scene_name}**\n\n')
            dir_scan = f'{dir_root_scannet}/{scene_name}'
            dir_neus = f'{dir_root_neus}/{scene_name}'
            os.makedirs(dir_neus,exist_ok=True)
            neuris_data.prepare_neuris_data_from_scannet(dir_scan, dir_neus, sample_interval=10, 
                                                b_sample = False, 
                                                b_generate_neus_data = False,
                                                b_pred_normal = True, 
                                                b_detect_planes = True) 
    if dataset_type == 'scannetpp':
        dir_root_scannetpp = '/home/du/Proj/Dataset/ScanNetpp/scans'
        dir_root_neus = '/home/du/Proj/3Dv_Reconstruction/NeRF-Reconstruction/MOSE/Data/dataset/indoor'

        for scene_name in lis_name_scenes:
            logging.info(f'***process scene: {scene_name}**\n\n')
            dir_scan = dir_root_scannetpp
            dir_neus = f'{dir_root_neus}/{scene_name}'
            os.makedirs(dir_neus,exist_ok=True)
            neuris_data.prepare_neuris_data_from_scannetpp(dir_scan, dir_neus, scene_name,
                                                            camera_device='dslr',
                                                            b_sample = True,
                                                            b_generate_neus_data = True, compute_normal = True, 
                                                            b_pred_normal = True) 

    if dataset_type == 'private':
        # example of processing iPhone video
        # put a video under folder tmp_sfm_mvs or put your images under tmp_sfm_mvs/images
        dir_neuris = '/home/ethan/Desktop/test_sfm'
        
        dir_neuris = os.path.abspath(dir_neuris)
        dir_sfm_mvs = os.path.abspath(f'{dir_neuris}/tmp_sfm_mvs')
        
        crop_image = True
        original_size_img = (1920, 1080)
        cropped_size_img = (1360, 1020) # cropped images for normal estimation
        reso_level = 1          
        
        # split video into frames and sample images
        b_split_images = True
        path_video = f'{dir_sfm_mvs}/video.MOV'
        dir_split = f'{dir_sfm_mvs}/images_split'
        dir_mvs_sample = f'{dir_sfm_mvs}/images' # for mvs reconstruction
        dir_neuris_sample = f'{dir_sfm_mvs}/images_calibrated' # remove uncalbrated images
        dir_neuris_sample_cropped = f'{dir_neuris}/image'
        
        if b_split_images:
            ImageUtils.split_video_to_frames(path_video, dir_split)     

        # sample images
        b_sample = True
        sample_interval = 10
        if b_sample:
            rename_mode = 'order_04d'
            ext_source = '.png' 
            ext_target = '.png'
            ImageUtils.convert_images_type(dir_split, dir_mvs_sample, rename_mode, 
                                            target_img_size = None, ext_source = ext_source, ext_target =ext_target, 
                                            sample_interval = sample_interval)
        
        # SfM camera calibration
        b_sfm = True
        if b_sfm:
            os.system(f'python ./preprocess/sfm_mvs.py --dir_mvs {dir_sfm_mvs} --image_width {original_size_img[0]} --image_height {original_size_img[1]} --reso_level {reso_level}')
            
        b_crop_image = True
        if crop_image:
            neuris_data.crop_images_neuris(dir_imgs = dir_neuris_sample, 
                                dir_imgs_crop = dir_neuris_sample_cropped, 
                                path_intrin = f'{dir_sfm_mvs}/intrinsics.txt', 
                                path_intrin_crop = f'{dir_neuris}/intrinsics.txt', 
                                crop_size = cropped_size_img)

            # crop depth
            if IOUtils.checkExistence(f'{dir_sfm_mvs}/depth_calibrated'):
                ImageUtils.crop_images(dir_images_origin = f'{dir_sfm_mvs}/depth_calibrated',
                                            dir_images_crop = f'{dir_neuris}/depth', 
                                            crop_size = cropped_size_img, 
                                            img_ext = '.npy')
        
        b_prepare_neus = True
        if b_prepare_neus:
            neuris_data.prepare_neuris_data_from_private_data(dir_neuris, cropped_size_img, 
                                                            b_generate_neus_data = True,
                                                                b_pred_normal = True, 
                                                                b_detect_planes = False)
            
    print('Done')
