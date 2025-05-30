# some snippets are borrowed from https://github.com/baegwangbin/surface_normal_uncertainty
import numpy as np
import torch
from pathlib import Path
import glob
from tqdm import tqdm
from PIL import Image
import cv2

import utils.utils_geometry as GeoUtils
import utils.utils_image as ImageUtils
import utils.utils_io as IOUtils

def compute_normal_errors_metrics(total_normal_errors):
    metrics = {
        'mean': np.average(total_normal_errors),
        'median': np.median(total_normal_errors),
        'rmse': np.sqrt(np.sum(total_normal_errors * total_normal_errors) / total_normal_errors.shape[0]),
        'a1': 100.0 * (np.sum(total_normal_errors < 5) / total_normal_errors.shape[0]),
        'a2': 100.0 * (np.sum(total_normal_errors < 7.5) / total_normal_errors.shape[0]),
        'a3': 100.0 * (np.sum(total_normal_errors < 11.25) / total_normal_errors.shape[0]),
        'a4': 100.0 * (np.sum(total_normal_errors < 22.5) / total_normal_errors.shape[0]),
        'a5': 100.0 * (np.sum(total_normal_errors < 30) / total_normal_errors.shape[0])
    }
    metrics_lis = []
    
    for metircs_name, value in metrics.items():
        metrics_lis.append(value)
    return metrics, metrics_lis

# log normal errors
def log_normal_errors(metrics, where_to_write = None, first_line = ''):
    print(first_line)
    print("mean   median   rmse   5    7.5   11.25   22.5    30")
    print("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f \n" % (
        metrics['mean'], metrics['median'], metrics['rmse'],
        metrics['a1'], metrics['a2'], metrics['a3'], metrics['a4'], metrics['a5']))

    if where_to_write is not None:
        with open(where_to_write, 'a') as f:
            f.write('%s\n' % first_line)
            f.write("mean median rmse 5 7.5 11.25 22.5 30\n")
            f.write("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n\n" % (
                metrics['mean'], metrics['median'], metrics['rmse'],
                metrics['a1'], metrics['a2'], metrics['a3'], metrics['a4'], metrics['a5']))

def calculate_normal_error(pred_norm, gt_norm, mask = None):
    if not torch.is_tensor(pred_norm):
        pred_norm = torch.from_numpy(pred_norm)
    if not torch.is_tensor(gt_norm):
        gt_norm = torch.from_numpy(gt_norm)
    prediction_error = torch.cosine_similarity(pred_norm, gt_norm, dim=1)
    prediction_error = torch.clamp(prediction_error, min=-1.0, max=1.0)
    E = torch.acos(prediction_error) * 180.0 / np.pi
    # mask = None
    if mask is not None:
        return E[mask]
    else:
        return E
    
def visualiza_normal(path, normal, extrin = None):
    if extrin is not None:
        shape = normal.shape
        normal = GeoUtils.get_world_normal(normal.reshape(-1,3), extrin).reshape(shape)
    pred_norm_rgb = ((normal + 1) * 0.5) * 255
    pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
    if path is not None:
        ImageUtils.write_image(path, pred_norm_rgb, color_space='RGB')
    return pred_norm_rgb
     
def evauate_normal(dir_normal_exp, dir_normal_pred, dir_normal_gt, dir_poses, interval = 1):
    vec_path_normal_exp = sorted(glob.glob(f'{dir_normal_exp}/*.npz'))
    vec_path_normal_pred = sorted(glob.glob(f'{dir_normal_pred}/*.npz'))
    #assert len(vec_path_normal_exp) == len(vec_path_normal_pred)
    
    target_img_size = (320, 240)
    input_width, input_height = target_img_size
    
    num_normals = len(vec_path_normal_exp)
    num_imgs_eval_gt = 0
    
    dir_normal_exp_eval = dir_normal_exp + '_eval'
    IOUtils.ensure_dir_existence(dir_normal_exp_eval)
    
    error_neus_all, error_pred_all, ratio_all = [], [], []
    for i in tqdm(range(0, num_normals, interval)):
        stem = Path(vec_path_normal_exp[i]).stem[9:13] # 0000
        idx_img = int(stem)
        
        # 2. load GT normal     
        # todo: scannetpp  
        path_normal_gt = f'{dir_normal_gt}/frame-{idx_img:06d}-normal.png'
        path_normal_mask_gt = f'{dir_normal_gt}/frame-{idx_img:06d}-orient.png'
        # path_normal_gt = f'{dir_normal_gt}/{stem}.npz'
        # path_normal_mask_gt = f'{dir_normal_gt}/{stem}_mask.npz'
        if not IOUtils.checkExistence(path_normal_gt): # or stem in ['0300', '0330']
                continue

        # print(path_normal_exp)
        # todo: scannetpp  
        normal_gt_cam = Image.open(path_normal_gt).convert("RGB").resize(size=(input_width, input_height), 
                                                                resample=Image.NEAREST)
        normal_gt_cam = ((np.array(normal_gt_cam).astype(np.float32) / 255.0) * 2.0) - 1.0
        # visualize normals in camera coordinate ('in' direction)
        path_normal_gt_cam = IOUtils.add_file_name_suffix(path_normal_gt, '_in_camera')
        visualiza_normal(path_normal_gt_cam, -normal_gt_cam) 
        # normal_gt_cam = -np.load(path_normal_gt)['normal']
        # if normal_gt_cam.shape[0] != input_height:
        #     normal_gt_cam = cv2.resize(normal_gt_cam, target_img_size, interpolation=cv2.INTER_NEAREST)

        # 1. load neus and predicted normal
        path_normal_exp = vec_path_normal_exp[i] # f'{dir_normal_exp}/00160000_{i:04d}_reso1.npz'
        normal_exp_world =  np.load(path_normal_exp)['arr_0']
        path_normal_pred  = f'{dir_normal_pred}/{stem}.npz'
        normal_pred_camera = -np.load(path_normal_pred)['arr_0']  # flip predicted camera
        if normal_pred_camera.shape[0] != input_height:
            normal_pred_camera = cv2.resize(normal_pred_camera, target_img_size, interpolation=cv2.INTER_NEAREST)

        # 2. normalize neus_world
        normal_neus_world_norm = np.linalg.norm(normal_exp_world, axis=-1, keepdims=True)
        # print(f'normal_neus_world_norm shape: {normal_neus_world_norm.shape}  {normal_exp_world.shape}')
        normal_exp_world = normal_exp_world/normal_neus_world_norm # out
        # print(f'Normalized shape: {normal_exp_world.shape}')
        # input('Continue?')

        # load_GT image
        data_mode = dir_normal_pred.split('/')[-2]
        path_img_gt  = f'{dir_normal_pred}/../../../image/{data_mode}/{stem}.png'
        img_rgb  = ImageUtils.read_image(path_img_gt, color_space='RGB')

        # 3. transform normal
        pose = np.loadtxt(f'{dir_poses}/{idx_img:04d}.txt')
        normal_pred_world = GeoUtils.get_world_normal(normal_pred_camera.reshape(-1,3), np.linalg.inv(pose))
        normal_gt_world = GeoUtils.get_world_normal(normal_gt_cam.reshape(-1,3), np.linalg.inv(pose))
        
        shape_img = normal_exp_world.shape
        # visualize normals in camera coordinate ('in' direction)
        path_normal_exp_cam = IOUtils.add_file_name_suffix(path_normal_exp, '_in_camera')[:-4] + '.png'
        img_visual_exp = visualiza_normal(path_normal_exp_cam, -normal_exp_world, pose) 
        img_visual_pred = visualiza_normal(None,  -normal_pred_world.reshape(shape_img), pose)
        img_visual_gt = visualiza_normal(None, -normal_gt_world.reshape(shape_img), pose)
        ImageUtils.write_image_lis(f'{dir_normal_exp_eval}/{stem}.png', [cv2.resize(img_rgb , target_img_size, interpolation=cv2.INTER_NEAREST), \
                                                                         img_visual_pred, img_visual_exp, img_visual_gt], color_space='RGB')
        
        # todo: scannetpp  
        mask_gt = Image.open(path_normal_mask_gt).convert("RGB").resize(size=(input_width, input_height),  resample=Image.NEAREST)           
        mask_gt = np.array(mask_gt) 
        mask_gt = np.logical_not(
                np.logical_and(
                    np.logical_and(
                        mask_gt[:, :, 0] == 127, mask_gt[:, :, 1] == 127),
                    mask_gt[:, :, 2] == 127))
        norm_valid_mask = mask_gt[:, :, np.newaxis]
        # mask_gt = np.load(path_normal_mask_gt)['mask']
        # if mask_gt.shape[0] != input_height:
        #     mask_gt = cv2.resize(mask_gt.astype(int), target_img_size, interpolation=cv2.INTER_NEAREST)
        # norm_valid_mask = mask_gt.astype(bool)[:, :, np.newaxis]

        ratio = norm_valid_mask.sum() /norm_valid_mask.size
        # cv2.imwrite('./test.png',norm_valid_mask.astype(np.float)*255 )
        ratio_all.append(ratio)
        
        error_neus = calculate_normal_error(normal_exp_world.reshape(-1,3), normal_gt_world, norm_valid_mask.reshape(-1))
        error_pred = calculate_normal_error(normal_pred_world, normal_gt_world, norm_valid_mask.reshape(-1))
        
        error_neus_all.append(error_neus)
        error_pred_all.append(error_pred)
        num_imgs_eval_gt += 1

    error_neus_all = torch.cat(error_neus_all).numpy()
    error_pred_all = torch.cat(error_pred_all).numpy()
    
    # error_neus_all = total_normal_errors.data.cpu().numpy()
    metrics_neus, metrics_neus_lis = compute_normal_errors_metrics(error_neus_all)
    metrics_pred, metrics_pred_lis = compute_normal_errors_metrics(error_pred_all)
    # print(f'Neus error: \n{metrics_neus}\nPred error: \n{metrics_pred}')
    print(f'Num imgs for evaluation: {num_imgs_eval_gt}')
    log_normal_errors(metrics_neus, first_line='metrics_neus')
    log_normal_errors(metrics_pred, first_line='metrics_pred')
    return error_neus_all, error_pred_all, num_imgs_eval_gt, metrics_neus_lis, metrics_pred_lis