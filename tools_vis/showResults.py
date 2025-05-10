import sys, os
sys.path.append(os.getcwd())
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

import utils.utils_correspondence as correspondence

def show_results(scene_name, method_name, source_id, x, y):
    data_dir = f'./Data/dataset/indoor/{scene_name}'
    depth_GT= cv2.imread(os.path.join(data_dir, 'depth/train', f'{source_id}.png'), cv2.IMREAD_UNCHANGED)[2*y, 2*x]
    path_trans_n2w = np.loadtxt(f'{data_dir}/trans_n2w.txt')
    scale=path_trans_n2w[0, 0]
    depth_GT = depth_GT/(scale*1000)

    results_dir=f'./exps/indoor/neus/{method_name}/{scene_name}/results_validate'
    file_name=f'00050000_{source_id}_reso2.npz'

    # read results
    z_results = np.load(os.path.join(results_dir, 'mid_z_vals', file_name))['arr_0']
    alpha_results = np.load(os.path.join(results_dir, 'alpha', file_name))['arr_0']
    weights_results = np.load(os.path.join(results_dir, 'weights', file_name))['arr_0']
    sdf_results = np.load(os.path.join(results_dir, 'sdf', file_name))['arr_0']
    semantic_results = np.load(os.path.join(results_dir, 'sampled_semantic', file_name))['arr_0']

    # read results of (x, y)
    t_sampler = [i+1 for i in range(0, 128)]
    z_sampler = (z_results[y,x,...]).squeeze()
    alpha_sampler = (alpha_results[y,x,...]/20).squeeze()
    weights_sampler = (weights_results[y,x,...]).squeeze()
    sdf_sampler= (sdf_results[y,x,...]).squeeze()

    depth = (z_results[y,x,...] * weights_results[y,x,...]).sum(axis=0)
    # logits and prob
    logits = (semantic_results[y,x,...] * weights_results[y,x,...]).sum(axis=0)
     
    prob = softmax(logits)
    logits_sampler = (semantic_results[y,x,...]).squeeze()
    probability_sampler = softmax(logits_sampler, axis=1)
    semantic = logits_sampler.argmax(axis=1)

    save_dir = os.path.join(results_dir, f'{source_id}_{x}_{y}')
    os.makedirs(save_dir, exist_ok=True)
    # pic for results
    plt.figure()
    plt.plot(t_sampler, alpha_sampler, marker='o', linestyle='-', color='b', label='alpha')
    plt.plot(t_sampler, weights_sampler, marker='.', linestyle='-', color='r', label='weights')
    plt.plot(t_sampler, sdf_sampler, marker='*', linestyle='-', color='g', label='sdf')

    plt.xlabel('t')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, f'results_t.png'))

    plt.figure()
    plt.plot(z_sampler, alpha_sampler, marker='o', linestyle='-', color='b', label='alpha')
    plt.plot(z_sampler, weights_sampler, marker='.', linestyle='-', color='r', label='weights')
    plt.plot(z_sampler, sdf_sampler, marker='*', linestyle='-', color='g', label='sdf')
    plt.axvline(x=depth, color='black', linestyle='--', label='render_depth')
    plt.axvline(x=depth_GT, color='orange', linestyle='--', label='GT_depth')
    plt.xlabel('t')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, f'results_z.png'))
    # pic for semantic
    plt.figure()
    t=[i for i in range(1,41)]
    plt.plot(t, logits, marker='.', linestyle='-', color='b', label='logits')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'logits.png'))

    plt.figure()
    plt.plot(t, prob, marker='*', linestyle='-', color='r', label='prob')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'prob.png'))
    # pic for prob and logits
    plt.figure()
    plt.imshow(logits_sampler.T, cmap='hot_r', vmin=0)
    plt.yticks(np.arange(0, 39, 5))
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, 'logits_sampler.png'))

    plt.figure()
    plt.imshow(probability_sampler.T, cmap='hot_r', vmin=0)
    plt.yticks(np.arange(0, 39, 5))
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, 'prob_sampler.png'))

    # pic for semantic
    plt.figure()
    plt.plot(t_sampler, semantic, marker='.', linestyle='-', color='b', label='semantic')
    plt.legend()
    plt.yticks(np.arange(0, 39, 5))
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'semantic.png'))
    # plt.show()

def main(scene_name, method_name, source_id, target_id, x, y):
    show_results(scene_name, method_name, source_id, x, y)

    target_x, target_y = correspondence.find_correspondence(scene_name, source_id, target_id, 2*x, 2*y)
    target_x, target_y=target_x//2, target_y//2
    exps_dir=f'./exps/indoor/neus/{method_name}/{scene_name}'
    semantic_source = cv2.imread(os.path.join(exps_dir, 'semantic/train/fine', f'00050000_{source_id}_reso2.png'))
    semantic_target = cv2.imread(os.path.join(exps_dir, 'semantic/train/fine', f'00050000_{target_id}_reso2.png'))
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(semantic_source[...,::-1])
    ax[0].plot(x, y, 'ro', markersize=5)
    ax[1].imshow(semantic_target[...,::-1])
    ax[1].plot(target_x, target_y, 'ro', markersize=5)
    plt.tight_layout()
    plt.savefig(os.path.join(exps_dir, 'results_validate', f'{source_id}_{x}_{y}_{target_id}_{target_x}_{target_y}.png'))
    # plt.show()

    if target_x*target_y>0:
        show_results(scene_name, method_name, target_id, target_x, target_y)

if __name__=='__main__':
    scene_name = 'scene0050_00'
    method_name_lis = ['SAM_ce_sv/num_celoss_lab']
    source_id='0080'
    target_id='0090'
    x, y= 76, 194  #(240,320)

    for method_name in method_name_lis:
        main(scene_name, method_name, source_id, target_id, x, y)