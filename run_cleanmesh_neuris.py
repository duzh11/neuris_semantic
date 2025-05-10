import os
import subprocess
import time

from confs.path import lis_name_scenes

### NeuRIS
method_name_lis = ['deeplab_ce/ce_a_a',  'deeplab_ce/ce_stop_a', \
                   'deeplab_svSAM/num_celoss_a', 'deeplab_svSPP/num_celossweight_0.3_a', \
                   'deeplab_igrsem/ce_stop_igrlabel_mixed', 'deeplab_igrsem/SAMsvcon_igrcons_mixed', \
                   'deeplab_igrsem/SAMsvcon_igrlabel_weightdecayg_mixed', 'deeplab_igrsem/SPPsvconweight0.3_igrlabel_mixed']

method_name_lis += ['Mask2Formera_ce/ce',  'Mask2Formera_ce/ce_stop', \
                   'Mask2Formera_svSAM/num_celossewight_0.3', 'Mask2Formera_svSPP/num_celossewight_0.3', \
                   'Mask2Formera_igrsem/SPPsvconweight0.3_igrlabel', 'Mask2Formera_igrsem/SAMsvconweight0.3_igrlabel']

exp_dir = '/home/du/Proj/3Dv_Reconstruction/NeRF-Reconstruction/MOSE/exps/indoor/neus'

for method_name in method_name_lis:
    for scene in lis_name_scenes:
        scene_exp_dir = os.path.join(exp_dir, method_name, scene)
        path_conf_source = os.path.join(scene_exp_dir, 'recording', 'config.conf')
        path_conf_target = './confs/train/train.conf'
        
        with open(path_conf_source, 'r') as file:
            lines = file.readlines()
        lines[3] = f'    exp_name = {method_name}\n'
        with open(path_conf_target, 'w') as file:
            file.writelines(lines)

        command = (f'python ./exp_runner.py --mode sem_from_mesh --conf {path_conf_target} --gpu 0 --scene_name {scene} --is_continue')
        subprocess.run(command, shell=True, text=True)
        time.sleep(5)
