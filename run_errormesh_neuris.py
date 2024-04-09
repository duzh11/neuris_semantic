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

for method_name in method_name_lis:
    command = (f'python exp_evaluation.py --mode eval_3D_mesh_TSDF --exp_name {method_name}')
    subprocess.run(command, shell=True, text=True)