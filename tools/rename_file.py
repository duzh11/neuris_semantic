import os
data_dir='../Data/dataset/indoor'
exp_dir='../exps/indoor/neus'
log=[]
i=0
metrics_eval_all = []
name_baseline='semantic_3'

dir_results_baseline = f'../exps/evaluation'

lis_name_scenes=['scene0050_00','scene0084_00','scene0580_00','scene0616_00']

for scene_name in lis_name_scenes:
    #dir
    GT_dir=os.path.join(data_dir,scene_name,name_baseline)

    render_dir=os.path.join(exp_dir,scene_name,name_baseline,'semantic_npz')

    GT_list=os.listdir(GT_dir)
    id_list=[int(os.path.splitext(frame)[0]) for frame in GT_list]
    id_list=sorted(id_list)
    
    for ii in range(len(id_list)):
        render_file_old=os.path.join(render_dir, '00160000_'+str(ii)+'_reso2.npz')
        render_file_new=os.path.join(render_dir, '00160000_'+'0'*(4-len(str(id_list[ii])))+str(id_list[ii])+'_reso2.npz')
        os.rename(render_file_old,render_file_new)

