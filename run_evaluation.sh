sleep 0
echo "Evaluation Start!!!"

# required to change exp_name
python exp_evaluation.py --mode eval_3D_mesh_TSDF --exp_name deeplab_ce/deepabSAMNum_ce_stop
python exp_evaluation.py --mode eval_chamfer --exp_name deeplab_ce/deepabSAMNum_ce_stop
python exp_evaluation.py --mode eval_mesh_2D_metrices --exp_name deeplab_ce/deepabSAMNum_ce_stop
python exp_evaluation.py --mode eval_semantic --exp_name deeplab_ce/deepabSAMNum_ce_stop

echo "Rendering vedio"
python ./render/render.py deeplab_ce/deepabSAMNum_ce_stop