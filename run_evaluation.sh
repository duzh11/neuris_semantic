sleep 0
echo "Evaluation Start!!!"

# required to change exp_name
python exp_evaluation.py --mode eval_3D_mesh_TSDF --exp_name SAM_ce_sv/num_celoss_viewALNeighbourviewAL_a
python exp_evaluation.py --mode eval_chamfer --exp_name SAM_ce_sv/num_celoss_viewALNeighbourviewAL_a
python exp_evaluation.py --mode eval_mesh_2D_metrices --exp_name SAM_ce_sv/num_celoss_viewALNeighbourviewAL_a
python exp_evaluation.py --mode eval_semantic_2D --exp_name SAM_ce_sv/num_celoss_viewALNeighbourviewAL_a

echo "Rendering vedio"
python ./render/render.py SAM_ce_sv/num_celoss_viewALNeighbourviewAL_a