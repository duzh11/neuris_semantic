echo "Evaluation Start!!!"

# required to change exp_name
python exp_evaluation.py --mode eval_3D_mesh_TSDF --exp_name deeplab_ce_sv/uncertainty_exp_prob_loss1
python exp_evaluation.py --mode eval_chamfer --exp_name deeplab_ce_sv/uncertainty_exp_prob_loss1
python exp_evaluation.py --mode eval_mesh_2D_metrices --exp_name deeplab_ce_sv/uncertainty_exp_prob_loss1
python exp_evaluation.py --mode eval_semantic --exp_name deeplab_ce_sv/uncertainty_exp_prob_loss1

echo "Rendering vedio"
python ./render/render.py deeplab_ce_sv/uncertainty_exp_prob_loss1
