sleep 0
echo "Evaluation Start!!!"

# required to change exp_name
python exp_evaluation.py --mode eval_3D_mesh_TSDF --exp_name neuris/neuris_smooth
python exp_evaluation.py --mode eval_chamfer --exp_name neuris/neuris_smooth
