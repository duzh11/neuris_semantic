sleep 0
echo "Evaluation Start!!!"

python exp_evaluation.py --mode eval_3D_mesh_TSDF --exp_name test
python exp_evaluation.py --mode eval_chamfer --exp_name test
python exp_evaluation.py --mode eval_mesh_2D_metrices --exp_name test
python exp_evaluation.py --mode eval_semantic --exp_name test
