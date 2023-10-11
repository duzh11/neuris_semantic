echo "Evaluation Start!!!"

# required to change exp_name
python exp_evaluation.py --mode eval_3D_mesh_TSDF --exp_name test/test2
python exp_evaluation.py --mode eval_chamfer --exp_name test/test2
python exp_evaluation.py --mode eval_mesh_2D_metrices --exp_name test/test2
python exp_evaluation.py --mode eval_semantic --exp_name test/test2

echo "Rendering vedio"
python ./render/render.py test/test2
