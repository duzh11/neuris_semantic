sleep 0
echo "Evaluation Start!!!"

# required to change exp_name
python exp_evaluation.py --mode eval_3D_mesh_TSDF --exp_name igrsem/SAMsvcon_igrlabel_detach_autodl
python exp_evaluation.py --mode eval_chamfer --exp_name igrsem/SAMsvcon_igrlabel_detach_autodl
python exp_evaluation.py --mode eval_mesh_2D_metrices --exp_name igrsem/SAMsvcon_igrlabel_detach_autodl
python exp_evaluation.py --mode eval_semantic_2D --exp_name igrsem/SAMsvcon_igrlabel_detach_autodl
python exp_evaluation.py --mode eval_semantic_3D --exp_name igrsem/SAMsvcon_igrlabel_detach_autodl

echo "Rendering vedio"
python ./render/render.py igrsem/SAMsvcon_igrlabel_detach_autodl