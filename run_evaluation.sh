sleep 0
echo "Evaluation Start!!!"

# Method:
# - Neuris*: 
#       - deeplab_ce/ce_stop_final
#       - Mask2Formera_ce/ce_stop
# - Mose: 
#       - deeplab_igrsem/SAMsvcon_igrlabel_weightdecayg_final, SPPsvconweight0.3_igrlabel_final
#       - Mask2Formera_igrsem/SAMsvconweight0.3_igrlabel, SPPsvconweight0.3_igrlabel

# required to change exp_name
# python exp_evaluation.py --mode eval_3D_mesh_TSDF --exp_name xx/xxxx
# python exp_evaluation.py --mode eval_chamfer --exp_name xx/xxxx
# python exp_evaluation.py --mode eval_mesh_2D_metrices --exp_name xx/xxxx
# python exp_evaluation.py --mode eval_semantic_2D --exp_name xx/xxxx
# python exp_evaluation.py --mode eval_semantic_3D --exp_name xx/xxxx

# echo "Rendering vedio"
# python ./render/render.py xx/xxxx

python exp_evaluation.py --mode eval_3D_mesh_TSDF --exp_name Mose_review/semantic-neus
python exp_evaluation.py --mode eval_semantic_2D --exp_name Mose_review/semantic-neus
python exp_evaluation.py --mode eval_semantic_3D --exp_name Mose_review/semantic-neus
python exp_evaluation.py --mode evaluate_normal --exp_name Mose_review/semantic-neus
python exp_evaluation.py --mode evaluate_nvs --exp_name Mose_review/semantic-neus

python exp_evaluation.py --mode eval_3D_mesh_TSDF --exp_name Mose_review/semantic-neus-s
python exp_evaluation.py --mode eval_semantic_2D --exp_name Mose_review/semantic-neus-s
python exp_evaluation.py --mode eval_semantic_3D --exp_name Mose_review/semantic-neus-s
python exp_evaluation.py --mode evaluate_normal --exp_name Mose_review/semantic-neus-s
python exp_evaluation.py --mode evaluate_nvs --exp_name Mose_review/semantic-neus-s