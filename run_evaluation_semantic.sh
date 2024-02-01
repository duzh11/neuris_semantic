sleep 0
echo "Evaluation Start!!!"


python exp_evaluation.py --mode eval_semantic_2D --exp_name deeplab_ce_plane/ce_stop_igrcons0.3
python exp_evaluation.py --mode eval_semantic_3D --exp_name deeplab_ce_plane/ce_stop_igrcons0.3

python exp_evaluation.py --mode eval_semantic_2D --exp_name deeplab_ce_plane/ce_stop_igrsem
python exp_evaluation.py --mode eval_semantic_3D --exp_name deeplab_ce_plane/ce_stop_igrsem