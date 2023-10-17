sleep 0
echo "Train Start!!!"

python ./exp_runner.py --mode train --conf ./confs/train/sv_num.conf --server lab --scene_name scene0378_00
python ./exp_runner.py --mode test --conf ./confs/train/sv_num.conf --server lab --scene_name scene0378_00 --is_continue

