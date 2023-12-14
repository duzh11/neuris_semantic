sleep 0
echo "Train Start!!!"

python ./exp_runner.py --mode train --conf ./confs/train/train.conf --server lab --scene_name scene0050_00
python ./exp_runner.py --mode test --conf ./confs/train/train.conf --server lab --scene_name scene0050_00 --is_continue
