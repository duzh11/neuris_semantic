sleep 0
echo "Train Start!!!"

python ./exp_runner.py --mode train --conf ./confs/train/train.conf --server server2 --gpu 0 --scene_name scene0050_00

