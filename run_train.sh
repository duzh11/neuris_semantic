sleep 0
echo "Train Start!!!"

python ./exp_runner.py --mode train --conf ./confs/neuris_server.conf --server local --gpu 0 --scene_name scene0616_00
python ./exp_runner.py --mode test --conf ./confs/neuris_server.conf --server local --gpu 0 --scene_name scene0616_00


