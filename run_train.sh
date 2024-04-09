sleep 0
echo "Train Start!!!"

python ./exp_runner.py --mode train --conf ./confs/train/SPP_igrlabel.conf --server lab --gpu 0 --scene_name scene0616_00
python ./exp_runner.py --mode train --conf ./confs/train/ce_igrlabel.conf --server lab --gpu 0 --scene_name scene0616_00