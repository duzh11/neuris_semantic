import subprocess
import time

# 设置延迟时间（以秒为单位）
delay = 13000  # 延迟1小时（3600秒）
command = "python ./exp_runner.py --mode train --conf ./confs/neuris_server2.conf --server server8 --gpu 4 --scene_name scene0426_00 --stop_semantic_grad --semantic_mode softmax"
print(command)
# 等待指定的延迟时间
time.sleep(delay)

# 在延迟后执行的代码
print("延迟时间已结束，执行程序")


subprocess.run(command, shell=True)