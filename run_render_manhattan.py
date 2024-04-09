import os
import subprocess
import time

### NeuRIS
# method_name_lis = ['deeplab3sigmoid', 'deeplab40retrain', 'deeplab40detach', \
#                    'Mask2Formera3sigmoid', 'Mask2Formera40retrain', 'Mask2Formera40detach']
method_name_lis = ['deeplab3sigmoid']
for method_name in method_name_lis:
    command = (f'python ./render_copy/render.py {method_name}')
    subprocess.run(command, shell=True, text=True)