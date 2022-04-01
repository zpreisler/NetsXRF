import os
from subprocess import Popen

for i in range(1,6):
    for j in range(1,4):
        cmd2run = 'python ../train.py config_cuda_windows.yaml -n ResNet1_'+str(i)+'_'+str(j)
        print(cmd2run)
        p = Popen(cmd2run)
        p.wait()
