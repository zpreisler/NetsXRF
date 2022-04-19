import os
from subprocess import Popen

for i in range(0,1):
    for j in range(3,4):
        cmd2run = 'python ../train.py config_cuda_windows.yaml -n ResNet1_'+str(i)+'_'+str(j) + ' -w ../src/ResNet1_warmed.yaml'
        print(cmd2run)
        p = Popen(cmd2run)
        p.wait()
