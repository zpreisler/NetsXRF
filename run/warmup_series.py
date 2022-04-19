import os
from subprocess import Popen

def warmup(cmd:str):
    p = Popen(cmd2run)
    p.wait()

if __name__=='__main__':
    cmd2run = 'python ../warmup.py config_warmup_nn.yaml -n N_ElementCNN_1'
    print(cmd2run)