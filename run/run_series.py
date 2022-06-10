import os
from subprocess import Popen

if __name__ == '__main__':
    model = 'CNN2'
    n_channels = [8]
    n_runs = 1
    lr = 0.0025

    for n_ch in n_channels:
        for run_n in range(n_runs):
            # cmd2run = 'python ../train_for_synth.py config_synth.yaml -n synth_scales0,5-1,5_uboundonly_'+ model +str(n_ch)+'_ch_run_'+str(run_n)
            # cmd2run = 'python ../train_for_synth.py config_synth.yaml -n synth_amuleti1_' + model + '_ch_' +str(n_ch) + '_run_' + str(run_n) + \

            cmd2run = 'python ../train_for_synth.py config_synth.yaml -n synth_amuleti_newtemplatecorrected_' + model + '_ch_' +str(n_ch) + '_run_' + str(run_n) + \
                ' -c ' + str(n_ch) + \
                ' -l ' + str(lr)
            
            # cmd2run = 'python ../train.py config_cuda_andrea.yaml -n Ti_withmargaret'+ model + loss +str(n_ch)+'_ch_run_'+str(run_n)+'_downsampledonly'
            # cmd2run = 'python ../train.py config_cuda_andrea_noweights.yaml -n Ti_withmargaret'+ model + loss +str(n_ch)+'_ch_noweights_run_'+str(run_n)+'_downsampledonly'
            print(cmd2run)
            p = Popen(cmd2run)
            p.wait()