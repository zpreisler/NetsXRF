#!/usr/bin/env python
from matplotlib import pyplot as plt
import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler

from sklearn.model_selection import train_test_split

from src.nets import ResNet1,CNN1,N_ElementCNN_1
from src.dataset import SpectraDataset

from XRDXRFutils import DataXRF

from numpy import log,array,asarray,save,zeros,ones,vstack
from argparse import ArgumentParser
from pathlib import Path
from glob import glob
from time import time

import h5py,yaml
import gc
import os
from  ChemElementRegressorDataset_for_evaluate import ChemElementRegressorDataset_for_evaluate

def main():

    """
    Read configs
    """
    parser = ArgumentParser()

    parser.add_argument('config_file')
    parser.add_argument('-n','--name',default=None)
    parser.add_argument('-l','--learning_rate',default=None,type=float)
    parser.add_argument('-m','--momentum',default=0.0,type=float)
    parser.add_argument('-c','--channels',default=None,type=int)
    parser.add_argument('-s','--epoch_size',default=65536,type=int)
    parser.add_argument('-w','--warm_up',default=None,type=str)

    args = parser.parse_args()

    with open(args.config_file,'r') as file:
        config = yaml.load(file,Loader=yaml.FullLoader)

    print('config:',config)
    print('args:',args)

    for k,v in args.__dict__.items():
        if v: config[k] = v

    print('updated_config:',config)

    try:
        os.mkdir(config['name'])
    except OSError as error:
        print(error)

    with open(config['name'] + '/config.yaml', 'w') as file:
        yaml.dump(config,file)

    """
    Read datasets
    """

    eval_datasets = []
    for file in config['eval_data']:
        dataXRF = DataXRF().load_h5(file)

        data = dataXRF.data
        data = torch.from_numpy(data).reshape(-1,1,data.shape[-1]).float()

        eval_dataset = ChemElementRegressorDataset_for_evaluate(data)
        eval_datasets += [eval_dataset]


    #train_data = SpectraDataset().load_h5(config['data'])

    #print(weights.shape)

    evals = []
    for eval_dataset in eval_datasets:
        evaluate = DataLoader(eval_dataset,
                batch_size = config['batch_size'],
                shuffle = False,
                drop_last = False,
                pin_memory = True)
        evals += [evaluate]

    """
    Define model
    """
    model = globals()[config['model']](channels = config['channels'])

    criterion = getattr(nn,config['loss'])(reduction = 'none')

    """
    Define device
    """

    if torch.cuda.is_available():
        device = torch.device(config['device'])
    else:
        device = torch.device('cpu')
    torch.set_num_threads(config['num_threads'])

    model.train()
    model.to(device)

    current_epoch = 1
    checkpoints = sorted(glob(config['name'] + '/*.pth'))
    print(checkpoints)
    if checkpoints:
        
        print('Loading Last Checkpoint:',checkpoints[-1])
        checkpoint = torch.load(checkpoints[-1])

        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        current_epoch = checkpoint['epoch'] + 1


    time0 = time()

    model.eval()

    images = []
    for evaluate in evals:
        image = []

        for i,batch in enumerate(evaluate):

            data = batch

            data = data.to(device)
            data = torch.squeeze(data, 1)
            outputs = model(data)
            image += [outputs.cpu().detach().numpy()]

        image = vstack(image)
        image = image.reshape((188,140, 5))
        print('\n',image.shape)
        images += [image]
        for img in range(5):
            plt.figure()
            plt.imshow(image[:,:,img])

    plt.show()

if __name__ == '__main__':
    main()
