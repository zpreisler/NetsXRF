#!/usr/bin/env python
import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler

from sklearn.model_selection import train_test_split

from src.nets import ResNet0,CNN1,CNN2
from src.dataset import SpectraDataset, SynthSpectraDataset

from XRDXRFutils import DataXRF, SyntheticDataXRF

from numpy import log,array,asarray,save,zeros,ones,vstack
from argparse import ArgumentParser
from pathlib import Path
from glob import glob
from time import time

import h5py,yaml
import gc
import os

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
    parser.add_argument('-k','--kernel_size',default=None,type=int)
    parser.add_argument('-s','--epoch_size',default=32768,type=int)

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

    train_data = []
    test_data = []

    train_labels = []
    test_labels = []

    print(config['data'])
    for file in config['data']:
        print('Reading:',file)

        dataXRF = SyntheticDataXRF().load_h5(file)
        print('selected labels:',dataXRF.metadata['labels'], len(dataXRF.metadata['labels']))
        
        dataXRF = dataXRF.select_labels(config['labels'])
        print('labels shape:', dataXRF.labels.shape)

        data = dataXRF.data.astype(float)
        labels = dataXRF.labels.astype(float)

        data = torch.from_numpy(data).reshape(-1,1,data.shape[-1]).float()
        labels = torch.from_numpy(labels).reshape(-1,labels.shape[-1]).float()

        print(data.shape,labels.shape)

        _train_data, _test_data, _train_labels, _test_labels = train_test_split(data,labels,test_size = 1100,random_state = 93719)

        train_data += [_train_data]
        test_data += [_test_data]

        train_labels += [_train_labels]
        test_labels += [_test_labels]


    eval_datasets = []
    for file in config['eval_data']:
        dataXRF = DataXRF().load_h5(file)
        dataXRF = dataXRF.select_labels(config['labels'])
        print('labels shape:', dataXRF.labels.shape)

        data = dataXRF.data
        labels = dataXRF.labels.astype(float)

        print("Eval:",file,labels.shape)

        data = torch.from_numpy(data).reshape(-1,1,data.shape[-1]).float()
        labels = torch.from_numpy(labels).reshape(-1,labels.shape[-1]).float()

        eval_dataset = SpectraDataset(data,labels)
        eval_dataset.shape = dataXRF.labels.shape
        eval_datasets += [eval_dataset]

    train_data = torch.cat(train_data)
    test_data = torch.cat(test_data)

    train_labels = torch.cat(train_labels)
    test_labels = torch.cat(test_labels)

    print("Train data:",train_data.shape,train_data.mean())
    print("Test  data:",test_data.shape,test_data.mean())

    print("Labels means:",train_labels.mean(axis=0))

    train_dataset = SynthSpectraDataset(train_data,train_labels, scales=config['scales'])
    test_dataset = SpectraDataset(test_data,test_labels)

    # sampler = WeightedRandomSampler(train_weights,config['epoch_size'])

    train = DataLoader(train_dataset,
            batch_size = config['batch_size'],
            # sampler = sampler,
            shuffle = False,
            drop_last = True,
            pin_memory = True)

    test = DataLoader(test_dataset,
            batch_size = config['batch_size'],
            shuffle = False,
            drop_last = True,
            pin_memory = True)

    evals = []
    for eval_dataset in eval_datasets:
        evaluate = DataLoader(eval_dataset,
                batch_size = config['batch_size'],
                shuffle = False,
                drop_last = False,
                pin_memory = True)
        evaluate.shape = eval_dataset.shape
        evals += [evaluate]

    """
    Define model
    """

    model = globals()[config['model']](channels = config['channels'], n_outputs=len(config['labels']))

    #optimizer = optim.SGD(model.parameters(), lr = config['learning_rate'])
    optimizer = getattr(optim,config['optimizer'])(model.parameters(), lr = config['learning_rate'])
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

    if checkpoints:

        print('Loading Last Checkpoint:',checkpoints[-1])
        checkpoint = torch.load(checkpoints[-1])

        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_epoch = checkpoint['epoch'] + 1

    #rescale = torch.Tensor([460,215,200,70,35]).to(device)
    # rescale = torch.Tensor([182.2652, 307.7705, 179.8532, 740.2400, 772.9026, 752.8382, 714.0045, 311.4732, 366.1621, 894.7768]).to(device)
    rescale = torch.Tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).to(device)


    loss_history = []
    test_loss_history = []

    time0 = time()
    
    for epoch in range(current_epoch,config['n_epochs']):
        print('\nEpoch: %d/%d'%(epoch,config['n_epochs']))

        model.train()
        for i,batch in enumerate(train):

            data,labels = batch

            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)

            loss = criterion(outputs/rescale,labels/rescale)
            loss = loss.mean(0)

            loss_history += [loss.tolist()]

            loss = loss.mean()
            print('loss:', loss, end='\r')

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # model.eval()
        # for i,batch in enumerate(test):

        #     data,labels = batch

        #     data = data.to(device)
        #     labels = labels.to(device)

        #     outputs = model(data)

        #     loss = criterion(outputs/rescale,labels/rescale)
        #     loss = loss.mean(0)

        #     test_loss_history += [loss.tolist()]

        # images = []
        # for evaluate in evals:
        #     image = []

        #     for i,batch in enumerate(evaluate):

        #         data,labels = batch

        #         data = data.to(device)
        #         labels = labels.to(device)

        #         outputs = model(data)
        #         image += [outputs.cpu().detach().numpy()]

        #     image = vstack(image)
        #     print('\n',image.shape)
        #     image = image.reshape(evaluate.shape)
        #     images += [image]

        if epoch % 10 == 0 or epoch==int(config['n_epochs'])-1:

            print()
            path = config['name'] + '/%04d.pth'%epoch
            print('Saving:',path)
            loss_history = asarray(loss_history)
            test_loss_history = asarray(test_loss_history)

            m_time = time() - time0
            print(loss_history.mean(),test_loss_history.mean())
            print('time:',m_time)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'loss': loss_history,
                'test_loss': test_loss_history,
                # 'image': images,
                'm_time': m_time},
                path)

            loss_history = []
            test_loss_history = []

if __name__ == '__main__':
    main()
