#!/usr/bin/env python
import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler

from sklearn.model_selection import train_test_split

from src.nets import ResNet1,CNN1
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

    train_weights = []
    test_weights = []

    #dataXRF = DataXRF().load_h5(config['data'])
    print(config['data'])
    for file in config['data']:
        print(file)

        dataXRF = DataXRF().load_h5(file)

        data = dataXRF.data.astype(float)
        labels = dataXRF.labels.astype(float)
        weights = dataXRF.weights.astype(float)

        data = torch.from_numpy(data).reshape(-1,1,data.shape[-1]).float()
        labels = torch.from_numpy(labels).reshape(-1,labels.shape[-1]).float()
        weights = torch.from_numpy(weights).flatten()

        print(data.shape,labels.shape,weights.shape)

        _train_data,_test_data, _train_labels,_test_labels, _train_weights,_test_weights = train_test_split(data,labels,weights,test_size = 4096,random_state = 93719)
        _train_weights /= len(_train_weights)

        train_data += [_train_data]
        test_data += [_test_data]

        train_labels += [_train_labels]
        test_labels += [_test_labels]

        train_weights += [_train_weights]
        test_weights += [_test_weights]

    eval_datasets = []
    for file in config['eval_data']:
        dataXRF = DataXRF().load_h5(file)

        data = dataXRF.data
        labels = dataXRF.labels.astype(float)

        print(labels.shape)

        data = torch.from_numpy(data).reshape(-1,1,data.shape[-1]).float()
        labels = torch.from_numpy(labels).reshape(-1,labels.shape[-1]).float()

        eval_dataset = SpectraDataset(data,labels)
        eval_dataset.shape = dataXRF.labels.shape
        eval_datasets += [eval_dataset]

    train_data = torch.cat(train_data)
    test_data = torch.cat(test_data)

    train_labels = torch.cat(train_labels)
    test_labels = torch.cat(test_labels)

    train_weights = torch.cat(train_weights)
    test_weights = torch.cat(test_weights)


    print("Train data:",train_data.shape,train_data.mean())
    print("Test  data:",test_data.shape,test_data.mean())

    print('Train weights:',train_weights)
    print(train_weights.mean())

    #train_data = SpectraDataset().load_h5(config['data'])

    train_dataset = SpectraDataset(train_data,train_labels)
    test_dataset = SpectraDataset(test_data,test_labels)

    #print(weights.shape)

    if not config['weights']:
        print('No weights')
        train_weights = ones(len(train_data.data))

    sampler = WeightedRandomSampler(train_weights,16384)

    train = DataLoader(train_dataset,
            batch_size = config['batch_size'],
            sampler = sampler,
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

    model = globals()[config['model']](channels = config['channels'])

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

    rescale = torch.Tensor([1.0,2.0,1.5,2.0,1.5]).to(device)

    loss_history = []
    test_loss_history = []

    time0 = time()
    
    for epoch in range(current_epoch,config['n_epochs']):
        print('Epoch: %d/%d'%(epoch,config['n_epochs']))

        model.train()
        for i,batch in enumerate(train):

            data,labels = batch


            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)

            loss = criterion(outputs,labels)
            loss = loss.mean(0)

            loss_history += [loss.tolist()]

            loss *= rescale
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        for i,batch in enumerate(test):

            data,labels = batch

            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)

            loss = criterion(outputs,labels)
            loss = loss.mean(0)

            test_loss_history += [loss.tolist()]

        images = []
        for evaluate in evals:
            image = []

            for i,batch in enumerate(evaluate):

                data,labels = batch

                data = data.to(device)
                labels = labels.to(device)

                outputs = model(data)
                image += [outputs.cpu().detach().numpy()]

            image = vstack(image)
            print(image.shape)
            image = image.reshape(evaluate.shape)
            images += [image]

        if epoch % 1 == 0:

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
                'image': images,
                'm_time': m_time},
                path)

            loss_history = []
            test_loss_history = []

if __name__ == '__main__':
    main()
