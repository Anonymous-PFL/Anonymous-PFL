# -*- coding: utf-8 -*-
import os

# import modules
# import base class
from pfedplat.Algorithm import Algorithm
from pfedplat.Client import Client
from pfedplat.DataLoader import DataLoader
from pfedplat.Model import Model
from pfedplat.Metric import Metric
from pfedplat.seed import setup_seed
# import metrics
from pfedplat.metric.Correct import Correct
from pfedplat.metric.Precision import Precision
from pfedplat.metric.Recall import Recall
# import models
from pfedplat.model.CNN_of_cifar10_tutorial import CNN_of_cifar10_tutorial
from pfedplat.model.CNN_CIFAR10_FedFV import CNN_CIFAR10_FedFV
from pfedplat.model.MLP import MLP
from pfedplat.model.NFResNet import NFResNet18

# import algorithm
import pfedplat.algorithm
from pfedplat.algorithm.Local.Local import Local
from pfedplat.algorithm.FedAvg.FedAvg import FedAvg
from pfedplat.algorithm.FedMGDA_plus.FedMGDA_plus import FedMGDA_plus
from pfedplat.algorithm.Ditto.Ditto import Ditto
from pfedplat.algorithm.FedProx.FedProx import FedProx
from pfedplat.algorithm.FedAMP.FedAMP import FedAMP
from pfedplat.algorithm.pFedMe.pFedMe import pFedMe
from pfedplat.algorithm.APFL.APFL import APFL
from pfedplat.algorithm.FedFomo.FedFomo import FedFomo
from pfedplat.algorithm.FedRep.FedRep import FedRep
from pfedplat.algorithm.FedROD.FedROD import FedROD
from pfedplat.algorithm.S_GPFL.S_GPFL import S_GPFL
from pfedplat.algorithm.GPFL.GPFL import GPFL

# import dataloader
from pfedplat.dataloaders.separate_data import separate_data, create_data_pool
from pfedplat.dataloaders.DataLoader_cifar10_pat import DataLoader_cifar10_pat
from pfedplat.dataloaders.DataLoader_cifar10_dir import DataLoader_cifar10_dir
from pfedplat.dataloaders.DataLoader_fashion_pat import DataLoader_fashion_pat
from pfedplat.dataloaders.DataLoader_fashion_dir import DataLoader_fashion_dir
from pfedplat.dataloaders.DataLoader_cifar100_pat import DataLoader_cifar100_pat
from pfedplat.dataloaders.DataLoader_cifar100_dir import DataLoader_cifar100_dir


# get the path of the data folder
data_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/data/'
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)

# get the path of the pool folder
pool_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/pool/'
if not os.path.exists(pool_folder_path):
    os.makedirs(pool_folder_path)

# import main
from pfedplat.main import initialize, read_params, outFunc
