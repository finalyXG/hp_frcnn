import argparse
import collections
import os
import types

import torch
import torchvision
import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import model.metric as module_metric
from data_loader.data_loaders import BboxDataset
from datasets import load_dataset
from trainer import Trainer
from utils.parse_config import ConfigParser
from utils.pytorch_vision.utils import collate_fn

if __name__ == '__main__':
    
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')

    config = ConfigParser.from_args(args)    
    dataset_img_test_lip = load_dataset("imagefolder", data_dir=config["dataset_files"]["image"]["path"], drop_labels=True, split='validation')
    dataset_seg_test_lip = load_dataset("imagefolder", data_dir=config["dataset_files"]["segmentation"]["path"], drop_labels=True, split='validation')
    dataset_img_train_lip = load_dataset("imagefolder", data_dir=config["dataset_files"]["image"]["path"], drop_labels=True, split='train')
    dataset_seg_train_lip = load_dataset("imagefolder", data_dir=config["dataset_files"]["segmentation"]["path"], drop_labels=True, split='train')
    
    dataset_info_dict = config["dataset"]
    cls_ls = dataset_info_dict["class_list"]
    cls_ls_old = []
    cls_map = None
        
    # create model and move it to GPU with id rank
    # cls_ls = config["dataset_info"]["class_list"]
    
    
    bbox_ds_lip = BboxDataset(  dataset_img_train_lip, dataset_seg_train_lip, 
                                train_size=len(dataset_img_train_lip), 
                                ds_name=dataset_info_dict['ds_name'], cls_ls=cls_ls, cls_ls_old=cls_ls_old, cls_map=cls_map, is_train=True, mode="gen")
    bbox_ds_te_lip = BboxDataset(   dataset_img_test_lip, dataset_seg_test_lip,
                                    train_size=len(dataset_img_train_lip), 
                                    ds_name=dataset_info_dict['ds_name'], cls_ls=cls_ls, cls_ls_old=cls_ls_old, cls_map=cls_map, is_train=False, mode="gen")
    
    
    bsz = config['data_loader']["args"]['batch_size']
    num_workers = config['data_loader']["args"]['num_workers']
    train_set = torch.utils.data.DataLoader(bbox_ds_lip, batch_size=bsz, 
                                            shuffle=True,
                                            num_workers=num_workers)
    
    test_set = torch.utils.data.DataLoader(bbox_ds_te_lip, batch_size=bsz, 
                                           shuffle=False,
                                           num_workers=num_workers)
    
    
    for bd in tqdm.tqdm(train_set):
        continue
    
    for bd in tqdm.tqdm(test_set):
        continue
