import argparse
import collections
import os

import torch
import torchvision
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
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    config = ConfigParser.from_args(args, options)    
        
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_devices"]
    device_ids = [d for d in range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))) ]
    
    dataset_img_test_lip = load_dataset("imagefolder", data_dir="./datasets/LIP/lip_img_sample", drop_labels=True, split='validation')
    dataset_seg_test_lip = load_dataset("imagefolder", data_dir="./datasets/LIP/lip_seg_sample", drop_labels=True, split='validation')
    dataset_img_train_lip = load_dataset("imagefolder", data_dir="./datasets/LIP/lip_img_sample", drop_labels=True, split='train')
    dataset_seg_train_lip = load_dataset("imagefolder", data_dir="./datasets/LIP/lip_seg_sample", drop_labels=True, split='train')

    
    dataset_name = config["use_dataset_config"]
    dataset_info_dict = config["datasets"][dataset_name]
    
    if "class_list_new" in dataset_info_dict.keys():
        cls_ls = dataset_info_dict["class_list_new"]
        cls_map = dict(dataset_info_dict["class_map"])
        cls_map = {int(key): value for key, value in cls_map.items()}
        cls_ls_old = dataset_info_dict["class_list_new"]
    else:
        cls_ls = dataset_info_dict["class_list"]
        cls_ls_old = []
        cls_map = None
        
    # create model and move it to GPU with id rank
    # cls_ls = config["dataset_info"]["class_list"]
    
    
    bbox_ds_lip = BboxDataset(  dataset_img_train_lip, dataset_seg_train_lip, 
                                train_size=len(dataset_img_train_lip), 
                                ds_name=dataset_info_dict['ds_name'], cls_ls=cls_ls, cls_ls_old=cls_ls_old, cls_map=cls_map, is_train=True)
    bbox_ds_te_lip = BboxDataset(   dataset_img_test_lip, dataset_seg_test_lip,
                                    train_size=len(dataset_img_train_lip), 
                                    ds_name=dataset_info_dict['ds_name'], cls_ls=cls_ls, cls_ls_old=cls_ls_old, cls_map=cls_map, is_train=False)
    
    
    bsz = config['data_loader']["args"]['batch_size']
    num_workers = config['data_loader']["args"]['num_workers']
    train_set = torch.utils.data.DataLoader(bbox_ds_lip, batch_size=bsz, 
                                            shuffle=True,
                                            num_workers=num_workers, collate_fn=collate_fn)
    
    test_set = torch.utils.data.DataLoader(bbox_ds_te_lip, batch_size=bsz, 
                                           shuffle=False,
                                           num_workers=num_workers, collate_fn=collate_fn)
    
    
    device_id = config["main_device_id"]
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT")
    num_classes = len(cls_ls)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device_id);
    
    # get function handles of loss and metrics
    criterion = []
    # criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) if hasattr(module_metric, met) else met for met in config['metrics']]    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    non_train_params = filter(lambda p: not p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    
    warmup_factor = 1.0 / 1000
    warmup_iters = min(1000, len(train_set) - 1)

    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_factor, total_iters=warmup_iters
    )

    trainer = Trainer(model, criterion, metrics, optimizer,
                    config=config,
                    device=device_id,
                    data_loader=train_set,
                    valid_data_loader=test_set,
                    lr_scheduler=lr_scheduler)
    
    trainer.train()
