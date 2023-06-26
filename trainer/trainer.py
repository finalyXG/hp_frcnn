import math
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from torchvision.utils import draw_bounding_boxes, make_grid

from base import BaseTrainer
from utils import MetricTracker, get_refine_box_result, inf_loop
from utils.pytorch_vision import engine, utils
from utils.pytorch_vision.coco_eval import CocoEvaluator
from utils.pytorch_vision.coco_utils import get_coco_api_from_dataset


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.metric_names = [m.__name__ if type(m) != str else m for m in self.metric_ftns]
        self.val_metric_names = [f"val_{m.__name__}" if type(m) != str else f"val_{m}" for m in self.metric_ftns]        
        self.train_metrics = MetricTracker('loss', *self.metric_names, writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *self.val_metric_names, writer=self.writer)
        self.coco_eval_names = [] 
        
        
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        model = self.model 
        optimizer = self.optimizer 
        lr_scheduler = self.lr_scheduler
        data_loader = self.data_loader
        device = self.device
        print_freq = self.config["trainer"]["print_freq"]
        
        model.train()
        self.train_metrics.reset()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{epoch}]"
        scaler = None
        batch_idx = 0
        # for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        self.display_class_ls = np.array(self.config["dataset_options"][self.config["use_dataset_config"]]["diaplay_class_ls"])
        
        
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            self.train_metrics.update('loss', loss_value)
            # for met in self.metric_ftns:
                # self.train_metrics.update(met.__name__, loss_dict_reduced[met.__name__])
            for met in self.metric_names:
                if met in loss_dict_reduced.keys():
                    self.train_metrics.update(met, loss_dict_reduced[met])
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss_value))
                                
            if batch_idx == self.len_epoch:
                break
            
            batch_idx += 1
            
        log = self.train_metrics.result()
        
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            # log.update(**{'val_'+k : v for k, v in val_log.items()})
            log.update(**{k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        return log
    
    def get_coco_eval_metric_names(self, coco_evaluator):
        if len(self.coco_eval_names) == 0:
            self.coco_eval_names.append(f"Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets={coco_evaluator.coco_eval['bbox'].params.maxDets[2]} ] ")
            self.coco_eval_names.append(f"Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets={coco_evaluator.coco_eval['bbox'].params.maxDets[2]} ] ")
            self.coco_eval_names.append(f"Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets={coco_evaluator.coco_eval['bbox'].params.maxDets[2]} ] ")
            self.coco_eval_names.append(f"Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets={coco_evaluator.coco_eval['bbox'].params.maxDets[2]} ] ")
            self.coco_eval_names.append(f"Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets={coco_evaluator.coco_eval['bbox'].params.maxDets[2]} ] ")
            self.coco_eval_names.append(f"Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets={coco_evaluator.coco_eval['bbox'].params.maxDets[2]} ] ")
            self.coco_eval_names.append(f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets={coco_evaluator.coco_eval['bbox'].params.maxDets[0]} ] ")
            self.coco_eval_names.append(f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets={coco_evaluator.coco_eval['bbox'].params.maxDets[1]} ] ")
            self.coco_eval_names.append(f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets={coco_evaluator.coco_eval['bbox'].params.maxDets[2]} ] ")
            self.coco_eval_names.append(f"Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets={coco_evaluator.coco_eval['bbox'].params.maxDets[2]} ] ")
            self.coco_eval_names.append(f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets={coco_evaluator.coco_eval['bbox'].params.maxDets[2]} ] ")
            self.coco_eval_names.append(f"Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets={coco_evaluator.coco_eval['bbox'].params.maxDets[2]} ] ")
        return dict(zip(self.coco_eval_names, coco_evaluator.coco_eval['bbox'].stats))
        
    
    def _get_iou_types(self, model):
        model_without_ddp = model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_without_ddp = model.module
        iou_types = ["bbox"]
        if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
            iou_types.append("segm")
        if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
            iou_types.append("keypoints")
        return iou_types
    
    def writer_add_detecion_images(self, data_loader, tag="image"):
        for i in [0,100,200,300,400,500,600,700,800,900]:
            with torch.no_grad():
                if i < len(data_loader.dataset):
                    img = data_loader.dataset[i][0]
                    # img_rs = data_loader.dataset.show_sample_img_w_bbox(i, plt_show=False, return_fig=True)
                    # img_rs = torch.Tensor(np.array(img_rs, dtype=np.float64) / 255.).permute([2,0,1])
                    predictions = self.model([img.to(self.device)])
                    iou_threshold = self.config["detector"]["iou_threshold"]
                    score_threshold = self.config["detector"]["score_threshold"]
                    label_name_dict_ls = self.display_class_ls
                    tmp_boxes, tmp_labels = get_refine_box_result(predictions, 
                                                                iou_threshold, 
                                                                score_threshold, 
                                                                label_name_dict_ls)
                    

                    drawn_boxes = draw_bounding_boxes((img * 255).to(torch.uint8), tmp_boxes, tmp_labels, colors="red")
                    self.writer.add_image(f'{tag}_{i}', drawn_boxes)
    
    
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        model = self.model
        data_loader = self.valid_data_loader
        device = self.device
        n_threads = torch.get_num_threads() # Laurence 20230610
        # print("n_threads",n_threads)
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        model.eval()
        self.valid_metrics.reset()
        
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"
        coco = get_coco_api_from_dataset(data_loader.dataset)
        iou_types = self._get_iou_types(model)
        coco_evaluator = CocoEvaluator(coco, iou_types)
        with torch.no_grad():
            for images, targets in metric_logger.log_every(data_loader, 100, header):
            # for images, targets in data_loader:
                images = list(img.to(device) for img in images)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                model_time = time.time()
                outputs = model(images)

                outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                model_time = time.time() - model_time

                res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
                evaluator_time = time.time()
                coco_evaluator.update(res)
                evaluator_time = time.time() - evaluator_time
                metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        
                

        self.writer_add_detecion_images(self.data_loader, tag="train_img")
        self.writer_add_detecion_images(self.valid_data_loader, tag="test_img")
        
        # gather the stats from all processes
        metric_logger.synchronize_between_processes() # Laurence 20230610
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes() # Laurence 20230610

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads) # Laurence 20230610
        
        val_log = self.get_coco_eval_metric_names(coco_evaluator)
        for k, v in val_log.items():
            self.valid_metrics.update('val_'+k, v)
        # return coco_evaluator
        log = self.valid_metrics.result()

        return log    
    
    
    def _valid_epoch_old(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()
    
    
    

    def _train_epoch_old(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
