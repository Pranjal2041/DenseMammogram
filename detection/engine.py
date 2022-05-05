import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import detection.utils as utils
from detection.coco_eval import CocoEvaluator
from detection.coco_utils import get_coco_api_from_dataset
from tqdm import tqdm
import numpy as np


sys.path.append("..")
from utils import AverageMeter
from advanced_logger import LogPriority

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    #for batch_idx,(images, targets) in enumerate(tqdm(data_loader)):
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        #print(images.shape)
        images = list(image.to(device) if len(image)>2 else [image[0].to(device),image[1].to(device)] for image in images)
        #print(len(images))
        #print(images[0].shape)
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
        
        #if(batch_idx%20==0):
        #    print('epoch {} batch {} : {}'.format(epoch,batch_idx,losses_reduced))

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def train_one_epoch_simplified(model, optimizer, data_loader, device, epoch, experimenter,optimizer_backbone=None):
    
    model.train()
    lr_scheduler = None
    lr_scheduler_backbone = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
        if(optimizer_backbone is not None):
           lr_scheduler_backbone = torch.optim.lr_scheduler.LinearLR(optimizer_backbone, start_factor=warmup_factor, total_iters=warmup_iters)
           

    loss_meter = AverageMeter()

    for step, (images, targets) in enumerate(tqdm(data_loader)):

        optimizer.zero_grad()
        if(optimizer_backbone is not None):
            optimizer_backbone.zero_grad()

        images = list(image.to(device) if len(image)>2 else [image[0].to(device),image[1].to(device)] for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())


        if not math.isfinite(losses.item()):
            print(f"Loss is {losses.item()}, stopping training")
            print(loss_dict)
            experimenter.log(f"Loss is {losses.item()}, stopping training")
            sys.exit(1)

        losses.backward()
        loss_meter.update(losses.item())
        optimizer.step()
        if optimizer_backbone is not None:
            optimizer_backbone.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if lr_scheduler_backbone is not None:
            lr_scheduler_backbone.step()
            
        if (step+1)%10 == 0:
            experimenter.log('Loss after {} steps: {}'.format(step+1, loss_meter.avg))
        if epoch == 0 and (step+1)%50 == 0:
            experimenter.log('LR after {} steps: {}'.format(step+1, optimizer.param_groups[0]['lr']))

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
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

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


def coco_summ(coco_eval, experimenter):
    self = coco_eval
    def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        experimenter.log(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s), priority = LogPriority.MEDIUM)
        return mean_s
    def _summarizeDets():
        stats = np.zeros((12,))
        stats[0] = _summarize(1)
        stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
        stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
        stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
        stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
        stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
        stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
        stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
        stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
        stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
        return stats
    _summarizeDets()

@torch.inference_mode()
def evaluate_simplified(model, data_loader, device, experimenter):
    cpu_device = torch.device("cpu")
    model.eval()
    experimenter.log('Evaluating Validation Parameters')

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)
        
    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()

    # Debug and see what all info it has
    # coco_evaluator.summarize()
    for iou_type, coco_eval in coco_evaluator.coco_eval.items():
        print(f"IoU metric: {iou_type}")
        coco_summ(coco_eval, experimenter)

    return coco_evaluator

def evaluate_loss(model, device, val_loader, experimenter=None):
    model.train()
    #experimenter.log('Evaluating Validation Loss')
    with torch.no_grad():
         loss_meter = AverageMeter()
    for images, targets in tqdm(val_loader):
        images = list(image.to(device) if len(image)>2 else [image[0].to(device),image[1].to(device)] for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_meter.update(losses.item())
    return loss_meter.avg