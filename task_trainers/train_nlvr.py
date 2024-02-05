'''
 adapted from code with the following copyright:
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.blip_nlvr import blip_nlvr

import utils
from utils import cosine_lr_schedule, warmup_lr_schedule, count_parameters
from data import create_dataset, create_sampler, create_loader

import loralib as lora

def train(model, data_loader, optimizer, epoch, device, config, agent):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 10
 
    for i, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if not isinstance(batch_data[1], list):
            image0, image1, text, targets = batch_data
        else:
            image0, pos, neg, idx = batch_data
            text = pos + neg
            image0 = image0.repeat(2, 1, 1, 1)
            targets = torch.zeros((len(text,)), dtype=torch.int64)
            targets[:len(pos)] = 1
            image1 = None

        if image1 is not None:
            images = torch.cat([image0, image1], dim=0)
        else:
            images = image0
        images, targets = images.to(device), targets.to(device)   

        loss = model(images, text, targets=targets, train=True, agent=agent)   
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
               
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())  
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    


@torch.no_grad()
def evaluate(model, data_loader, device, config, agent):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        if not isinstance(batch_data[1], list):
            image0, image1, text, targets = batch_data
        else:
            image0, pos, neg, idx = batch_data
            text = pos + neg
            image0 = image0.repeat(2, 1, 1, 1)
            targets = torch.zeros((len(text, )), dtype=torch.int64)
            targets[:len(pos)] = 1
            image1 = None
        if image1 is not None:
            images = torch.cat([image0, image1], dim=0)
        else:
            images = image0
        images, targets = images.to(device), targets.to(device)   
        
        prediction = model(images, text, targets=targets, train=False, agent=agent)  
 
        _, pred_class = prediction.max(1)
        accuracy = (targets==pred_class).sum() / targets.size(0)
        
        metric_logger.meters['acc'].update(accuracy.item(), n=image0.size(0))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger.global_avg())   
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def multi_task_evaluate(model, data_loader, device, config, agent):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        if not isinstance(batch_data[1], list):
            image0, image1, text, targets = batch_data
        else:
            image0, pos, neg, idx = batch_data
            text = pos + neg
            image0 = image0.repeat(2, 1, 1, 1)
            targets = torch.zeros((len(text,)), dtype=torch.int64)
            targets[:len(pos)] = 1
            image1 = None
        if image1 is not None:
            images = torch.cat([image0, image1], dim=0)
        else:
            images = image0
        images, targets = images.to(device), targets.to(device)

        predictions = []
        for iT in range(agent.get_num_tasks()):
            agent.prep_model4task(iT)
            prediction = model(images, text, targets=targets, train=False, agent=agent)
            if isinstance(prediction, tuple):
                fuse_weights = prediction[1].detach().cpu()
                prediction = prediction[0]
            predictions.append(prediction.detach().cpu())
        agent.prep_model4task(-1)

        if agent.fuse_type in ['last']:
            prediction = torch.stack([x.softmax(dim=-1) for x in predictions], dim=-1)
        else:
            prediction = torch.stack(predictions, dim=-1)

        if agent.fuse_type in ['last']:
            prediction = prediction[:, :, -1]
        else:
            raise NotImplementedError(f'Unsupported fuse type: {agent.fuse_type}')

        _, pred_class = prediction.max(1)
        accuracy = (targets.cpu() == pred_class).sum() / targets.size(0)

        metric_logger.meters['acc'].update(accuracy.item(), n=image0.size(0))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

        
def main(args, config, eval=False):

    agent = args['agent']

    # change all "args." to args[""]
    args['result_dir'] = os.path.join(args['out_dir'], 'result')
    if utils.is_main_process(): Path(args['result_dir']).mkdir(parents=True, exist_ok=True)
    device = args['device']

    #### Dataset #### 
    print("Creating dataset")
    dataset_pass_dict = {'training_data_sample':args['training_data_sample']}
    datasets = create_dataset(config['dataset'], config, dataset_pass_dict)
    
    if args['distributed']:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True,False,False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]
    
    batch_size=[config['batch_size_train'],config['batch_size_test'],config['batch_size_test']]
    train_loader, val_loader, test_loader = create_loader(datasets,samplers,batch_size=batch_size,
                                                          num_workers=[args['num_workers'], args['num_workers'], args['num_workers']],is_trains=[True,False,False],
                                                          collate_fns=[None,None,None])

    # agent
    agent = args['agent']

    #### Model #### 
    print("Creating model")
    model, head_not_loaded = blip_nlvr(pretrained=args['pretrained'], image_size=config['image_size'],
                         vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], agent=agent, single_image_model=('vl-checklist' in config['dataset']))

    model = model.to(device)   
    
    model_without_ddp = model
    if args['distributed']:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['gpu']], find_unused_parameters=True)
        model_without_ddp = model.module    
            
    
    if agent.freeze_encoders:
        
        param_to_optim = []
        
        # task heads
        if (not agent.freeze_heads) or head_not_loaded:
            print('Training head')
            param_to_optim += list(model_without_ddp.cls_head.parameters())
        else:
            print('Locking head')
            for p in model_without_ddp.cls_head.parameters():
                p.requires_grad = False

        if agent.lora:
            param_to_optim += list(model_without_ddp.text_encoder.parameters())
            param_to_optim += list(model_without_ddp.visual_encoder.parameters())
            lora.mark_only_lora_as_trainable(model_without_ddp.text_encoder)
            lora.mark_only_lora_as_trainable(model_without_ddp.visual_encoder)
        
        # optimizer
        optimizer = torch.optim.AdamW(params=param_to_optim, lr=config['init_lr'], weight_decay=config['weight_decay'])
        nparam = count_parameters(param_to_optim)
    else:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
        # optimizer = torch.optim.AdamW(params=model_without_ddp.text_encoder.embeddings.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
        nparam = count_parameters(model.parameters())

    # print num trainable params    
    print(f'trainable_parameters = {nparam}')

    # init agent
    if not eval: agent.update_model(model_without_ddp)

    print("Start training")
    start_time = time.time()
    best = 0
    best_epoch = 0

    # flag for no training
    if not eval and args['eval_every'] < 0:
        if utils.is_main_process():  
            torch.save({'model':model_without_ddp.state_dict()}, args['model_save_path'])
        return

    start_epoch = 0
    for epoch in range(start_epoch, config['max_epoch']):
        load_file = os.path.join(args['out_dir'], 'checkpoint_%02d.pth'%epoch)
        if os.path.exists(load_file):
            checkpoint = torch.load(load_file)
            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            best = checkpoint['best']
            best_epoch = checkpoint['best_epoch']

    for epoch in range(start_epoch, config['max_epoch']):
        if not eval:
            if args['distributed']:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            
            train_stats = train(model, train_loader, optimizer, epoch,  device, config, agent)
            
        if eval or (epoch + 1) % args['eval_every'] == 0:
            eval_func = evaluate
            if agent.multi:
                eval_func = multi_task_evaluate

            val_stats = eval_func(model, val_loader, device, config, agent)
            test_stats = eval_func(model, test_loader, device, config, agent)
            
            if utils.is_main_process():  
                if eval:                
                    log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                                **{f'test_{k}': v for k, v in test_stats.items()},
                                }
                    with open(os.path.join(args['out_dir'], "log.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")   
                    
                else:       
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'val_{k}': v for k, v in val_stats.items()},
                                **{f'test_{k}': v for k, v in test_stats.items()},
                                'epoch': epoch,
                                }

                    if float(val_stats['acc'])>best:
                        torch.save({'model':model_without_ddp.state_dict()}, args['model_save_path'])

                        best = float(val_stats['acc'])
                        best_epoch = epoch

                    with open(os.path.join(args['out_dir'], "log.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                    # save checkpoint
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'epoch': epoch,
                        'best': best,
                        'best_epoch': best_epoch,
                    }
                    torch.save(save_obj, os.path.join(args['out_dir'], 'checkpoint_%02d.pth' % epoch))
                    epoch_old = epoch - 1
                    old_file = os.path.join(args['out_dir'], 'checkpoint_%02d.pth' % epoch_old)
                    if os.path.isfile(old_file):
                        os.remove(old_file)

                    print(f'Finished epoch {epoch} best epoch is {best_epoch} with acc {best}')
                
        dist.barrier()
        torch.cuda.empty_cache()
        if eval: 
            if utils.is_main_process():
                return test_stats['acc']  
            else:
                return -0.1

    if config['max_epoch'] < args['eval_every']:
        if utils.is_main_process():  
            torch.save({'model':model_without_ddp.state_dict()}, args['model_save_path'])