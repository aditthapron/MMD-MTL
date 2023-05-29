import numpy as np
import math
import torch
import torch.nn as nn
from torch.optim import SGD,Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
import torch.nn.functional as F
import tensorflow as tf

import json
import sys
from tqdm import tqdm
from argparse import ArgumentParser
from collections import deque
from datetime import datetime
from itertools import count
import itertools
from multiprocessing import cpu_count
from multiprocessing import Pool
from pathlib import Path

from data_loader import get_dataloader
from model import get_model
from utility import *

import warnings
warnings.filterwarnings("ignore")

#[dataloader's name, original Hz, length(s), formatted Hz,fold,k], fold and k will be assigned in train()

task = {
        'study1a_50hz':['features_study1a',50,3,50,None,None],
        'study1a_25hz':['features_study1a_25hz',50,3,25,None,None],
        'study1a_10hz':['features_study1a_10hz',50,3,10,None,None],
        'study1a_5hz':['features_study1a_5hz',50,3,5,None,None],
        'CRA': ['features_Dataset_balanced_no_missing_normalized',5,3,5,None,None]
        }
source_task = ['study1a_50hz','study1a_25hz','study1a_10hz','study1a_5hz']
target_task = ['CRA']

def nllloss_var(output, var, target,w):
    return torch.mean(torch.mean(torch.log(var+1e-8)+(output - target)**2/(var+1e-8),dim=2).mul(w))
     
def MSE(output, target,w):
    return  torch.mean(torch.mean((output - target)**2,dim=2).mul(w))

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        source = source.reshape(batch_size,-1)
        target = target.reshape(batch_size,-1)
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

def load_data(task):
    print(task)
   #Format of task: [name, original Hz, length(s), formatted Hz,fold,k]
    return get_dataloader(task[0],task[4],task[5],task[2],task[1],4,'iphone') 

def train(data,model,platform,fold,k,length,hz,valid_every,decay_every,batch_per_valid,n_workers,comment,):


    # setup job name
    start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    job_name = f"{fold}_{k}_{platform}_{start_time}"

    # setup checkpoint and log dirs
    checkpoints_path = Path('experiment') / model/data/ "checkpoints" / job_name
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    writer = Logger(str(Path('experiment') /model/data/ "logs" / job_name))
    print('MMD')
    print(str(Path('experiment') /model/data/ "logs" / job_name))
    #dataloader
    data_loader_train,data_loader_val,train_iter,valid_iter ={},{},{},{}
    #assign fold and k
    for t in task.keys():
        task[t][4] = fold
        task[t][5] = k

    #create Pool of dataloaders using multi processing
    with Pool(len(task)) as p:
        data_loader_temp = p.map(load_data,list(task.values()))
    for i,t in enumerate(task.keys()):
        data_loader_train[t],data_loader_val[t] = data_loader_temp[i]
        train_iter[t] =  iter(data_loader_train[t])
        valid_iter[t] =  iter(data_loader_val[t])
        print(task[t][0]+' loaded')

    # build network and training tools
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = get_model('CRUFT_encoder',length,hz).to(device)
    # encoder = torch.jit.script(encoder)

    classifier = {}
    for t in task.keys():
        classifier[t] = get_model('CRUFT_classifier',length,hz).to(device)
        # classifier[t] = torch.jit.script(classifier[t])

    optimizer = Adam(list(itertools.chain(*[list(c.parameters()) for c in classifier.values()])) + list(encoder.parameters()), lr=0.002)
    scheduler = StepLR(optimizer, step_size=decay_every, gamma=0.5)
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    MMD = MMD_loss()
    # record training infos
    pbar = tqdm(total=valid_every, ncols=0, desc="Train")
    running_train_loss, running_grad_norm, running_train_acc_act,running_train_acc_phone,running_train_acc_context, running_train_mmd = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100), deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    running_valid_loss, running_valid_acc_act, running_valid_acc_phone, running_valid_acc_context = deque(maxlen=batch_per_valid),deque(maxlen=batch_per_valid),deque(maxlen=batch_per_valid),deque(maxlen=batch_per_valid)
    best_val_loss = np.inf
    save_iter=0
    # start training
    for step in count(start=1):

        h1,h2,loss_task ={},{},{}
        loss = 0
        for t in task.keys():
            try:
                batch_1,batch_2,target_activity,target_phone,act_w,phone_w = next(train_iter[t])
            except:
                train_iter[t] =  iter(data_loader_train[t])
                batch_1,batch_2,target_activity,target_phone,act_w,phone_w = next(train_iter[t])
            
            batch_1,batch_2 = torch.nan_to_num(batch_1),torch.nan_to_num(batch_2)
            batch_1,batch_2 = torch.clamp(batch_1,min=-1E3, max=1E3),torch.clamp(batch_2,min=-1E3, max=1E3)
            batch_1,batch_2,target_activity,target_phone = batch_1.to(device),batch_2.to(device),target_activity.to(device),target_phone.to(device).to(device)
            act_w,phone_w = act_w.to(device),phone_w.to(device)
            
            h1[t],h2[t] = encoder(batch_1,batch_2)
            pred_activity,pred_phone,unc_activity,unc_phone = classifier[t](h1[t],h2[t])

            alpha=0.3
            mse = 0.5*MSE(pred_activity,F.one_hot(target_activity, num_classes=4).float(),act_w) + 0.5*MSE(pred_phone,F.one_hot(target_phone, num_classes=3).float(),phone_w)
            nll = 0.5*nllloss_var(pred_activity,unc_activity,F.one_hot(target_activity, num_classes=4).float(),act_w) + 0.5*nllloss_var(pred_phone,unc_phone,F.one_hot(target_phone, num_classes=3).float(),phone_w)
            
            loss = loss + (1-alpha) *nll
            loss = loss + alpha* (0.5* torch.mean(criterion(pred_activity.swapaxes(1,2),target_activity)*act_w) + 0.5* torch.mean(criterion(pred_phone.swapaxes(1,2),target_phone)*phone_w))

        loss = loss/len(task)
        #MMD compute  L_s
        MMDloss = 0
        for i,t1 in enumerate(source_task):
            for j,t2 in enumerate(source_task):
                if j>i:
                    MMDloss = MMDloss + MMD(h1[t1],h1[t2])
        #compute L_D
        MMDloss = MMDloss + MMD(h1[source_task[-1]],h1[target_task])
        MMDloss = MMDloss/ ((len(task)*(len(task)-1))/2)
        lambd = 2 / (1 + math.exp(-10 * (step) / (valid_every*20))) - 1
        total_loss = loss + lambd*MMDloss
        optimizer.zero_grad()
        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(itertools.chain(*[list(c.parameters()) for c in classifier.values()])),
            max_norm=3,
            norm_type=2.0,
        )

        optimizer.step()
        scheduler.step()

        running_train_loss.append(loss.item())
        running_train_mmd.append(lambd*MMDloss.item())
        running_grad_norm.append(grad_norm)


        running_train_acc_act.append(torch.mean(torch.Tensor.float(torch.mode(pred_activity.argmax(dim=2),dim=1)[0] == torch.mode(target_activity,dim=1)[0])).item())
        running_train_acc_phone.append(torch.mean(torch.Tensor.float(torch.mode(pred_phone.argmax(dim=2),dim=1)[0] == torch.mode(target_phone,dim=1)[0])).item())
        running_train_acc_context.append(torch.mean(torch.Tensor.float(torch.logical_and(torch.mode(pred_activity.argmax(dim=2),dim=1)[0] == torch.mode(target_activity,dim=1)[0],
            torch.mode(pred_phone.argmax(dim=2),dim=1)[0] == torch.mode(target_phone,dim=1)[0]))).item())

        avg_train_loss = sum(running_train_loss) / len(running_train_loss)
        avg_train_mmd = sum(running_train_mmd) / len(running_train_mmd)
        avg_grad_norm = sum(running_grad_norm) / len(running_grad_norm)
        avg_train_acc_act = sum(running_train_acc_act) / len(running_train_acc_act)
        avg_train_acc_phone = sum(running_train_acc_phone) / len(running_train_acc_phone)
        avg_train_acc_context = sum(running_train_acc_context) / len(running_train_acc_context)


        if step % (valid_every//10) == 0:
            pbar.update(valid_every//10)
            pbar.set_postfix(loss=avg_train_loss, mmd = avg_train_mmd, grad_norm=avg_grad_norm.item(), acc_act=avg_train_acc_act, acc_phone=avg_train_acc_phone, acc_context = avg_train_acc_context)

        if step % valid_every == 0:
            pbar.reset()
            for t in task.keys():
                classifier[t].eval()
            encoder.eval()
            
            h1,h2,loss_task ={},{},{}
            
            for t in task.keys():
                for _ in range(batch_per_valid):
                    try:
                        batch_1,batch_2,target_activity,target_phone,act_w,phone_w = next(valid_iter[t])
                    except:
                        valid_iter[t] =  iter(data_loader_val[t])
                        batch_1,batch_2,target_activity,target_phone,act_w,phone_w = next(valid_iter[t])

                    batch_1,batch_2 = torch.nan_to_num(batch_1),torch.nan_to_num(batch_2)
                    batch_1,batch_2 = torch.clamp(batch_1,min=-1E3, max=1E3),torch.clamp(batch_2,min=-1E3, max=1E3)
                    batch_1,batch_2,target_activity,target_phone = batch_1.to(device),batch_2.to(device),target_activity.to(device),target_phone.to(device).to(device)
                    act_w,phone_w = act_w.to(device),phone_w.to(device)
                    with torch.no_grad():
                        h1[t],h2[t] = encoder(batch_1,batch_2)
                        pred_activity,pred_phone,unc_activity,unc_phone = classifier[t](h1[t],h2[t])
                    alpha=0.3
                    mse = 0.5*MSE(pred_activity,F.one_hot(target_activity, num_classes=4).float(),act_w) + 0.5*MSE(pred_phone,F.one_hot(target_phone, num_classes=3).float(),phone_w)
                    nll = 0.5*nllloss_var(pred_activity,unc_activity,F.one_hot(target_activity, num_classes=4).float(),act_w) + 0.5*nllloss_var(pred_phone,unc_phone,F.one_hot(target_phone, num_classes=3).float(),phone_w)
                    
                    # loss =  alpha*mse + (1-alpha) *nll
                    # loss = 0.5* torch.mean(criterion(pred_activity.swapaxes(1,2),target_activity)*act_w) + 0.5* torch.mean(criterion(pred_phone.swapaxes(1,2),target_phone)*phone_w)
                    loss = (1-alpha) *nll
                    loss = loss + alpha* (0.5* torch.mean(criterion(pred_activity.swapaxes(1,2),target_activity)*act_w) + 0.5* torch.mean(criterion(pred_phone.swapaxes(1,2),target_phone)*phone_w))

                    running_valid_loss.append(loss.item())
                    running_valid_acc_act.append(torch.mean(torch.Tensor.float(torch.mode(pred_activity.argmax(dim=2),dim=1)[0] == torch.mode(target_activity,dim=1)[0])).item())
                    running_valid_acc_phone.append(torch.mean(torch.Tensor.float(torch.mode(pred_phone.argmax(dim=2),dim=1)[0] == torch.mode(target_phone,dim=1)[0])).item())
                    running_valid_acc_context.append(torch.mean(torch.Tensor.float(torch.logical_and(torch.mode(pred_activity.argmax(dim=2),dim=1)[0] == torch.mode(target_activity,dim=1)[0],\
                        torch.mode(pred_phone.argmax(dim=2),dim=1)[0] == torch.mode(target_phone,dim=1)[0]))).item())

                avg_valid_loss = sum(running_valid_loss) / len(running_valid_loss)
                avg_valid_acc_act = sum(running_valid_acc_act) / len(running_valid_acc_act)
                avg_valid_acc_phone = sum(running_valid_acc_phone) / len(running_valid_acc_phone)
                avg_valid_acc_context = sum(running_valid_acc_context) / len(running_valid_acc_context)
                tqdm.write(f"Valid: step={step}, task:{t}, loss={avg_valid_loss:.3f}, acc_act={avg_valid_acc_act:.3f}, acc_phone={avg_valid_acc_phone:.3f}, acc_context={avg_valid_acc_context:.3f}")
                writer.scalar_summary("Loss/train", avg_train_loss, step)
                writer.scalar_summary("Loss/valid", avg_valid_loss, step)
                writer.scalar_summary("Acc/train", avg_train_acc_act, step)
                writer.scalar_summary("Acc/valid", avg_valid_acc_act, step)
                writer.scalar_summary("Acc/train", avg_train_acc_phone, step)
                writer.scalar_summary("Acc/valid", avg_valid_acc_phone, step)
                writer.scalar_summary("Acc/train", avg_train_acc_context, step)
                writer.scalar_summary("Acc/valid", avg_valid_acc_context, step)
            for t in task.keys():
                classifier[t].train()
            encoder.train()

            if avg_valid_loss < best_val_loss:
                save_iter = step
                best_val_loss = avg_valid_loss
                ckpt_path = checkpoints_path / f"{step}.pt"
                torch.save(encoder.state_dict(), str(ckpt_path))



if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("--model", type=str, default='CRUFT')
    PARSER.add_argument("--data", type=str, default='features_study1a')
    PARSER.add_argument("--platform", type=str, default='iphone')
    PARSER.add_argument("--fold", type=int, default=5)
    PARSER.add_argument("-k", type=int, default=0)
    PARSER.add_argument("--length", type=int, default=3)
    PARSER.add_argument("--hz", type=int, default=50)
    PARSER.add_argument("--valid_every", type=int, default=5000)
    PARSER.add_argument("--decay_every", type=int, default=20000)
    PARSER.add_argument("--batch_per_valid", type=int, default=100)
    PARSER.add_argument("--n_workers", type=int, default=4)
    PARSER.add_argument("--comment", type=str)
    train(**vars(PARSER.parse_args()))
