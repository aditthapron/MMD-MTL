import glob
import os
import pickle
import numpy as np
import torch
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import glob
import random
import sys
from scipy.stats import entropy
from scipy.signal import butter, lfilter, find_peaks
from GAN_model import Generator
from scipy.interpolate import CubicSpline

import pandas as pd
import zipfile,json,io
import ESfeature
from data_loader_helper import *
    

def str_to_class(name):
    return getattr(sys.modules[__name__],name)

class dummy(Dataset):
    # for dubugging
    def __init__(self, mode:str, n_fold:int, k:int, length, overlapping:float, hz:float, platform:str):
        super(dummy, self).__init__()
        # self.dim=length
    def __getitem__(self,idx):
        return torch.rand(9,150),torch.rand(78,3),torch.tensor([0], dtype=torch.long),torch.tensor([0], dtype=torch.long),torch.tensor([0], dtype=torch.long),torch.tensor([0], dtype=torch.long)
    def __len__(self):
        return 1000000

class CRA_dataset(Dataset):
    """ Data loader"""
    def __init__(self, mode:str, n_fold:int, k:int, length:float, overlapping:float, hz:float, platform:str):

        super(CRA_dataset, self).__init__()
        self.datadir = '/home/aditthapron/DARPA/CRA/data/raw_feature_'+platform
        self.mode = mode
        self.k = k
        self.n_fold = n_fold
        self.length = length
        self.overlapping = overlapping
        self.hz=hz
        self.platform = platform

        self.subj = np.array(glob.glob(self.datadir+'/*'))
        self.create_normalizer()
        self.data_split()
        self.load_normalizer()

        self.activity_label = {'sit':0,'lie':1,'walk':2,'stand':3}
        self.phone_label = {'hand':0,'pocket':1,'bag':2}

        self.n_subject = len(self.subj)
        self.file = {}
        for file_subj in self.subj:
            self.file[file_subj] = glob.glob(file_subj+'/*.npy')

        assert len(self.mean)>0


    def data_split(self):
        #perform subject-wise splitting
        np.random.seed(0)
        np.random.shuffle(self.subj)
        self.subj_testing = self.subj[self.k* (len(self.subj)//self.n_fold):(self.k+1)* (len(self.subj)//self.n_fold)]
        self.subj_training = np.array([s for s in self.subj if s not in self.subj_testing])
        if self.mode == 'train':
            self.subj = self.subj_training
        elif self.mode == 'val':
            self.subj = self.subj_testing
        else:
            raise NameError('Mode is incorrect')

    def create_normalizer(self):
        if not os.path.exists(self.datadir+'_normalize'):
            os.makedirs(self.datadir+'_normalize')
            for file_subj in self.subj:
                event_file = glob.glob(file_subj+'/*.npy')
                all_record = []
                for file in event_file:
                    all_record.append(np.load(file)[:,1:10])

                all_record = np.concatenate(all_record)
                mean_arr,std_arr =np.mean(all_record,axis=0),np.std(all_record,axis=0)
                np.save('{}_normalize/{}_mean.npy'.format(self.datadir,file_subj.split('/')[-1]),mean_arr)
                np.save('{}_normalize/{}_std.npy'.format(self.datadir,file_subj.split('/')[-1]),std_arr)

    def load_normalizer(self):
        self.mean = {}
        self.std = {}
        for file_subj in self.subj:
            subj_id = file_subj.split('/')[-1]
            self.mean[subj_id] = np.load('{}_normalize/{}_mean.npy'.format(self.datadir,subj_id))
            self.std[subj_id] = np.load('{}_normalize/{}_std.npy'.format(self.datadir,subj_id))

    def __getitem__(self, idx):
        
        subj = self.subj[idx%self.n_subject]
        event = self.file[subj]
        idx_event = np.random.randint(len(event))
        signal = np.load(event[idx_event])
        signal_length = len(signal)
        try:
            start = np.random.randint(signal_length-self.hz*self.length)
            end = start+self.hz*self.length
            signal_out = signal[start:end,1:10]
        except:
            signal_out = np.zeros((self.hz*self.length,9))
            signal_out[:signal_length] = signal[:,1:10]
        signal_out = self.normalize(signal_out,subj.split('/')[-1]).swapaxes(0,1)

        [_,act_label,phone_label] = event[idx_event].split('/')[-1][:-4].split('_')


        return torch.FloatTensor(signal_out), torch.tensor(self.activity_label[act_label], dtype=torch.long),torch.tensor(self.phone_label[phone_label], dtype=torch.long) 


    def normalize(self,feature,subj):
        mean_arr,std_arr = self.mean[subj],self.std[subj]
        mean_arr,std_arr = np.tile(mean_arr,(len(feature),1)),np.tile(std_arr,(len(feature),1))
        return (feature-mean_arr)/(std_arr+1E-8)

    def __len__(self):
        return self.n_subject*10*3000

class raw_Dataset_balanced(CRA_dataset):

    def __getitem__(self, idx):
        
        offset=0
        while True:
            subj = self.subj[(idx+offset)%self.n_subject]
            event = self.file[subj]
            #activity
            act_select = list(self.activity_label.keys())[idx%4]
            event = [i for i in event if i.split('_')[-2]==act_select]
            if len(event)>0:
                break
            else:
                offset+=1
        
        idx_event = np.random.randint(len(event))
        signal = np.load(event[idx_event])
        signal_length = len(signal)
        try:
            start = np.random.randint(signal_length-self.hz*self.length)
            end = start+self.hz*self.length
            signal_out = signal[start:end,1:10]
        except:
            signal_out = np.zeros((self.hz*self.length,9))
            signal_out[:signal_length] = signal[:,1:10]
        signal_out = self.normalize(signal_out,subj.split('/')[-1]).swapaxes(0,1)

        [_,act_label,phone_label] = event[idx_event].split('/')[-1][:-4].split('_')


        return torch.FloatTensor(signal_out), torch.tensor(self.activity_label[act_label], dtype=torch.long),torch.tensor(self.phone_label[phone_label], dtype=torch.long) 


class raw_Dataset_balanced_no_missing(raw_Dataset_balanced):
    """ Data loader"""
    def __init__(self, mode:str, n_fold:int, k:int, length:float, overlapping:float, hz:float, platform:str):

        self.datadir = '/home/aditthapron/DARPA/CRA/data/raw_feature_'+platform
        self.mode = mode
        self.k = k
        self.n_fold = n_fold
        self.length = length
        self.overlapping = overlapping
        self.hz=hz
        self.platform = platform

        self.subj = np.array(glob.glob(self.datadir+'/*'))

        self.temp_subj = []
        for i in range(len(self.subj)):
            file_list = glob.glob(self.subj[i]+'/*.npy')
            if np.all(np.load(file_list[0])[:,1:10]) and len(file_list)>10:
                self.temp_subj.append(self.subj[i])
        self.subj = self.temp_subj

        self.create_normalizer()
        self.data_split()
        self.load_normalizer()

        self.activity_label = {'sit':0,'lie':1,'walk':2,'stand':3}
        self.phone_label = {'hand':0,'pocket':1,'bag':2}

        self.n_subject = len(self.subj)
        self.file = {}
        for file_subj in self.subj:
            self.file[file_subj] = glob.glob(file_subj+'/*.npy')
        print(f'Number of subject: {len(self.file)}')

        assert len(self.mean)>0



class raw_Dataset_balanced_no_missing_augment(raw_Dataset_balanced_no_missing):
    """ Data loader"""

    def __getitem__(self, idx):
        
        offset=0
        while True:
            subj = self.subj[(idx+offset)%self.n_subject]
            event = self.file[subj]
            #activity
            act_select = list(self.activity_label.keys())[idx%4]
            event = [i for i in event if i.split('_')[-2]==act_select]
            if len(event)>0:
                break
            else:
                offset+=1
        
        idx_event = np.random.randint(len(event))
        signal = np.load(event[idx_event])
        signal_length = len(signal)
        try:
            start = np.random.randint(signal_length-self.hz*self.length)
            end = start+self.hz*self.length
            signal_out = signal[start:end,1:10]
        except:
            signal_out = np.zeros((self.hz*self.length,9))
            signal_out[:signal_length] = signal[:,1:10]
        signal_out = self.normalize(signal_out,subj.split('/')[-1])
        signal_out[:,0:3] = self.Augment.augment(signal_out[:,0:3])
        signal_out[:,3:6] = self.Augment.augment(signal_out[:,3:6])
        signal_out[:,6:9] = self.Augment.augment(signal_out[:,6:9])

        [_,act_label,phone_label] = event[idx_event].split('/')[-1][:-4].split('_')


        return torch.FloatTensor(signal_out.swapaxes(0,1)), torch.tensor(self.activity_label[act_label], dtype=torch.long),torch.tensor(self.phone_label[phone_label], dtype=torch.long) 



class raw_Dataset_subjGAN_no_missing(raw_Dataset_balanced_no_missing):
    """Augment each {activity,phone} to 10 samples per each subject,i.e., 120 samples per subjects with equal distribution"""

    def __init__(self, mode:str, n_fold:int, k:int, length:float, overlapping:float, hz:float, platform:str):

        self.datadir = '/home/aditthapron/DARPA/CRA/data/raw_feature_'+platform
        self.mode = mode
        self.k = k
        self.n_fold = n_fold
        self.length = length
        self.overlapping = overlapping
        self.hz=hz
        self.platform = platform

        self.n_augment=1

        self.subj = np.array(glob.glob(self.datadir+'/*'))

        self.delete_missing()
        self.create_normalizer()
        self.data_split()
        self.load_normalizer()

        self.activity_label = {'sit':0,'lie':1,'walk':2,'stand':3}
        self.phone_label = {'hand':0,'pocket':1,'bag':2}
        self.activity_idx = {v: k for k, v in self.activity_label.items()}
        self.phone_idx = {v: k for k, v in self.phone_label.items()}

        self.n_subject = len(self.subj)
        self.file = {}
        for file_subj in self.subj:
            self.file[file_subj] = glob.glob(file_subj+'/*.npy')

        print(f'Number of subject: {len(self.file)}')
        
        self.class_distribution()
        self.load_GAN()

        assert len(self.mean)>0
        assert len(self.file)>0

    def load_GAN(self):
        
        self.G_model = Generator(16).to(device)
        self.G_model.load_state_dict(torch.load('../subjectGAN/weight/{}_{}_{}.ckpt'.format(self.k,self.n_fold,self.platform)))
        self.G_model.eval()
        self.dvector = torch.jit.load('../dvector/weight/{}_{}_{}.pt'.format(self.k,self.n_fold,self.platform)).eval().to(device)

    def augment(self,org,tar):
        with torch.no_grad():
            emb = self.dvector.embed_utterance(torch.FloatTensor(tar).to(device)).detach().view(1,-1)
            org = torch.FloatTensor(org.reshape(1,org.shape[0],org.shape[1]).swapaxes(1,2)).to(device)
            return self.G_model(org,emb).detach().swapaxes(1,2)[0]

    def delete_missing(self):
        self.temp_subj = []
        for i in range(len(self.subj)):
            file_list = glob.glob(self.subj[i]+'/*.npy')
            if np.all(np.load(file_list[0])[:,1:10]) and len(file_list)>10:
                self.temp_subj.append(self.subj[i])
        self.subj = self.temp_subj

    def class_distribution(self):
        self.act_dist = {}
        self.all_combination_path = {}
        for i in range(4):
                for j in range(3):
                    self.all_combination_path[i,j]=[]

        for file_subj in self.subj:
            self.act_dist[file_subj] = {}
            for i in range(4):
                for j in range(3):
                    self.act_dist[file_subj][i,j]=self.n_augment

            for f in self.file[file_subj]:
                [_,act_label,phone_label] = f.split('/')[-1][:-4].split('_')
                self.act_dist[file_subj][self.activity_label[act_label],self.phone_label[phone_label]] -= 1
                if self.act_dist[file_subj][self.activity_label[act_label],self.phone_label[phone_label]]<0:
                    self.act_dist[file_subj][self.activity_label[act_label],self.phone_label[phone_label]]  =0  

                self.all_combination_path[self.activity_label[act_label],self.phone_label[phone_label]].append(f)

    def __getitem__(self, idx):
        
        if self.mode=='val':
            offset=0
            while True:
                subj = self.subj[(idx+offset)%self.n_subject]
                event = self.file[subj]
                #activity
                act_select = list(self.activity_label.keys())[idx%4]
                event = [i for i in event if i.split('_')[-2]==act_select]
                if len(event)>0:
                    break
                else:
                    offset+=1
            
            idx_event = np.random.randint(len(event))
            signal = np.load(event[idx_event])
            signal_length = len(signal)
            try:
                start = np.random.randint(signal_length-self.hz*self.length)
                end = start+self.hz*self.length
                signal_out = signal[start:end,1:10]
            except:
                signal_out = np.zeros((self.hz*self.length,9))
                signal_out[:signal_length] = signal[:,1:10]
            signal_out = self.normalize(signal_out,subj.split('/')[-1])

            [_,act_label,phone_label] = event[idx_event].split('/')[-1][:-4].split('_')

        else:
            subj = self.subj[idx%self.n_subject]
            temp = np.random.randint(12)
            act_select = temp//4
            phone_select = temp%3
            if np.random.random_sample()*self.n_augment <= self.act_dist[subj][act_select,phone_select]:
                #compute source
                src = self.all_combination_path[act_select,phone_select]
                src_file = src[np.random.randint(len(src))]
                subj_src= src_file.split('/')[-2]
                signal = np.load(src_file)
                signal_length = len(signal)
                try:
                    start = np.random.randint(signal_length-self.hz*self.length)
                    end = start+self.hz*self.length
                    signal_out = signal[start:end,1:10]
                except:
                    signal_out = np.zeros((self.hz*self.length,9))
                    signal_out[:signal_length] = signal[:,1:10]
                src_signal = self.normalize(signal_out,subj_src)

                #load target (current subject) file for d-vector extraction
                event = self.file[subj]
                idx_event = np.random.randint(len(event))
                signal = np.load(event[idx_event])
                signal_length = len(signal)
                try:
                    start = np.random.randint(signal_length-self.hz*self.length)
                    end = start+self.hz*self.length
                    signal_out = signal[start:end,1:10]
                except:
                    signal_out = np.zeros((self.hz*self.length,9))
                    signal_out[:signal_length] = signal[:,1:10]
                trg_signal = self.normalize(signal_out,subj.split('/')[-1])
                #augment
                signal_out = self.augment(src_signal,trg_signal)
                [_,act_label,phone_label] = src_file.split('/')[-1][:-4].split('_')

            else:

                event = self.file[subj]
                
                act_select = self.activity_idx[act_select]
                phone_select = self.phone_idx[phone_select]
                event = [i for i in event if i.split('/')[-1][:-4].split('_')[-2]==act_select and i.split('/')[-1][:-4].split('_')[-1]==phone_select]
                idx_event = np.random.randint(len(event))
                signal = np.load(event[idx_event])
                signal_length = len(signal)
                try:
                    start = np.random.randint(signal_length-self.hz*self.length)
                    end = start+self.hz*self.length
                    signal_out = signal[start:end,1:10]
                except:
                    signal_out = np.zeros((self.hz*self.length,9))
                    signal_out[:signal_length] = signal[:,1:10]
                signal_out = self.normalize(signal_out,subj.split('/')[-1])

                [_,act_label,phone_label] = event[idx_event].split('/')[-1][:-4].split('_')


        return torch.FloatTensor(signal_out.swapaxes(0,1)), torch.tensor(self.activity_label[act_label], dtype=torch.long),torch.tensor(self.phone_label[phone_label], dtype=torch.long) 


class raw_Dataset_contextGAN_no_missing(raw_Dataset_balanced_no_missing):
    """Augment each {activity,phone} to 10 samples per each subject,i.e., 120 samples per subjects with equal distribution"""

    def __init__(self, mode:str, n_fold:int, k:int, length:float, overlapping:float, hz:float, platform:str):
        self.datadir = '/home/aditthapron/DARPA/CRA/data/raw_feature_'+platform
        self.mode = mode
        self.k = k
        self.n_fold = n_fold
        self.length = length
        self.overlapping = overlapping
        self.hz=hz
        self.platform = platform

        self.n_augment=1

        self.subj = np.array(glob.glob(self.datadir+'/*'))

        self.delete_missing()
        self.create_normalizer()
        self.data_split()
        self.load_normalizer()

        self.activity_label = {'sit':0,'lie':1,'walk':2,'stand':3}
        self.phone_label = {'hand':0,'pocket':1,'bag':2}
        self.activity_idx = {v: k for k, v in self.activity_label.items()}
        self.phone_idx = {v: k for k, v in self.phone_label.items()}

        self.n_subject = len(self.subj)
        self.file = {}
        for file_subj in self.subj:
            self.file[file_subj] = glob.glob(file_subj+'/*.npy')

        print(f'Number of subject: {len(self.file)}')
        
        self.class_distribution()
        self.load_GAN()

        assert len(self.mean)>0
        assert len(self.file)>0

    def load_GAN(self):
        
        self.G_model = Generator(7).to(device)
        self.G_model.load_state_dict(torch.load('../contextGAN/weight/{}_{}_{}.ckpt'.format(self.k,self.n_fold,self.platform)))
        self.G_model.eval()

    def augment(self,org,context):
        with torch.no_grad():
            org = torch.FloatTensor(org.reshape(1,org.shape[0],org.shape[1]).swapaxes(1,2)).to(device)
            context = context.to(device).reshape(1,-1)
            return self.G_model(org,context).detach().swapaxes(1,2)[0]

    def delete_missing(self):
        self.temp_subj = []
        for i in range(len(self.subj)):
            file_list = glob.glob(self.subj[i]+'/*.npy')
            if np.all(np.load(file_list[0])[:,1:10]) and len(file_list)>10:
                self.temp_subj.append(self.subj[i])
        self.subj = self.temp_subj

    def class_distribution(self):
        self.act_dist = {}
        self.all_combination_path = {}
        for i in range(4):
                for j in range(3):
                    self.all_combination_path[i,j]=[]

        for file_subj in self.subj:
            self.act_dist[file_subj] = {}
            for i in range(4):
                for j in range(3):
                    self.act_dist[file_subj][i,j]=self.n_augment

            for f in self.file[file_subj]:
                [_,act_label,phone_label] = f.split('/')[-1][:-4].split('_')
                self.act_dist[file_subj][self.activity_label[act_label],self.phone_label[phone_label]] -= 1
                if self.act_dist[file_subj][self.activity_label[act_label],self.phone_label[phone_label]]<0:
                    self.act_dist[file_subj][self.activity_label[act_label],self.phone_label[phone_label]]  =0  

                self.all_combination_path[self.activity_label[act_label],self.phone_label[phone_label]].append(f)


    def __getitem__(self, idx):
        
        if self.mode=='val':
            offset=0
            while True:
                subj = self.subj[(idx+offset)%self.n_subject]
                event = self.file[subj]
                #activity
                act_select = list(self.activity_label.keys())[idx%4]
                event = [i for i in event if i.split('_')[-2]==act_select]
                if len(event)>0:
                    break
                else:
                    offset+=1
            
            idx_event = np.random.randint(len(event))
            signal = np.load(event[idx_event])
            signal_length = len(signal)
            try:
                start = np.random.randint(signal_length-self.hz*self.length)
                end = start+self.hz*self.length
                signal_out = signal[start:end,1:10]
            except:
                signal_out = np.zeros((self.hz*self.length,9))
                signal_out[:signal_length] = signal[:,1:10]
            signal_out = self.normalize(signal_out,subj.split('/')[-1])

            [_,act_label,phone_label] = event[idx_event].split('/')[-1][:-4].split('_')

        else:
            subj = self.subj[idx%self.n_subject]
            temp = np.random.randint(12)
            act_select = temp//4
            phone_select = temp%3
            if np.random.random_sample()*self.n_augment <= self.act_dist[subj][act_select,phone_select]:
                #compute source
                src = self.all_combination_path[act_select,phone_select]
                src_file = src[np.random.randint(len(src))]
                subj_src= src_file.split('/')[-2]
                signal = np.load(src_file)
                signal_length = len(signal)
                try:
                    start = np.random.randint(signal_length-self.hz*self.length)
                    end = start+self.hz*self.length
                    signal_out = signal[start:end,1:10]
                except:
                    signal_out = np.zeros((self.hz*self.length,9))
                    signal_out[:signal_length] = signal[:,1:10]
                src_signal = self.normalize(signal_out,subj_src)

                #load target (current subject) file for d-vector extraction
                event = self.file[subj]
                idx_event = np.random.randint(len(event))
                signal = np.load(event[idx_event])
                signal_length = len(signal)
                try:
                    start = np.random.randint(signal_length-self.hz*self.length)
                    end = start+self.hz*self.length
                    signal_out = signal[start:end,1:10]
                except:
                    signal_out = np.zeros((self.hz*self.length,9))
                    signal_out[:signal_length] = signal[:,1:10]
                src_signal = self.normalize(signal_out,subj.split('/')[-1])
                #augment
                #create one-hot with shape (7,) where 0-3 represents activity and 4-6 represents phone
                trg_context = np.zeros(7)
                trg_context[act_select] =1
                trg_context[phone_select+4] =1
                trg_context = torch.tensor(trg_context, dtype=torch.long)
                signal_out = self.augment(src_signal,trg_context)
                [_,act_label,phone_label] = src_file.split('/')[-1][:-4].split('_')

            else:

                event = self.file[subj]
                
                act_select = self.activity_idx[act_select]
                phone_select = self.phone_idx[phone_select]
                event = [i for i in event if i.split('/')[-1][:-4].split('_')[-2]==act_select and i.split('/')[-1][:-4].split('_')[-1]==phone_select]
                idx_event = np.random.randint(len(event))
                signal = np.load(event[idx_event])
                signal_length = len(signal)
                try:
                    start = np.random.randint(signal_length-self.hz*self.length)
                    end = start+self.hz*self.length
                    signal_out = signal[start:end,1:10]
                except:
                    signal_out = np.zeros((self.hz*self.length,9))
                    signal_out[:signal_length] = signal[:,1:10]
                signal_out = self.normalize(signal_out,subj.split('/')[-1])

                [_,act_label,phone_label] = event[idx_event].split('/')[-1][:-4].split('_')


        return torch.FloatTensor(signal_out.swapaxes(0,1)), torch.tensor(self.activity_label[act_label], dtype=torch.long),torch.tensor(self.phone_label[phone_label], dtype=torch.long) 


class features_Dataset(Dataset):
    """ Data loader"""
    def __init__(self, mode:str, n_fold:int, k:int, length:float, overlapping:float, hz:float, platform:str):

        super(features_Dataset, self).__init__()
        self.datadir = '/home/aditthapron/DARPA/CRA/data/raw_feature_'+platform
        self.mode = mode
        self.k = k
        self.n_fold = n_fold
        self.length = length
        self.overlapping = overlapping
        self.hz=hz
        self.platform = platform

        self.subj = np.array(glob.glob(self.datadir+'/*'))
        self.create_normalizer()
        self.data_split()
        self.load_normalizer()

        self.activity_label = {'sit':0,'lie':1,'walk':2,'stand':3}
        self.phone_label = {'hand':0,'pocket':1,'bag':2}

        self.n_subject = len(self.subj)
        self.file = {}
        for file_subj in self.subj:
            self.file[file_subj] = glob.glob(file_subj+'/*.npy')

        assert len(self.mean)>0


    def data_split(self):
        #perform subject-wise splitting
        np.random.seed(0)
        np.random.shuffle(self.subj)
        self.subj_testing = self.subj[self.k* (len(self.subj)//self.n_fold):(self.k+1)* (len(self.subj)//self.n_fold)]
        self.subj_training = np.array([s for s in self.subj if s not in self.subj_testing])
        if self.mode == 'train':
            self.subj = self.subj_training
        elif self.mode == 'val':
            self.subj = self.subj_testing
        else:
            raise NameError('Mode is incorrect')

    def create_normalizer(self):
        if not os.path.exists(self.datadir+'_normalize'):
            os.makedirs(self.datadir+'_normalize')
            for file_subj in self.subj:
                event_file = glob.glob(file_subj+'/*.npy')
                all_record = []
                for file in event_file:
                    all_record.append(np.load(file)[:,1:10])

                all_record = np.concatenate(all_record)
                mean_arr,std_arr =np.mean(all_record,axis=0),np.std(all_record,axis=0)
                np.save('{}_normalize/{}_mean.npy'.format(self.datadir,file_subj.split('/')[-1]),mean_arr)
                np.save('{}_normalize/{}_std.npy'.format(self.datadir,file_subj.split('/')[-1]),std_arr)

    def load_normalizer(self):
        self.mean = {}
        self.std = {}
        for file_subj in self.subj:
            subj_id = file_subj.split('/')[-1]
            self.mean[subj_id] = np.load('{}_normalize/{}_mean.npy'.format(self.datadir,subj_id))
            self.std[subj_id] = np.load('{}_normalize/{}_std.npy'.format(self.datadir,subj_id))
        self.mean_feature = np.load('{}_normalize_feature/feature_mean.npy'.format(self.datadir))
        self.std_feature = np.load('{}_normalize_feature/feature_std.npy'.format(self.datadir))

    def get_feature(self,signal):
        acc = np.vstack([feature_extraction(signal[i*self.hz:(i+1)*self.hz,0:3]) for i in range(len(signal)//self.hz)])
        gyro = np.vstack([feature_extraction(signal[i*self.hz:(i+1)*self.hz,3:6]) for i in range(len(signal)//self.hz)])
        mag = np.vstack([feature_extraction(signal[i*self.hz:(i+1)*self.hz,6:9]) for i in range(len(signal)//self.hz)])
        return np.nan_to_num(np.hstack([acc,gyro,mag]))

    def __getitem__(self, idx):
        
        subj = self.subj[idx%self.n_subject]
        event = self.file[subj]
        idx_event = np.random.randint(len(event))
        signal = np.load(event[idx_event])
        signal_length = len(signal)
        try:
            start = np.random.randint(signal_length-self.hz*self.length)
            end = start+self.hz*self.length
            signal_out = signal[start:end,1:]
        except:
            signal_out = np.zeros((self.hz*self.length,10))
            signal_out[:signal_length] = signal[:,1:]

        signal_out = signal_out[:,:-1]
        signal_out[:,:] = self.normalize(signal_out[:,:],subj.split('/')[-1])

        feature = self.get_feature(signal_out)
        feature = self.feature_normalize(feature)

        signal_out = np.repeat(signal_out,10,axis=0)
        [_,act_label,phone_label] = event[idx_event].split('/')[-1][:-4].split('_')

        return torch.FloatTensor(signal_out.swapaxes(0,1)), torch.FloatTensor(feature.swapaxes(0,1)), \
            torch.tensor(self.activity_label[act_label], dtype=torch.long).repeat(self.length),torch.tensor(self.phone_label[phone_label], dtype=torch.long).repeat(self.length), \
            torch.ones(1).repeat(self.length),torch.ones(1).repeat(self.length)

    def normalize(self,feature,subj):
        mean_arr,std_arr = self.mean[subj],self.std[subj]
        mean_arr,std_arr = np.tile(mean_arr,(len(feature),1)),np.tile(std_arr,(len(feature),1))
        return (feature-mean_arr)/(std_arr+1E-8)

    def feature_normalize(self,feature):
        return (feature-self.mean_feature)/(self.std_feature+1E-8)

    def __len__(self):
        return self.n_subject*10*3000


class features_Dataset_balanced(features_Dataset):
    """ Data loader"""
    
    def __getitem__(self, idx):
        
        offset=0
        while True:
            subj = self.subj[(idx+offset)%self.n_subject]
            event = self.file[subj]
            #activity
            act_select = list(self.activity_label.keys())[idx%4]
            event = [i for i in event if i.split('_')[-2]==act_select]
            if len(event)>0:
                break
            else:
                offset+=1
        
        idx_event = np.random.randint(len(event))
        signal = np.load(event[idx_event])
        signal_length = len(signal)

        try:
            start = np.random.randint(signal_length-self.hz*self.length)
            end = start+self.hz*self.length
            signal_out = signal[start:end,1:]
        except:
            signal_out = np.zeros((self.hz*self.length,10))
            signal_out[:signal_length] = signal[:,1:]
        
        signal_out = signal_out[:,:-1]
        signal_out[:,:] = self.normalize(signal_out[:,:],subj.split('/')[-1])

        feature = self.get_feature(signal_out)
        feature = self.feature_normalize(feature)

        [_,act_label,phone_label] = event[idx_event].split('/')[-1][:-4].split('_')

        signal_out = np.repeat(signal_out,10,axis=0)

        return torch.FloatTensor(signal_out.swapaxes(0,1)), torch.FloatTensor(feature.swapaxes(0,1)),  \
            torch.tensor(self.activity_label[act_label], dtype=torch.long).repeat(self.length),torch.tensor(self.phone_label[phone_label], dtype=torch.long).repeat(self.length), \
            torch.ones(1).repeat(self.length),torch.ones(1).repeat(self.length)



class features_Dataset_balanced_no_missing(features_Dataset_balanced):
    """ Data loader"""
    def __init__(self, mode:str, n_fold:int, k:int, length:float, overlapping:float, hz:float, platform:str):
        self.datadir = '/home/aditthapron/DARPA/CRA/data/raw_feature_'+platform
        self.mode = mode
        self.k = k
        self.n_fold = n_fold
        self.length = length
        self.overlapping = overlapping
        self.hz=hz
        self.platform = platform

        self.subj = np.array(glob.glob(self.datadir+'/*'))
        self.temp_subj = []
        for i in range(len(self.subj)):
            file_list = glob.glob(self.subj[i]+'/*.npy')
            if np.all(np.load(file_list[0])[:,1:10]) and len(file_list)>10:
                self.temp_subj.append(self.subj[i])
        self.subj = self.temp_subj
        
        self.create_normalizer()
        self.data_split()
        self.load_normalizer()

        self.activity_label = {'sit':0,'lie':1,'walk':2,'stand':3}
        self.phone_label = {'hand':0,'pocket':1,'bag':2}

        self.n_subject = len(self.subj)
        self.file = {}
        for file_subj in self.subj:
            self.file[file_subj] = glob.glob(file_subj+'/*.npy')
        print(f'Number of subject: {len(self.file)}')
        assert len(self.mean)>0

class features_Dataset_balanced_no_missing_normalized(features_Dataset_balanced_no_missing):
    def load_normalizer(self):
        self.mean = {}
        self.std = {}
        for file_subj in self.subj:
            subj_id = file_subj.split('/')[-1]
            self.mean[subj_id] = np.load('{}_normalize/{}_mean.npy'.format(self.datadir,subj_id))
            self.std[subj_id] = np.load('{}_normalize/{}_std.npy'.format(self.datadir,subj_id))

        #ES feature normalization
        for file_subj in self.subj:
            event_file = glob.glob(file_subj+'/*.npy')
            all_record = []
            for file in event_file:
                try:
                    all_record.append(self.get_feature(np.load(file)[:,1:10]))
                except:
                    pass
            all_record = np.concatenate(all_record)
            self.mean_feature,self.std_feature =np.mean(all_record,axis=0),np.std(all_record,axis=0)

    def __getitem__(self, idx):
        offset=0
        while True:
            subj = self.subj[(idx+offset)%self.n_subject]
            event = self.file[subj]
            #activity
            act_select = list(self.activity_label.keys())[idx%4]
            event = [i for i in event if i.split('_')[-2]==act_select]
            if len(event)>0:
                break
            else:
                offset+=1
        
        idx_event = np.random.randint(len(event))
        signal = np.load(event[idx_event])
        signal_length = len(signal)

        try:
            start = np.random.randint(signal_length-self.hz*self.length)
            end = start+self.hz*self.length
            signal_out = signal[start:end,1:]
        except:
            signal_out = np.zeros((self.hz*self.length,10))
            signal_out[:signal_length] = signal[:,1:]
        
        signal_out = signal_out[:,:-1]

        feature = self.get_feature(signal_out)
        feature = self.feature_normalize(feature)
        signal_out = self.normalize(signal_out,subj.split('/')[-1])
        [_,act_label,phone_label] = event[idx_event].split('/')[-1][:-4].split('_')

        signal_out = np.repeat(signal_out,10,axis=0)

        return torch.FloatTensor(signal_out.swapaxes(0,1)), torch.FloatTensor(feature.swapaxes(0,1)),  \
            torch.tensor(self.activity_label[act_label], dtype=torch.long).repeat(self.length),torch.tensor(self.phone_label[phone_label], dtype=torch.long).repeat(self.length), \
            torch.ones(1).repeat(self.length)/len(self.activity_label),torch.ones(1).repeat(self.length)/len(self.phone_label)

class features_Dataset_subjGAN_no_missing(features_Dataset_balanced_no_missing):
    """Augment each {activity,phone} to 10 samples per each subject,i.e., 120 samples per subjects with equal distribution"""

    def __init__(self, mode:str, n_fold:int, k:int, length:float, overlapping:float, hz:float, platform:str):

        super(features_Dataset_subjGAN_no_missing, self).__init__()
        self.datadir = '/home/aditthapron/DARPA/CRA/data/raw_feature_'+platform
        self.mode = mode
        self.k = k
        self.n_fold = n_fold
        self.length = length
        self.overlapping = overlapping
        self.hz=hz
        self.platform = platform

        self.n_augment=1

        self.subj = np.array(glob.glob(self.datadir+'/*'))

        self.delete_missing()
        self.create_normalizer()
        self.data_split()
        self.load_normalizer()

        self.activity_label = {'sit':0,'lie':1,'walk':2,'stand':3}
        self.phone_label = {'hand':0,'pocket':1,'bag':2}
        self.activity_idx = {v: k for k, v in self.activity_label.items()}
        self.phone_idx = {v: k for k, v in self.phone_label.items()}

        self.n_subject = len(self.subj)
        self.file = {}
        for file_subj in self.subj:
            self.file[file_subj] = glob.glob(file_subj+'/*.npy')

        print(f'Number of subject: {len(self.file)}')
        
        self.class_distribution()
        self.load_GAN()

        assert len(self.mean)>0
        assert len(self.file)>0

    def load_GAN(self):
        
        self.G_model = Generator(16).to(device)
        self.G_model.load_state_dict(torch.load('../subjectGAN/weight/{}_{}_{}.ckpt'.format(self.k,self.n_fold,self.platform)))
        self.G_model.eval()
        self.dvector = torch.jit.load('../dvector/weight/{}_{}_{}.pt'.format(self.k,self.n_fold,self.platform)).eval().to(device)

    def augment(self,org,tar):
        with torch.no_grad():
            emb = self.dvector.embed_utterance(torch.FloatTensor(tar).to(device)).detach().view(1,-1)
            org = torch.FloatTensor(org.reshape(1,org.shape[0],org.shape[1]).swapaxes(1,2)).to(device)
            return self.G_model(org,emb).detach().swapaxes(1,2)[0].numpy()

    def delete_missing(self):
        self.temp_subj = []
        for i in range(len(self.subj)):
            file_list = glob.glob(self.subj[i]+'/*.npy')
            if np.all(np.load(file_list[0])[:,1:10]) and len(file_list)>10:
                self.temp_subj.append(self.subj[i])
        self.subj = self.temp_subj

    def class_distribution(self):
        self.act_dist = {}
        self.all_combination_path = {}
        for i in range(4):
                for j in range(3):
                    self.all_combination_path[i,j]=[]

        for file_subj in self.subj:
            self.act_dist[file_subj] = {}
            for i in range(4):
                for j in range(3):
                    self.act_dist[file_subj][i,j]=self.n_augment

            for f in self.file[file_subj]:
                [_,act_label,phone_label] = f.split('/')[-1][:-4].split('_')
                self.act_dist[file_subj][self.activity_label[act_label],self.phone_label[phone_label]] -= 1
                if self.act_dist[file_subj][self.activity_label[act_label],self.phone_label[phone_label]]<0:
                    self.act_dist[file_subj][self.activity_label[act_label],self.phone_label[phone_label]]  =0  

                self.all_combination_path[self.activity_label[act_label],self.phone_label[phone_label]].append(f)

    def __getitem__(self, idx):
        
        if self.mode=='val':
            offset=0
            while True:
                subj = self.subj[(idx+offset)%self.n_subject]
                event = self.file[subj]
                #activity
                act_select = list(self.activity_label.keys())[idx%4]
                event = [i for i in event if i.split('_')[-2]==act_select]
                if len(event)>0:
                    break
                else:
                    offset+=1
            
            idx_event = np.random.randint(len(event))
            signal = np.load(event[idx_event])
            signal_length = len(signal)
            try:
                start = np.random.randint(signal_length-self.hz*self.length)
                end = start+self.hz*self.length
                step_raw = signal[start:end,10]
                signal_out = signal[start:end,1:10]
            except:
                signal_out = np.zeros((self.hz*self.length,9))
                step_raw = np.zeros((self.hz*self.length))
                step_raw [:signal_length] =  signal[:,10]
                signal_out[:signal_length] = signal[:,1:10]
            signal_out = self.normalize(signal_out,subj.split('/')[-1])

            [_,act_label,phone_label] = event[idx_event].split('/')[-1][:-4].split('_')

        else:
            subj = self.subj[idx%self.n_subject]
            temp = np.random.randint(12)
            act_select = temp//4
            phone_select = temp%3
            if np.random.random_sample()*self.n_augment <= self.act_dist[subj][act_select,phone_select]:
                #compute source
                src = self.all_combination_path[act_select,phone_select]
                src_file = src[np.random.randint(len(src))]
                subj_src= src_file.split('/')[-2]
                signal = np.load(src_file)
                signal_length = len(signal)
                try:
                    start = np.random.randint(signal_length-self.hz*self.length)
                    end = start+self.hz*self.length
                    step_raw = signal[start:end,10]
                    signal_out = signal[start:end,1:10]
                except:
                    signal_out = np.zeros((self.hz*self.length,9))
                    step_raw = np.zeros((self.hz*self.length))
                    step_raw [:signal_length] =  signal[:,10]
                    signal_out[:signal_length] = signal[:,1:10]
                src_signal = self.normalize(signal_out,subj_src)

                #load target (current subject) file for d-vector extraction
                event = self.file[subj]
                idx_event = np.random.randint(len(event))
                signal = np.load(event[idx_event])
                signal_length = len(signal)
                try:
                    start = np.random.randint(signal_length-self.hz*self.length)
                    end = start+self.hz*self.length
                    signal_out = signal[start:end,1:10]
                except:
                    signal_out = np.zeros((self.hz*self.length,9))
                    signal_out[:signal_length] = signal[:,1:10]
                trg_signal = self.normalize(signal_out,subj.split('/')[-1])
                #augment
                signal_out = self.augment(src_signal,trg_signal)
                [_,act_label,phone_label] = src_file.split('/')[-1][:-4].split('_')

            else:

                event = self.file[subj]
                
                act_select = self.activity_idx[act_select]
                phone_select = self.phone_idx[phone_select]
                event = [i for i in event if i.split('/')[-1][:-4].split('_')[-2]==act_select and i.split('/')[-1][:-4].split('_')[-1]==phone_select]
                idx_event = np.random.randint(len(event))
                signal = np.load(event[idx_event])
                signal_length = len(signal)
                try:
                    start = np.random.randint(signal_length-self.hz*self.length)
                    end = start+self.hz*self.length
                    step_raw = signal[start:end,10]
                    signal_out = signal[start:end,1:10]
                except:
                    signal_out = np.zeros((self.hz*self.length,9))
                    step_raw = np.zeros((self.hz*self.length))
                    step_raw [:signal_length] =  signal[:,10]
                    signal_out[:signal_length] = signal[:,1:10]
                signal_out = self.normalize(signal_out,subj.split('/')[-1])

                [_,act_label,phone_label] = event[idx_event].split('/')[-1][:-4].split('_')



        acc_feature = feature_extraction(signal_out[:,:3])
        gyro_feature = feature_extraction(signal_out[:,3:6])
        meg_feature = feature_extraction(signal_out[:,6:9])
        step_feature = np.ediff1d(step_raw)
        step_feature[-1] = step_feature[-2]

        feature = np.hstack((acc_feature,gyro_feature,meg_feature,np.mean(step_feature),np.std(step_feature)))
        feature = self.feature_normalize(feature)
        

        return torch.FloatTensor(signal_out.swapaxes(0,1)), torch.FloatTensor(feature.swapaxes(0,1)), torch.tensor(self.activity_label[act_label], dtype=torch.long),torch.tensor(self.phone_label[phone_label], dtype=torch.long) 



class features_Dataset_contextGAN_no_missing(features_Dataset_balanced_no_missing):
    """Augment each {activity,phone} to 10 samples per each subject,i.e., 120 samples per subjects with equal distribution"""

    def __init__(self, mode:str, n_fold:int, k:int, length:float, overlapping:float, hz:float, platform:str):

        super(features_Dataset_contextGAN_no_missing, self).__init__()
        self.datadir = '/home/aditthapron/DARPA/CRA/data/raw_feature_'+platform
        self.mode = mode
        self.k = k
        self.n_fold = n_fold
        self.length = length
        self.overlapping = overlapping
        self.hz=hz
        self.platform = platform

        self.n_augment=1

        self.subj = np.array(glob.glob(self.datadir+'/*'))

        self.delete_missing()
        self.create_normalizer()
        self.data_split()
        self.load_normalizer()

        self.activity_label = {'sit':0,'lie':1,'walk':2,'stand':3}
        self.phone_label = {'hand':0,'pocket':1,'bag':2}
        self.activity_idx = {v: k for k, v in self.activity_label.items()}
        self.phone_idx = {v: k for k, v in self.phone_label.items()}

        self.n_subject = len(self.subj)
        self.file = {}
        for file_subj in self.subj:
            self.file[file_subj] = glob.glob(file_subj+'/*.npy')

        print(f'Number of subject: {len(self.file)}')
        
        self.class_distribution()
        self.load_GAN()

        assert len(self.mean)>0
        assert len(self.file)>0

    def load_GAN(self):
        
        self.G_model = Generator(7).to(device)
        self.G_model.load_state_dict(torch.load('../contextGAN/weight/{}_{}_{}.ckpt'.format(self.k,self.n_fold,self.platform)))
        self.G_model.eval()

    def augment(self,org,context):
        with torch.no_grad():
            org = torch.FloatTensor(org.reshape(1,org.shape[0],org.shape[1]).swapaxes(1,2)).to(device)
            context = context.to(device).reshape(1,-1)
            return self.G_model(org,context).detach().swapaxes(1,2)[0].numpy()

    def delete_missing(self):
        self.temp_subj = []
        for i in range(len(self.subj)):
            file_list = glob.glob(self.subj[i]+'/*.npy')
            if np.all(np.load(file_list[0])[:,1:10]) and len(file_list)>10:
                self.temp_subj.append(self.subj[i])
        self.subj = self.temp_subj

    def class_distribution(self):
        self.act_dist = {}
        self.all_combination_path = {}
        for i in range(4):
                for j in range(3):
                    self.all_combination_path[i,j]=[]

        for file_subj in self.subj:
            self.act_dist[file_subj] = {}
            for i in range(4):
                for j in range(3):
                    self.act_dist[file_subj][i,j]=self.n_augment

            for f in self.file[file_subj]:
                [_,act_label,phone_label] = f.split('/')[-1][:-4].split('_')
                self.act_dist[file_subj][self.activity_label[act_label],self.phone_label[phone_label]] -= 1
                if self.act_dist[file_subj][self.activity_label[act_label],self.phone_label[phone_label]]<0:
                    self.act_dist[file_subj][self.activity_label[act_label],self.phone_label[phone_label]]  =0  

                self.all_combination_path[self.activity_label[act_label],self.phone_label[phone_label]].append(f)


    def __getitem__(self, idx):
        
        if self.mode=='val':
            offset=0
            while True:
                subj = self.subj[(idx+offset)%self.n_subject]
                event = self.file[subj]
                #activity
                act_select = list(self.activity_label.keys())[idx%4]
                event = [i for i in event if i.split('_')[-2]==act_select]
                if len(event)>0:
                    break
                else:
                    offset+=1
            
            idx_event = np.random.randint(len(event))
            signal = np.load(event[idx_event])
            signal_length = len(signal)
            try:
                start = np.random.randint(signal_length-self.hz*self.length)
                end = start+self.hz*self.length
                step_raw = signal[start:end,10]
                signal_out = signal[start:end,1:10]
            except:
                signal_out = np.zeros((self.hz*self.length,9))
                step_raw = np.zeros((self.hz*self.length))
                step_raw [:signal_length] =  signal[:,10]
                signal_out[:signal_length] = signal[:,1:10]
            signal_out = self.normalize(signal_out,subj.split('/')[-1])

            [_,act_label,phone_label] = event[idx_event].split('/')[-1][:-4].split('_')

        else:
            subj = self.subj[idx%self.n_subject]
            temp = np.random.randint(12)
            act_select = temp//4
            phone_select = temp%3
            if np.random.random_sample()*self.n_augment <= self.act_dist[subj][act_select,phone_select]:
                #compute source
                src = self.all_combination_path[act_select,phone_select]
                src_file = src[np.random.randint(len(src))]
                subj_src= src_file.split('/')[-2]
                signal = np.load(src_file)
                signal_length = len(signal)
                try:
                    start = np.random.randint(signal_length-self.hz*self.length)
                    end = start+self.hz*self.length
                    step_raw = signal[start:end,10]
                    signal_out = signal[start:end,1:10]
                except:
                    signal_out = np.zeros((self.hz*self.length,9))
                    step_raw = np.zeros((self.hz*self.length))
                    step_raw [:signal_length] =  signal[:,10]
                    signal_out[:signal_length] = signal[:,1:10]
                src_signal = self.normalize(signal_out,subj_src)

                #augment
                trg_context = np.zeros(7)
                trg_context[act_select] =1
                trg_context[phone_select+4] =1
                trg_context = torch.tensor(trg_context, dtype=torch.long)
                signal_out = self.augment(src_signal,trg_context)

                [_,act_label,phone_label] = src_file.split('/')[-1][:-4].split('_')

            else:

                event = self.file[subj]
                
                act_select = self.activity_idx[act_select]
                phone_select = self.phone_idx[phone_select]
                event = [i for i in event if i.split('/')[-1][:-4].split('_')[-2]==act_select and i.split('/')[-1][:-4].split('_')[-1]==phone_select]
                idx_event = np.random.randint(len(event))
                signal = np.load(event[idx_event])
                signal_length = len(signal)
                try:
                    start = np.random.randint(signal_length-self.hz*self.length)
                    end = start+self.hz*self.length
                    step_raw = signal[start:end,10]
                    signal_out = signal[start:end,1:10]
                except:
                    signal_out = np.zeros((self.hz*self.length,9))
                    step_raw = np.zeros((self.hz*self.length))
                    step_raw [:signal_length] =  signal[:,10]
                    signal_out[:signal_length] = signal[:,1:10]
                signal_out = self.normalize(signal_out,subj.split('/')[-1])

                [_,act_label,phone_label] = event[idx_event].split('/')[-1][:-4].split('_')


        acc_feature = feature_extraction(signal_out[:,:3])
        gyro_feature = feature_extraction(signal_out[:,3:6])
        meg_feature = feature_extraction(signal_out[:,6:9])
        step_feature = np.ediff1d(step_raw)
        step_feature[-1] = step_feature[-2]

        feature = np.hstack((acc_feature,gyro_feature,meg_feature,np.mean(step_feature),np.std(step_feature)))
        feature = self.feature_normalize(feature)
        

        return torch.FloatTensor(signal_out.swapaxes(0,1)), torch.FloatTensor(feature.swapaxes(0,1)), torch.tensor(self.activity_label[act_label], dtype=torch.long),torch.tensor(self.phone_label[phone_label], dtype=torch.long) 


class features_study1a(Dataset):
    """ Data loader"""
    def __init__(self, mode:str, n_fold:int, k:int, length:float, overlapping:float, hz:float, platform:str):

        super(features_study1a, self).__init__()
        self.datadir = '/home/aditthapron/DARPA/data/study1a'
        self.feature = '/home/aditthapron/DARPA/data/featurized_csvs/featurized_data_csvs_with_label_1Second'
        self.mode = mode
        self.k = k
        self.n_fold = n_fold
        self.length = length
        self.overlapping = overlapping
        self.hz=hz
        self.platform = platform

        self.loading_label()
        self.load_normalizer()
        self.n_subject = len(self.subj)
        print(self.n_subject)

        assert len(self.mean)>0

    def loading_label(self):
        self.subj = np.array(glob.glob(self.feature+'/*'))
        #perform subject-wise splitting
        np.random.seed(0)
        np.random.shuffle(self.subj)
        self.subj_testing = self.subj[self.k* (len(self.subj)//self.n_fold):(self.k+1)* (len(self.subj)//self.n_fold)]
        self.subj_training = np.array([s for s in self.subj if s not in self.subj_testing])

        if self.mode == 'train':
            self.subj = self.subj_training
        elif self.mode == 'val':
            self.subj = self.subj_testing
        else:
            raise NameError('Mode is incorrect')
        count_act = np.zeros(4)
        count_phone = np.zeros(3)
        #loading label, mean, std for standadize
        self.all_labels_features = {}
        for f in self.subj:
            df = pd.read_csv(f)
            selection = ((df['Lying Down']==1) | (df['Sitting']==1) | (df['Standing']==1) | (df['Walking']==1) ) & \
                ((df['Phone in Bag']==1) | (df['Phone in Hand']==1) | (df['Phone in Pocket']==1))
            df = df[selection]
            df['Activity'] = df.apply(f_act, axis=1)
            df['Phone'] = df.apply(f_phone, axis=1)
            unique,count = np.unique( df.apply(f_act, axis=1).to_numpy(),return_counts=True)
            count_act[unique] += count
            unique,count = np.unique( df.apply(f_phone, axis=1).to_numpy(),return_counts=True)
            count_phone[unique] += count
            self.all_labels_features[f.split('/')[-1].split('_')[3]] = df.loc[:, feature_name].fillna(0)
        self.weight_act = 1./count_act
        self.weight_act = self.weight_act/np.sum(self.weight_act)
        self.weight_phone = 1./count_phone
        self.weight_phone= self.weight_phone/np.sum(self.weight_phone)

    def load_normalizer(self):
        self.mean = {}
        self.std = {}

        subject_faeture_mean = []
        subject_faeture_std = []
        for file_subj in self.subj:
            subj_id = file_subj.split('/')[-1].split('_')[3]
            #recompute ES features
            for i in range(self.all_labels_features[subj_id].shape[0]):
                signal = self.get_feature(self.loadzip(subj_id,self.all_labels_features[subj_id].iloc[i,0],1))
                self.all_labels_features[subj_id].iloc[i,3:] = signal.flatten()
                self.all_labels_features[subj_id].dropna(axis=0)
            #raw stat
            self.mean[subj_id] = []
            for sensor in sensors_list:
                self.mean[subj_id].append(self.all_labels_features[subj_id].loc[:,sensor+':3d:mean_x'].mean())
                self.mean[subj_id].append(self.all_labels_features[subj_id].loc[:,sensor+':3d:mean_y'].mean())
                self.mean[subj_id].append(self.all_labels_features[subj_id].loc[:,sensor+':3d:mean_z'].mean())

            self.std[subj_id] = []
            for sensor in sensors_list:
                self.std[subj_id].append(self.all_labels_features[subj_id].loc[:,sensor+':3d:std_x'].mean())
                self.std[subj_id].append(self.all_labels_features[subj_id].loc[:,sensor+':3d:std_y'].mean())
                self.std[subj_id].append(self.all_labels_features[subj_id].loc[:,sensor+':3d:std_z'].mean())

            self.mean[subj_id] = np.array(self.mean[subj_id])
            self.std[subj_id] = np.array(self.std[subj_id])
            #feature stat
            subject_faeture_mean.append(self.all_labels_features[subj_id].mean(axis=0).loc[feature_name[3:]].to_numpy()) #omit [timestamp,act,phone]
            subject_faeture_std.append(self.all_labels_features[subj_id].std(axis=0).loc[feature_name[3:]].to_numpy())

        self.mean_feature = np.array(subject_faeture_mean).mean(axis=0)
        self.std_feature = np.array(subject_faeture_std).mean(axis=0)

    def time_query(self,arr,time_start,length):
        #arr is in format of [[time],[x],[y],[z]]
        #return [[x,y,z]]
        idx_start = int(np.sort(np.argwhere(arr[0]>time_start*1000))[0])
        idx_end = int(min(idx_start + length*self.hz,len(arr[0]),len(arr[1]),len(arr[2]),len(arr[3])))
        x = np.zeros((length*self.hz,3))
        for i in range(0,3):
            x[:idx_end-idx_start,i] = arr[i+1][idx_start:idx_end]

        return x

    def loadzip(self,UUID,time,length):
        #length is in seconds
        # time is timestamp in second
        file = sorted(glob.glob(self.datadir + "/UUID_"+str(UUID).zfill(4)+"/*"))

        for f in file:
            if int(f.split('/')[-1])>time:
                zip_file = glob.glob(f+'/*.zip')[0]
                break
        if 'zip_file' not in locals():
            #to handle file not found, used in preprocessing
            empty = np.zeros((self.hz*length,9))
            empty[:] = np.nan
            return empty
        #unzip file
        with zipfile.ZipFile(zip_file, 'r') as archive:
            with io.TextIOWrapper(archive.open("HF_DUR_DATA.txt"), encoding="utf-8") as f:
                js = json.loads(f.read())
                acc = np.array([np.array(js[name]) for name in [u'raw_acc_timestamp',u'raw_acc_x',u'raw_acc_y',u'raw_acc_z']],dtype=object)
                gyro = np.array([np.array(js[name]) for name in [u'processed_gyro_timestamp',u'processed_gyro_x',u'processed_gyro_y',u'processed_gyro_z']],dtype=object)
                mag = np.array([np.array(js[name]) for name in [u'raw_magnet_timestamp',u'raw_magnet_x',u'raw_magnet_y',u'raw_magnet_z']],dtype=object)

        acc = self.time_query(acc,time,length)
        gyro = self.time_query(gyro,time,length)
        mag = self.time_query(mag,time,length)

        return np.hstack([acc,gyro,mag]).astype(float)

    def get_feature(self,signal):
        acc = np.vstack([feature_extraction(signal[i*self.hz:(i+1)*self.hz,0:3]) for i in range(len(signal)//self.hz)])
        gyro = np.vstack([feature_extraction(signal[i*self.hz:(i+1)*self.hz,3:6]) for i in range(len(signal)//self.hz)])
        mag = np.vstack([feature_extraction(signal[i*self.hz:(i+1)*self.hz,6:9]) for i in range(len(signal)//self.hz)])
        return np.nan_to_num(np.hstack([acc,gyro,mag]))

    def __getitem__(self, idx):
        """ output : [N,C,length*self.hz]"""
        while True:
            subj = self.subj[idx%self.n_subject]
            subj_id = subj.split('/')[-1].split('_')[3]
            event = self.all_labels_features[subj_id]
            idx_event = np.random.randint(len(event))
            
            #get label
            act_label = event.iloc[idx_event:idx_event+self.length,1].to_numpy()
            phone_label = event.iloc[idx_event:idx_event+self.length,2].to_numpy()
            if len(act_label)<self.length:
                continue # Resample, due to signal ends before specified length
            #get time
            time = event.iloc[idx_event,0]

            signal_out = self.loadzip(subj_id,time,self.length)
            signal_out = np.nan_to_num(signal_out)

            feature = self.get_feature(signal_out)

            signal_out = self.normalize(signal_out,subj_id)
            feature = self.feature_normalize(feature)


            return torch.FloatTensor(signal_out).permute(1,0), torch.FloatTensor(feature).permute(1,0), torch.tensor(act_label, dtype=torch.long),torch.tensor(phone_label, dtype=torch.long), torch.FloatTensor(self.weight_act[act_label]), torch.FloatTensor(self.weight_phone[phone_label])


    def normalize(self,feature,subj):
        mean_arr,std_arr = self.mean[subj],self.std[subj]
        mean_arr,std_arr = np.tile(mean_arr,(len(feature),1)),np.tile(std_arr,(len(feature),1))
        return (feature-mean_arr)/(std_arr+1E-8)

    def feature_normalize(self,feature):
        return (feature-self.mean_feature)/(self.std_feature+1E-8)

    def __len__(self):
        return self.n_subject*10*3000


class features_study1a_10hz(features_study1a):
    """ Data loader"""

    def loadzip(self,UUID,time,length):
        #length is in seconds
        # time is timestamp in second
        file = sorted(glob.glob(self.datadir + "/UUID_"+str(UUID).zfill(4)+"/*"))

        for f in file:
            if int(f.split('/')[-1])>time:
                zip_file = glob.glob(f+'/*.zip')[0]
                break
        if 'zip_file' not in locals():
            #to handle file not found, used in preprocessing
            empty = np.zeros((self.hz*length,9))
            empty[:] = np.nan
            return empty
        #unzip file
        with zipfile.ZipFile(zip_file, 'r') as archive:
            with io.TextIOWrapper(archive.open("HF_DUR_DATA.txt"), encoding="utf-8") as f:
                js = json.loads(f.read())
                acc = np.array([np.array(js[name]) for name in [u'raw_acc_timestamp',u'raw_acc_x',u'raw_acc_y',u'raw_acc_z']],dtype=object)
                gyro = np.array([np.array(js[name]) for name in [u'processed_gyro_timestamp',u'processed_gyro_x',u'processed_gyro_y',u'processed_gyro_z']],dtype=object)
                mag = np.array([np.array(js[name]) for name in [u'raw_magnet_timestamp',u'raw_magnet_x',u'raw_magnet_y',u'raw_magnet_z']],dtype=object)

        acc = self.time_query(acc,time,length)
        gyro = self.time_query(gyro,time,length)
        mag = self.time_query(mag,time,length)
        #downsample to 5hz with same data format
        tensor = np.hstack([acc,gyro,mag]).astype(float)
        tensor = np.repeat(tensor[::5],5,axis=0)

        return tensor

class features_study1a_25hz(features_study1a):
    """ Data loader"""

    def loadzip(self,UUID,time,length):
        #length is in seconds
        # time is timestamp in second
        file = sorted(glob.glob(self.datadir + "/UUID_"+str(UUID).zfill(4)+"/*"))

        for f in file:
            if int(f.split('/')[-1])>time:
                zip_file = glob.glob(f+'/*.zip')[0]
                break
        if 'zip_file' not in locals():
            #to handle file not found, used in preprocessing
            empty = np.zeros((self.hz*length,9))
            empty[:] = np.nan
            return empty
        #unzip file
        with zipfile.ZipFile(zip_file, 'r') as archive:
            with io.TextIOWrapper(archive.open("HF_DUR_DATA.txt"), encoding="utf-8") as f:
                js = json.loads(f.read())
                acc = np.array([np.array(js[name]) for name in [u'raw_acc_timestamp',u'raw_acc_x',u'raw_acc_y',u'raw_acc_z']],dtype=object)
                gyro = np.array([np.array(js[name]) for name in [u'processed_gyro_timestamp',u'processed_gyro_x',u'processed_gyro_y',u'processed_gyro_z']],dtype=object)
                mag = np.array([np.array(js[name]) for name in [u'raw_magnet_timestamp',u'raw_magnet_x',u'raw_magnet_y',u'raw_magnet_z']],dtype=object)

        acc = self.time_query(acc,time,length)
        gyro = self.time_query(gyro,time,length)
        mag = self.time_query(mag,time,length)
        #downsample to 5hz with same data format
        tensor = np.hstack([acc,gyro,mag]).astype(float)
        tensor = np.repeat(tensor[::2],2,axis=0)

        return tensor

class features_study1a_5hz(features_study1a):
    """ Data loader"""

    def loadzip(self,UUID,time,length):
        #length is in seconds
        # time is timestamp in second
        file = sorted(glob.glob(self.datadir + "/UUID_"+str(UUID).zfill(4)+"/*"))

        for f in file:
            if int(f.split('/')[-1])>time:
                zip_file = glob.glob(f+'/*.zip')[0]
                break
        if 'zip_file' not in locals():
            #to handle file not found, used in preprocessing
            empty = np.zeros((self.hz*length,9))
            empty[:] = np.nan
            return empty
        #unzip file
        with zipfile.ZipFile(zip_file, 'r') as archive:
            with io.TextIOWrapper(archive.open("HF_DUR_DATA.txt"), encoding="utf-8") as f:
                js = json.loads(f.read())
                acc = np.array([np.array(js[name]) for name in [u'raw_acc_timestamp',u'raw_acc_x',u'raw_acc_y',u'raw_acc_z']],dtype=object)
                gyro = np.array([np.array(js[name]) for name in [u'processed_gyro_timestamp',u'processed_gyro_x',u'processed_gyro_y',u'processed_gyro_z']],dtype=object)
                mag = np.array([np.array(js[name]) for name in [u'raw_magnet_timestamp',u'raw_magnet_x',u'raw_magnet_y',u'raw_magnet_z']],dtype=object)

        acc = self.time_query(acc,time,length)
        gyro = self.time_query(gyro,time,length)
        mag = self.time_query(mag,time,length)
        #downsample to 5hz with same data format
        tensor = np.hstack([acc,gyro,mag]).astype(float)
        tensor = np.repeat(tensor[::10],10,axis=0)

        return tensor

def get_dataloader(loader_name,fold,k,length,hz,n_workers,platform):
    dataset_train = str_to_class(loader_name)(mode='train',n_fold=fold,k=k,length=length,overlapping=0,hz=hz,platform=platform)
    dataset_val = str_to_class(loader_name)(mode='val',n_fold=fold,k=k,length=length,overlapping=0,hz=hz,platform=platform)
    data_loader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=n_workers,drop_last=True)
    data_loader_val = DataLoader(dataset_val, batch_size=16, shuffle=True, num_workers=n_workers,drop_last=True)
    return data_loader_train,data_loader_val


if __name__ == '__main__':
    dataset_train = features_Dataset_balanced_no_missing(mode='train',n_fold=5,k=0,length=20,overlapping=150,hz=5,platform='android')
    data_loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=2,drop_last=True)
    train_iter = iter(data_loader_train)
    # dataset_train.debug()
    for i in range(1):
        out = next(train_iter)

