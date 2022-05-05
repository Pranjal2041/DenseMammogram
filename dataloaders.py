# Get the dataloaders
# There are only two types of dataloaders, viz. VanillaFRCNN and BilaterialFRCNN

import torch
import cv2
import torchvision.transforms as T
import detection.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import detection.utils as utils
import os
from tqdm import tqdm
import pandas as pd
from os.path import join
# VanillaFRCNN DataLoaders

class FRCNNDataset(Dataset):
    def __init__(self,inputs,transform):
        self.transform = transform
        self.dataset_dicts = inputs

    def __len__(self):
        return len(self.dataset_dicts)


    def __getitem__(self,index: int):
        # Select the sample
        record = self.dataset_dicts[index]
        # Load input and target
        img = cv2.imread(record['file_name'])

        target = {k:torch.tensor(v) for k,v in record.items() if k != 'file_name'}
        if self.transform is not None:
            img = T.ToPILImage()(img)
            img,target = self.transform(img,target)

        return img,target

def xml_to_dicts(paths):
    dataset_dicts = []
    i=1
    for path in paths:
        for image in tqdm(os.listdir(os.path.join(path,'mal/images/'))):
            xmlfile = os.path.join(path,'mal/gt/',image[:-4]+'.txt')
            if(not os.path.exists(xmlfile)):
                continue
            img = cv2.imread(os.path.join(path,'mal/images/',image))
            record = {}
            record['file_name'] = os.path.join(path , 'mal/images/',image)
            record['image_id'] = i
            i+=1
            record['width'] = img.shape[1]
            record['height'] = img.shape[0]
            objs = []
            boxes = []
            labels = []
            area = []
            iscrowd = []
            f = open(xmlfile,'r')
            for line in f.readlines():
                box = list(map(int,map(float,line.split()[1:])))
                boxes.append(box)
                labels.append(1)
                area.append((box[2]-box[0])*(box[3]-box[1]))
                iscrowd.append(False)
            f.close()
            record["boxes"] = boxes
            record["labels"] = labels
            record["area"] = area
            record["iscrowd"] = iscrowd
            if(len(boxes)>0):
                dataset_dicts.append(record)
        for image in tqdm(os.listdir(os.path.join(path,'ben/images/'))):
            img = cv2.imread(os.path.join(path,'ben/images/',image))
            record = {}
            record['file_name'] = os.path.join(path, 'ben/images/',image)
            record['image_id'] = i
            i+=1
            record['width'] = img.shape[1]
            record['height'] = img.shape[0]
            record['boxes'] = torch.tensor([[0,0,img.shape[1],img.shape[0]]])
            record['labels'] = torch.tensor([0])
            record['area'] = [img.shape[1]*img.shape[0]]
            record["iscrowd"] = [False]
            dataset_dicts.append(record)
    return dataset_dicts



def get_FRCNN_dataloaders(cfg, batch_size = 2, data_dir = '../bilateral_new',):
    transform_test = transforms.Compose([transforms.ToTensor()])
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    train_paths = [join(data_dir,cfg['AIIMS_DATA'],cfg['AIIMS_TRAIN_SPLIT']),join(data_dir,cfg['DDSM_DATA'],cfg['DDSM_TRAIN_SPLIT']),]
    val_aiims_path = [join(data_dir,cfg['AIIMS_DATA'],cfg['AIIMS_VAL_SPLIT'])]
    train_data = FRCNNDataset(xml_to_dicts(train_paths),transform_train)
    test_aiims = FRCNNDataset(xml_to_dicts(val_aiims_path),transform_test)

    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=4,collate_fn = utils.collate_fn)
    test_aiims_loader = DataLoader(test_aiims,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=4,collate_fn = utils.collate_fn)
    #test_ddsm_loader = DataLoader(test_ddsm,batch_size=2,shuffle=True,drop_last=True,num_workers=5,collate_fn = utils.collate_fn)

    return train_loader, test_aiims_loader

# BilaterialFRCNN DataLoaders

def get_direction(dset,file_name):
        # 1 if right else -1
        if dset == 'aiims' or dset == 'ddsm':
            file_name = file_name.lower()
            r = file_name.find('right')
            l = file_name.find('left')
            if l == r and l == -1:
                raise Exception(f'Unidentifiable Direction {file_name}')
            if l!=-1 and r!=-1:
                raise Exception(f'Unidentifiable Direction {file_name}')
            return 1 if r!=-1 else -1
        if dset == 'inbreast':
            dir =file_name.split('_')[3]
            if dir == 'R': return 1
            if dir == 'L': return -1
            raise Exception(f'Unidentifiable Direction {file_name}')
        if dset == 'irch': 
            r = file_name.find('_R ')
            l = file_name.find('_L ')
            if l == r and l == -1:
                raise Exception(f'Unidentifiable Direction {file_name}')
            if l!=-1 and r!=-1:
                raise Exception(f'Unidentifiable Direction {file_name}')
            return 1 if r!=-1 else -1


class BilateralDataset(torch.utils.data.Dataset):

    def __init__(self,inputs,transform,dset):
        self.transform = transform
        self.dataset_dicts = inputs
        self.dset = dset

    def __len__(self):
        return len(self.dataset_dicts)
    

    def __getitem__(self,index: int):
        # Select the sample
        record = self.dataset_dicts[index]
        # Load input and target
        img1 = cv2.imread(record['file_name'])
        img2 = cv2.imread(record['file_2'])

        target = {k:torch.tensor(v) for k,v in record.items() if k != 'file_name' and k!='file_2'}
        if self.transform is not None:
            img1 = T.ToPILImage()(img1)
            img2 = T.ToPILImage()(img2)
            if(get_direction(self.dset,record['file_name'].split('/')[-1])==1):
                img1,target = transforms.RandomHorizontalFlip(1.0)(img1,target)
            else:
                img2,_ = transforms.RandomHorizontalFlip(1.0)(img2)
            img1,target = self.transform(img1,target)
            img2,target = self.transform(img2,target)

        images = [img1,img2]
        return images,target


def xml_to_dicts_bilateral(paths,cor_dicts):
    dataset_dicts = []
    i=1
    for path,cor_dict in zip(paths,cor_dicts):
        for image in tqdm(os.listdir(os.path.join(path,'mal/images/'))):
            if(not os.path.join(path,'mal/images/',image) in cor_dict):
                continue
            if(not os.path.isfile(cor_dict[os.path.join(path,'mal/images/',image)])):
                continue
            xmlfile = os.path.join(path,'mal/gt/',image[:-4]+'.txt')
            if(not os.path.exists(xmlfile)):
                continue
            img = cv2.imread(os.path.join(path,'mal/images/',image))
            
            record = {}
            record['file_name'] = os.path.join(path , 'mal/images/',image)
            record['file_2'] = cor_dict[os.path.join(path,'mal/images/',image)]
            record['image_id'] = i
            i+=1
            record['width'] = img.shape[1]
            record['height'] = img.shape[0]
            objs = []
            boxes = []
            labels = []
            area = []
            iscrowd = []
            f = open(xmlfile,'r')
            for line in f.readlines():
                box = list(map(int,map(float,line.split()[1:])))
                boxes.append(box)
                labels.append(1)
                area.append((box[2]-box[0])*(box[3]-box[1]))
                iscrowd.append(False)

            f.close()
            record["boxes"] = boxes
            record["labels"] = labels
            record["area"] = area
            record["iscrowd"] = iscrowd
            if(len(boxes)>0):
                dataset_dicts.append(record)

        for image in tqdm(os.listdir(os.path.join(path,'ben/images/'))):
            if(not os.path.join(path,'ben/images/',image) in cor_dict):
                continue
            if(not os.path.isfile(cor_dict[os.path.join(path,'ben/images/',image)])):
                continue
            img = cv2.imread(os.path.join(path,'ben/images/',image))

            record = {}
            record['file_name'] = os.path.join(path , 'ben/images/',image)
            record['file_2'] = cor_dict[os.path.join(path,'ben/images/',image)]
            img2 = cv2.imread(cor_dict[os.path.join(path,'ben/images/',image)])
            record['image_id'] = i
            i+=1
            record['width'] = img.shape[1]
            record['height'] = img.shape[0]

            record["boxes"] = torch.tensor([[0,0,min(img.shape[1],img2.shape[1]),min(img.shape[0],img2.shape[0])]])
            record['labels'] = torch.tensor([0])
            record['area'] = [ min(img.shape[1],img2.shape[1]) *min(img.shape[0],img2.shape[0])]
            record["iscrowd"] = [False]
            if(len(boxes)>0):
               dataset_dicts.append(record)

    return dataset_dicts



def get_dict(data_dir, filename):
    df = pd.read_csv(filename, header=None, sep=r'\s+', quotechar='"').to_numpy()
    cor_dict = dict()
    for a in df:
      if(a[0]==a[1]):
        continue
      cor_dict[a[0]] = a[1]
    # print(cor_dict)
    cor_dict = {join(data_dir,k):join(data_dir,v) for k,v in cor_dict.items()}
    return cor_dict

def get_bilateral_dataloaders(cfg, batch_size = 1, data_dir = '../bilateral_new'):
    transform_test = transforms.Compose([transforms.ToTensor()])
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    train_paths = [join(data_dir,cfg['AIIMS_DATA'],cfg['AIIMS_TRAIN_SPLIT']),join(data_dir,cfg['DDSM_DATA'],cfg['DDSM_TRAIN_SPLIT']),]
    val_aiims_path = [join(data_dir,cfg['AIIMS_DATA'],cfg['AIIMS_VAL_SPLIT'])]
    cor_lists_train = [get_dict(data_dir,join(data_dir,cfg['AIIMS_CORRS_LIST'])),get_dict(data_dir,join(data_dir,cfg['DDSM_CORRS_LIST']))]
    cor_lists_val = [get_dict(data_dir,join(data_dir,cfg['AIIMS_CORRS_LIST']))]
    cor_lists_train = [get_dict(data_dir,join(data_dir,cfg['AIIMS_CORRS_LIST']))]
    train_data = BilateralDataset(xml_to_dicts_bilateral(train_paths,cor_lists_train),transform_test,'aiims')
    val_aiims = BilateralDataset(xml_to_dicts_bilateral(val_aiims_path,cor_lists_val),transform_test,'aiims')

    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=4,collate_fn = utils.collate_fn)
    val_aiims_loader = DataLoader(val_aiims,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=4,collate_fn = utils.collate_fn)
    #test_ddsm_loader = DataLoader(test_ddsm,batch_size=2,shuffle=True,drop_last=True,num_workers=5,collate_fn = utils.collate_fn)

    return train_loader, val_aiims_loader