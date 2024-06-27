#!/usr/bin/env python
import os

from datasets import load_dataset
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import os
import argparse

import json
from shutil import copyfile

copyfile(src = os.path.join("/kaggle/input/vizwiz-dataset", 'vqa.py'), dst = os.path.join("../working", 'vqa.py'))
copyfile(src = os.path.join("/kaggle/input/vizwiz-dataset", 'prepare_data.py'), dst = os.path.join("../working", 'prepare_data.py'))

from vqa import *
from prepare_data import *

BASE_MODEL = "Salesforce/blip2-opt-2.7b"
import random

def split_dataset(dataset, train_ratio=0.7, valid_ratio=0.1, test_ratio=0.2):
    # Shuffle the dataset
    random.shuffle(dataset)

    # Calculate split indices
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)
    test_size = total_size - train_size - valid_size

    # Split the dataset
    train_set = dataset[:train_size]
    valid_set = dataset[train_size:train_size + valid_size]
    test_set = dataset[train_size + valid_size:]

    return train_set, valid_set, test_set

def load_dataset_vizwiz(data_path="/kaggle/input/vizwiz"):
    INPUT_PATH = data_path
    IMG_PATH = INPUT_PATH 
    ANNOTATIONS = INPUT_PATH + '/Annotations/Annotations'
    TRAIN_PATH = INPUT_PATH + '/train/train'
    VALIDATION_PATH = INPUT_PATH + '/val/val'
    TEST_PATH = INPUT_PATH + '/test/test'
    ANNOTATIONS_TRAIN_PATH = ANNOTATIONS + '/train.json'
    ANNOTATIONS_VAL_PATH = ANNOTATIONS + '/val.json'
    ANNOTATIONS_TEST_PATH = ANNOTATIONS + '/test.json'

    annFile = ANNOTATIONS_TRAIN_PATH
    imgDir = TRAIN_PATH

    # initialize VQA api for QA annotations
    data_VQA = {d_type:None for d_type in ['train','valid','test']}
    for d_type, a_path, d_path in zip(['train','valid','test'],
                            [TRAIN_PATH,VALIDATION_PATH,TEST_PATH], 
                            [ANNOTATIONS_TRAIN_PATH,ANNOTATIONS_VAL_PATH,ANNOTATIONS_TEST_PATH]):
        annFile = d_path
        imgDir = a_path

        # initialize VQA api for QA annotations
        vqa=VQA(annFile)
        
        # load and display QA annotations for given answer types
        """
        ansTypes can be one of the following
        yes/no
        number
        other
        unanswerable
        """
        anns = vqa.getAnns(ansTypes=['other','yes/no','number']);  
        anns = vqa.getBestAnns(ansTypes=['other','yes/no','number']);  

        data_VQA[d_type] = anns

    train_n, valid_n = len(data_VQA['train']), len(data_VQA['valid'])

    data_VQA['train'] = data_VQA['train'][:10000]
    data_VQA['valid'] = data_VQA['valid'][:1000]
    print("Training sets: {}->{} - Validating set: {}->{}".format(train_n, len(data_VQA['train']), valid_n, len(data_VQA['valid'])))

    return data_VQA, TRAIN_PATH, VALIDATION_PATH
    # train_dataset = VQADataset(dataset=data_VQA['train'],
    #                         processor=processor,
    #                         img_path=TRAIN_PATH)
    # valid_dataset = VQADataset(dataset=data_VQA['valid'],
    #                         processor=processor,
    #                         img_path=VALIDATION_PATH)

    # batch_size = 1
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # return train_dataloader, valid_dataloader

def load_dataset_kvqa(data_path:str="/kaggle/input/vqa-blind-ko"):
    INPUT_PATH = data_path
    TRAIN_PATH = INPUT_PATH + '/VQA_train/images'
    ANNOTATIONS = INPUT_PATH 
    TEST_PATH = INPUT_PATH + '/VQA_test/task07_images'
    ANNOTATIONS_TRAIN_PATH = ANNOTATIONS + '/train_en.json'
    ANNOTATIONS_TEST_PATH = TEST_PATH + '/test.json'

    annFile = ANNOTATIONS_TRAIN_PATH

    # initialize VQA api for QA annotations
    vqa=VQA(annFile)
    
    # load and display QA annotations for given answer types
    """
    ansTypes can be one of the following
    yes/no
    number
    other
    unanswerable
    """
    anns = vqa.getAnns();  
    anns = vqa.getBestAnns();  

    # Split the dataset into train, validation, and test sets
    train_set, valid_set, test_set = split_dataset(anns)
    train_n, valid_n = len(train_set), len(valid_set)
    train_set = train_set[:20000]
    valid_set = valid_set[:2000]
    data_VQA = {
        'train': train_set,
        'valid': valid_set,
        'test': test_set
    }
    print("Training sets: {}->{} - Validating set: {}->{}".format(train_n, len(data_VQA['train']), valid_n, len(data_VQA['valid'])))

    return data_VQA, TRAIN_PATH, TRAIN_PATH

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--kaggle", type=bool, required=True, default=True)
#     parser.add_argument("--vizwiz_path", type=str, required=True, default="/kaggle/input/vizwiz")
#     parser.add_argument("--kvqa_path", type=str, required=True, default="/kaggle/input/vqa-blind-ko")
#     parser.add_argument("--lib_path", type=str, required=True, default="/kaggle/input/vizwiz-dataset")
#     args = parser.parse_args()

#     # load_dataset()
#     if args.kaggle:
#         from shutil import copyfile
#         copyfile(src = os.path.join(args.lib_path, 'vqa.py'), dst = os.path.join("../working", 'vqa.py'))

#         from vqa import *
#         load_dataset()