import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.dataset import MyDataset
from datasets.augmentation import *
from utils.loss import BCEDiceLoss
from utils.Metrics import IouMetric, FscoreMetric
from nets.model import UNet2
from utils.train import TrainEpoch, ValidEpoch


def training(model, loss, metrics):
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    train_epoch = TrainEpoch(model=model, loss=loss, metrics=metrics, optimizer=optimizer, device='cuda', verbose=True)
    valid_epoch = ValidEpoch(model=model, loss=loss, metrics=metrics, device='cuda', verbose=True)
    max_score = 0
    score_list = []
    for i in range(300):
        print(f'\nEpoch: {i}')
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        score_list.append(train_logs['iou'])
        if (i + 1) % 100 == 0:
            optimizer.param_groups[0]['lr'] *= 0.5
            print(f"Decrease learning rate to {optimizer.param_groups[0]['lr']}")
        if max_score < valid_logs['iou']:
            torch.save(model, os.path.join(SAVE_DIR, 'best_model.pth'))
            print('Model saved.')
    torch.save(model, os.path.join(SAVE_DIR, 'final_model.pth'))
if __name__ == '__main__':
    TRAIN_DATA_DIR = r'F:\breast_cancer\train_data'
    TRAIN_MASK_DIR = r'F:\breast_cancer\train_mask'
    VALID_DATA_DIR = r'F:\breast_cancer\valid_data'
    VALID_MASK_DIR = r'F:\breast_cancer\valid_mask'
    SAVE_DIR = r'G:\ADRUNET\result\models'

    LEARNING_RATE = 1e-4

    train_transforms = Compose([

        RandomHorizontalFlip(),
        RandomHorizontalFlip(),
        RandomRotation(degrees=3),
        RandomPerspective(),
        RandomAffine(degrees=3),
        ColorJitter(),
        ToTensor()
    ])
    valid_transforms = ToTensor()
    train_dataset = MyDataset(TRAIN_DATA_DIR, TRAIN_MASK_DIR, transform=train_transforms)
    valid_dataset = MyDataset(VALID_DATA_DIR, VALID_MASK_DIR, transform=valid_transforms)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4)

    loss = BCEDiceLoss()
    metrics = [
        IouMetric(),
        FscoreMetric()
    ]
    model = UNet2()
    training(model, loss, metrics)



