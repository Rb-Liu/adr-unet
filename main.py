from datasets.dataset import MyDataset, TestDataset
from datasets.augmentation import *
from nets.model import UNet2
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms import transforms
from torch.nn import BCELoss
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import os
import cv2
from math import inf

def training(model, criterion, optimizer, device, train_dataloader, valid_dataloader, epoch_num, save_dir):

    def cal_loss(lists):
        a = torch.Tensor([0])
        for l in lists:
            a += l
        return a / len(lists)

    print('Training...')
    max_loss = inf
    for epoch in range(epoch_num):
        print(f"Epoch: {epoch + 1}|| {epoch_num}\n")
        with tqdm(train_dataloader, desc=f'train{epoch + 1}') as iterator:
            model.to(device)
            criterion.to(device)
            loss_list = []
            for loader in iterator:
                model.train()
                x = loader['image'].to(device)
                y = loader['mask'].to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss_list.append(loss)
                loss.backward()
                optimizer.step()
            print(f"Loss: {cal_loss(loss_list).detach().numpy()}\n")
            model.eval()
            model.cpu()
            criterion.cpu()
            valid_loss = []
            for v_loader in valid_dataloader:
                vx = v_loader['image']
                vy = v_loader['mask']
                with torch.no_grad():
                    vpred = model(vx)
                    vloss = criterion(vpred, vy)
                    valid_loss.append(vloss)
                    if vloss < max_loss:
                        torch.save(model, os.path.join(save_dir, 'model.path'))
                        max_loss = vloss
            print(f"Valid loss: {cal_loss(valid_loss).detach().numpy()}\n")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet2()
    optimizer = Adam(params=model.parameters(), lr=1e-4)

    criterion = BCELoss().to(device)
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
    train_dataset = MyDataset(r'F:\breast_cancer\train_data', r'F:\breast_cancer\train_mask', transform=train_transforms)
    dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0)
    valid_dataset = MyDataset(r'F:\breast_cancer\valid_data', r'F:\breast_cancer\valid_mask', transform=valid_transforms)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=0)
    training(model=model,
             criterion=criterion,
             optimizer=optimizer,
             device=device,
             train_dataloader=dataloader,
             valid_dataloader=valid_loader,
             epoch_num=300,
             save_dir=r'D:\models')






