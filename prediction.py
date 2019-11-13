import torch
import cv2
import os
import numpy as np


def predict(image_dir=r'F:\lrb\TestSet(500)', save_dir=r'F:\lrb\prediction'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(r'D:\models\model.path')
    model = model.to(device)
    model.eval()

    image_list = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
    for image in image_list:
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        r,l = np.shape(img)
        # print(r,l)
        img = cv2.resize(img, (128, 128))
        # print(img.shape)
        # pdb
        # img = img[np.newaxis, 1,:, :]
        img = img[np.newaxis,:,:]
        img = img[np.newaxis,:,:,:]
        img = img/255.0
        img = img.astype(np.float32)
        # print(img.dtype)

        img = torch.from_numpy(img)
        # img = img.Tensor()
        img = img.to(device)
        with torch.no_grad():
            pred = model(img)
            pred = pred.cpu()
            pred = pred.numpy()
            # print(pred)
            pred = pred * 256.0
            pred = pred.astype(np.uint8)
            pred = pred[0,:,:,:]
            pred = pred[0, :, :]
            pred = cv2.resize(pred,(l,r))
            pred = pred * 255
            # print(pred.shape)
            # print(np.sum(pred))
            # for i in range(128):
            #     print(pred[i,:])


            # print(pred.dtype)
            # cv2.imshow('result',pred)
            # cv2.waitKey()

            # pred_ = cv2.resize(pred,(int(r),int(l)))
            # pred.save(os.path.join(save_dir, os.path.basename(image)), quality=95, subsampling=0)
            cv2.imwrite(os.path.join(save_dir,os.path.basename( image)),pred)
            print(f'Write {image} accomplishment')

predict()