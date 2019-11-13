from torch.utils.data import Dataset
from PIL import Image
import os


class MyDataset(Dataset):
    def __init__(self, data_dir, mask_dir, transform=None):
        self.data_paths, self.mask_paths = self.get_file(data_dir, mask_dir)
        self.transform = transform

    @staticmethod
    def get_file(data_dir, mask_dir):
        data_paths = []
        mask_paths = []
        for file in os.listdir(data_dir):
            data_paths.append(os.path.join(data_dir, file))
            mask_paths.append(os.path.join(mask_dir, file))
        return data_paths, mask_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image = Image.open(self.data_paths[idx]).convert('L').resize((128, 128), Image.ANTIALIAS)
        mask = Image.open(self.mask_paths[idx]).convert('L').resize((128, 128), Image.ANTIALIAS)

        sample = {'image': image, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample


class TestDataset(Dataset):
    def __init__(self, data_dir=r'F:\lrb\TestSet(500)'):
        self.data_paths = self.get_file(data_dir)

    @staticmethod
    def get_file(data_dir):
        data_paths = []

        for file in os.listdir(data_dir):
            data_paths.append(os.path.join(data_dir, file))

        return data_paths

    def __len__(self):
        return len(self.data_paths)

    @staticmethod
    def read_image(image):
        images = Image.open(image).convert('L').resize((128, 128), Image.ANTIALIAS)
        return images

    def __getitem__(self, idx):
        image = self.read_image(self.data_paths[idx])
        return image
