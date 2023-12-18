from torch.utils.data import Dataset
import numpy as np
from scipy import io

import torch
import torchvision.transforms as transforms
from PIL import Image
import glob

class wdm(Dataset):
    def __init__(self, path, height=52, width=52, augmentation=False, task='train'):
        super().__init__()
        assert task == 'train' or task == 'val' or task == 'test', f'Invalid task....'

        self.img_size = (height, width)
        self.augmentation = augmentation
        self.task = task

        self.data_paths = sorted(glob.glob(path+'*.mat'))
        self.fn_transform = self.get_transformer()

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_dict = io.loadmat(self.data_paths[index])

        image_rgb_numpy = self.mat_to_rgb(data_dict['data_wdm']['waferDefectMap'][0][0])
        image_rgb_pil = Image.fromarray(image_rgb_numpy.astype(dtype=np.uint8)).convert("RGB")
        image_tensor = self.fn_transform(image_rgb_pil).unsqueeze(dim=0)

        label_numpy = torch.tensor(data_dict['data_wdm']['label_1'][0][0].transpose())

        return image_tensor, label_numpy

    @staticmethod
    def mat_to_rgb(image):
        image_rgb = np.zeros([image.shape[0], image.shape[1], 3])
        image_rgb[np.where(image == 0)] = (0, 0, 0)
        image_rgb[np.where(image == 1)] = (71, 100, 100)
        image_rgb[np.where(image == 2)] = (255, 228, 0)

        return image_rgb

    def get_transformer(self):
        if self.augmentation:
            if self.task=='train':
                fn_trans = transforms.Compose([transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.RandomVerticalFlip(),
                                               transforms.RandomRotation(degrees=(0,90)),
                                               transforms.ToTensor()])
            else:
                fn_trans = transforms.Compose([transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST),
                                               transforms.ToTensor()])
        else:
            fn_trans = transforms.Compose([transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST),
                                          transforms.ToTensor()])

        return fn_trans

    @staticmethod
    def collate_fn(batch):
        inputs, labels = zip(*batch)

        inputs = torch.cat(inputs, dim=0)
        labels = torch.cat(labels, dim=0)

        return inputs, labels


if __name__ == '__main__':
    object = wdm(path='/Users/ssun/abstract_structure/course_data/train_aug/',
                       height=52,
                       width=52,
                       augmentation = False,
                       task = 'train')

    img, label = object.__getitem__(5409)

    loader = torch.utils.data.DataLoader(object,
                                         batch_size=5,
                                         shuffle=True,
                                         collate_fn=wdm.collate_fn,
                                         )

    for batch_id, data in enumerate(loader):
        image, label = data[0], data[1]