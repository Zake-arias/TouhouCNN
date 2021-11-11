import re

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm, trange


class CustomTouhouDataset(Dataset):
    def delStr(self, line):
        line = re.sub('#.*', "", line)
        if line != "":
            return line.strip().split(',')
        else:
            return None

    def delNone(self, item: list):
        index = 0
        # fileLoadBar = tqdm(range(len(item)))
        for i in trange(len(item), desc='加载训练数据集'):
            if item[index] == ['']:
                item.pop(index)
            else:
                index += 1
        # fileLoadBar.close()
        return item

    def __init__(self, label_file_path,
                 transforms=transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor()])):
        with open(label_file_path, 'r', encoding='GB18030') as f:
            self.imgs = self.delNone(list(map(lambda line: self.delStr(line), f)))
            self.transforms = transforms

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.transforms(Image.open(path).convert('RGB'))
        label = str(label)
        return img, label

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)
