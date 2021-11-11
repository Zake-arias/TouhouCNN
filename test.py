import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm

from dataSet.customDataset import CustomTouhouDataset

if __name__ == '__main__':
    summaryWriter = SummaryWriter('data/logs/vgg16_demo')
    train_data = CustomTouhouDataset("data/touhou/index/index.csv",)
    data = DataLoader(train_data, )
    for item in data:
        data, label = item
        data = torch.reshape(data, (3, 224, 224))
        print(label[0])
        print(data)
        summaryWriter.add_image(str(label), data)
    summaryWriter.close()
    print('done')
