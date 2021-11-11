import torch
import torchvision.models
from torch import nn, Tensor, reshape, flatten
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from dataSet.customDataset import CustomTouhouDataset
from models.Vgg16 import VGGNet
from models.nn_Module import MyModule

summaryWriter = SummaryWriter()

if __name__ == '__main__':
    vgg16 = VGGNet(num_classes=158)
    train_data = CustomTouhouDataset("data/touhou/index.txt")
    data = DataLoader(train_data,)
    step = 0
    pbar = tqdm(data)
    if torch.cuda.is_available():
        vgg16 = vgg16.cuda()

    # for data in pbar:
    #     img, target = data
    #     gimg = img.cuda()
    #     gtarget = target.cuda()
    #     output: torch.Tensor = flatten(vgg16(gimg).cuda())
    #     summaryWriter.add_scalar(
    #         'vgg16', output[0], step
    #     )
    #     step += 1

    summaryWriter.close()
