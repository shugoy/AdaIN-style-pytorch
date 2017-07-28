import argparse
import os
import sys
import time

from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.serialization import load_lua
from torchvision import datasets
from torchvision import transforms
from torchvision import models

from vgg import Vgg16
from AdaIN import AdaIN

# def main():
parser = argparse.ArgumentParser()
parser.add_argument("--contentImg", default="images/hoovertowernight.jpg")
parser.add_argument("--styleImg", default="images/starry_night_google.jpg")
parser.add_argument("--vgg", default="models/vgg_normalised.t7")
parser.add_argument("--decoder", default="models/decoder.t7")
parser.add_argument("--cuda", default=False)
args = parser.parse_args()

# outputImg = styleTransfer(contentImg, styleImg)
# vgg = models.vgg16(pretrained=True)
vgg = Vgg16(requires_grad=False)
# vgg = load_lua(args.vgg)
# vgg.cuda.
# vgg.register_forward_hook

# vgg.features.
adain = AdaIN(4, False, 1e-5)

decoder = load_lua(args.decoder)


## load images
contentImg = Image.open(args.contentImg)
styleImg = Image.open(args.styleImg)

content_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])
content = content_transform(contentImg)
style   = content_transform(styleImg)

content = content.unsqueeze(0)
style   = style.unsqueeze(0)

if args.cuda:
    content = content.cuda()
    style   = style.cuda()

content = Variable(content, volatile=True)
style   = Variable(style, volatile=True)

styleFeature   = vgg.forward(style)
contentFeature = vgg.forward(content)

targetFeature = adain.updateOutput(contentFeature, styleFeature)
targetFeature = targetFeature.squeeze()


Image.save(args.outputImg, outputImg)