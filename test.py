import argparse
import os
import sys
import time

from PIL import Image

import numpy as np
import torch
import torch.nn as nn
# import torch.legacy.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.serialization import load_lua
from torchvision import datasets
from torchvision import transforms
from torchvision import models

# from vgg import Vgg16
# import vgg
from AdaIN import AdaIN

# def main():
parser = argparse.ArgumentParser()
parser.add_argument("--contentImg", default="images/brad_pitt.jpg")
parser.add_argument("--styleImg", default="images/picasso_self_portrait.jpg")
parser.add_argument("--outputImg", default="images/out.jpg")
parser.add_argument("--vgg", default="models/vgg_normalised.t7")
parser.add_argument("--decoder", default="models/decoder.t7")
parser.add_argument("--cuda", default=False)
parser.add_argument("--alpha", default=1)#, 'The weight that controls the degree of stylization. Should be between 0 and 1')
args = parser.parse_args()

# vgg = models.vgg16(pretrained=True).features
# vgg = load_lua(args.vgg)
import vgg_normalised
vgg = vgg_normalised.vgg_normalised
vgg.load_state_dict(torch.load('vgg_normalised.pth'))
# for i in reversed(range(31,52)):
#     vgg.reduce(i)
vgg = nn.Sequential(
            *(vgg[i] for i in range(31)))
# print vgg
# vgg.features = nn.Sequential(
#                     # stop at conv4
#                     *list(vgg.features.children())[:-3]
#                 )
# vgg.eval()
# vgg.cuda.

## load images
contentImg = Image.open(args.contentImg)
styleImg = Image.open(args.styleImg)

content_transform = transforms.Compose([
    transforms.Scale(512),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.mul(255))
])
content = content_transform(contentImg)
style   = content_transform(styleImg)
# print "content_transform {0}".format(content.size())

# content = Variable(content, volatile=True, requires_grad=False)
# style   = Variable(style, volatile=True, requires_grad=False)
content = Variable(content)
style   = Variable(style)

content = content.unsqueeze(0)
style   = style.unsqueeze(0)

if args.cuda:
    content = content.cuda()
    style   = style.cuda()

print "content.size() {0}".format(content.size()) #torch.Size([1, 3, 400, 760])
# print "style.size() {0}".format(style.size())  #torch.Size([1, 3, 1014, 1280])
print "vgg.forward.."
styleFeature   = vgg.forward(style)
contentFeature = vgg.forward(content)
print "contentFeature {0}".format(contentFeature.size())

# AdaIN
adain = AdaIN(contentFeature.size(1), False, 1e-5)
targetFeature = adain.updateOutput(contentFeature, styleFeature)
# targetFeature = contentFeature.clone()
# targetFeature = args.alpha * targetFeature + (1 - args.alpha) * contentFeature

# decode
# decoder = load_lua(args.decoder)
import decoder
decoder = decoder.decoder
decoder.load_state_dict(torch.load('decoder.pth'))

print "decoder.forward.."
outputImg = decoder.forward(targetFeature)
# print "decoder.forward {0}".format(outputImg.size())
outputImg = outputImg.squeeze() 
# print "outputImg.size {0}".format(outputImg.size())

outputImg = outputImg.clamp(min=0, max=1.0)

trans2img = transforms.Compose([
    transforms.ToPILImage(),
])
img = trans2img(outputImg.data)

# img.show()
img.save(args.outputImg)

print "finished!"