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
from AdaIN import AdaIN

# def main():
parser = argparse.ArgumentParser()
parser.add_argument("--contentImg", default="images/brad_pitt.jpg")
parser.add_argument("--styleImg", default="images/picasso_self_portrait.jpg")
parser.add_argument("--outputImg", default=None)
parser.add_argument("--vgg", default="models/vgg_normalised.pth")
parser.add_argument("--decoder", default="models/decoder.pth")
parser.add_argument("--cuda", default=False)
parser.add_argument("--alpha", default=1)#, 'The weight that controls the degree of stylization. Should be between 0 and 1')
args = parser.parse_args()

# import vgg_normalised
# vgg = vgg_normalised.vgg_normalised
# vgg.load_state_dict(torch.load(args.vgg))
# vgg = nn.Sequential(*(vgg[i] for i in range(31)))
# vgg.cuda.
import vgg
vgg = vgg.Vgg16(requires_grad=False)

# decode
from decoder import Decoder
decoder = Decoder()
decoder.model.load_state_dict(torch.load(args.decoder))

## load images
contentImg = Image.open(args.contentImg)
styleImg = Image.open(args.styleImg)

content_transform = transforms.Compose([
    transforms.Scale(512),
    transforms.ToTensor(),
])
content = content_transform(contentImg)
style   = content_transform(styleImg)

content = Variable(content)
style   = Variable(style)

content = content.unsqueeze(0)
style   = style.unsqueeze(0)

if args.cuda:
    content = content.cuda()
    style   = style.cuda()

print "vgg.forward.."
styleFeature   = vgg.forward(style).relu4_3
contentFeature = vgg.forward(content).relu4_3
print (styleFeature.size())
print (contentFeature.size())

# AdaIN
print "adaIN.forward.."
adain = AdaIN(contentFeature.size(1), False, 1e-5)
targetFeature = adain.forward(contentFeature, styleFeature)
alpha = float(args.alpha)
targetFeature = alpha * targetFeature + (1 - alpha) * contentFeature

print "decoder.forward.."
outputImg = decoder.forward(targetFeature)
outputImg = outputImg.squeeze() 

outputImg = outputImg.clamp(min=0, max=1.0)
trans2img = transforms.Compose([transforms.ToPILImage()])
img = trans2img(outputImg.data)

if args.outputImg is None:
    path_c, file_c = os.path.split(args.contentImg)
    name_c, ext_c  = os.path.splitext(file_c)
    path_s, file_s = os.path.split(args.styleImg)
    name_s, ext_s  = os.path.splitext(file_s)
    outname = "out/"+name_c+"_"+name_s
    if alpha < 1.0:
        outname += "_"+str(alpha)
    outpath = outname + ext_c
else:
    outpath = args.outputImg

img.save(outpath)
print "saved "+outpath