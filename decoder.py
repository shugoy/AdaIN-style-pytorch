
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self):
		super(Decoder, self).__init__()
		self.model = nn.Sequential( # Sequential,
			nn.ReflectionPad2d((1, 1, 1, 1)),
			nn.Conv2d(512,256,(3, 3)),
			nn.ReLU(),
			nn.UpsamplingNearest2d(scale_factor=2),
			nn.ReflectionPad2d((1, 1, 1, 1)),
			nn.Conv2d(256,256,(3, 3)),
			nn.ReLU(),
			nn.ReflectionPad2d((1, 1, 1, 1)),
			nn.Conv2d(256,256,(3, 3)),
			nn.ReLU(),
			nn.ReflectionPad2d((1, 1, 1, 1)),
			nn.Conv2d(256,256,(3, 3)),
			nn.ReLU(),
			nn.ReflectionPad2d((1, 1, 1, 1)),
			nn.Conv2d(256,128,(3, 3)),
			nn.ReLU(),
			nn.UpsamplingNearest2d(scale_factor=2),
			nn.ReflectionPad2d((1, 1, 1, 1)),
			nn.Conv2d(128,128,(3, 3)),
			nn.ReLU(),
			nn.ReflectionPad2d((1, 1, 1, 1)),
			nn.Conv2d(128,64,(3, 3)),
			nn.ReLU(),
			nn.UpsamplingNearest2d(scale_factor=2),
			nn.ReflectionPad2d((1, 1, 1, 1)),
			nn.Conv2d(64,64,(3, 3)),
			nn.ReLU(),
			nn.ReflectionPad2d((1, 1, 1, 1)),
			nn.Conv2d(64,3,(3, 3)),
		)

    def forward(self, x):
		return self.model(x)