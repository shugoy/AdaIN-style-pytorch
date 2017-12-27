import torch
import torch.nn as nn
# import torch.legacy.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
# from BatchNormalization import BatchNormalization

# Implements adaptive instance normalization (AdaIN) as described in the paper:

# Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization
# Xun Huang, Serge Belongie

class AdaIN(nn.Module):

    def __init__(self, nOutput, disabled=False, eps=1e-5):
        # __init__(self, nOutput, disabled, eps)
        super(AdaIN, self).__init__()
        self.eps = eps or 1e-5
        self.nOutput = nOutput
        self.batchSize = -1
        self.disabled = disabled

    # function Module:forward(input)
    #     return self:updateOutput(input)
    # end

    def forward(self, content, style): # {content, style}
        # content = input[1]
        # style   = input[2]

        if self.disabled:
            self.output = content
            return self.output

        # N, Hc, Wc, Hs, Ws
        # ndim = content.nDimension
        # print content.dim()
        ndim = content.dim()
        if ndim == 3:
            assert(content.size(0) == self.nOutput)
            assert(style.size(0) == self.nOutput)
            N = 1
            Hc, Wc = content.size(1), content.size(2)
            Hs, Ws = style.size(1), style.size(2)
            content = content.view(1, self.nOutput, Hc, Wc)
            style = style.view(1, self.nOutput, Hs, Ws)
        elif ndim == 4:
            assert(content.size(0) == style.size(0))
            assert(content.size(1) == self.nOutput)
            assert(style.size(1) == self.nOutput)
            N = content.size(0)
            Hc, Wc = content.size(2), content.size(3)
            Hs, Ws = style.size(2), style.size(3)

        # -- compute target mean and standard deviation from the style input
        styleView = style.view((N, self.nOutput, -1))
        targetStd = styleView.std(2).view(-1)
        targetMean = styleView.mean(2).view(-1)
        
        # bn = BatchNormalization(N * self.nOutput)
        bn = nn.BatchNorm1d(N * self.nOutput)
        # bn = nn.InstanceNorm1d(N * self.nOutput)
        bn.batchSize = N
        bn.weight = Parameter(targetStd.data)
        bn.bias   = Parameter(targetMean.data)
        bn.training = True
        print(bn.weight, bn.bias)
        contentView = content.view(1, N * self.nOutput, -1)
        out = bn.forward(contentView)
        return out.view(1, 512, Hc, Wc)

        
        # import torch.nn.modules as modules
        # modules.BatchNorm3d

    # def updateGradInput(self, input, gradOutput):
    #     # -- Not implemented
    #     self.gradInput = nil
    #     return self.gradInput
    
    # def clearState(self):
    #     self.output = self.output.new()
    #     self.gradInput[1] = self.gradInput[1].new()
    #     self.gradInput[2] = self.gradInput[2].new()
    #     if bn.BatchNormalization():
    #         bn.clearState
    