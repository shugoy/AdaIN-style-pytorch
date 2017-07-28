import torch
import torch.nn as nn


# Implements adaptive instance normalization (AdaIN) as described in the paper:

# Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization
# Xun Huang, Serge Belongie

class AdaIN(nn.Module):

    def __init__(self, nOutput, disabled, eps):
        # __init__(self, nOutput, disabled, eps)
        super(AdaIN, self).__init__()

        self.eps = eps or 1e-5

        self.nOutput = nOutput
        self.batchSize = -1
        self.disabled = disabled

    # function Module:forward(input)
    #     return self:updateOutput(input)
    # end

    def updateOutput(self, content, style): # {content, style}
        # content = input[1]
        # style   = input[2]

        if self.disabled:
            self.output = content
            return self.output

        # N, Hc, Wc, Hs, Ws
        # if content.nDimension == 3:
            # assert(content.size(1) == self.nOutput)
            # assert(style.size(1) == self.nOutput)
        # N = 1
        # content.size
        # Hc, Wc = content.size(2), content.size(3)
        # Hs, Ws = style.size(2), style.size(3)
        # content = content.view(1, self.nOutput, Hc, Wc)
        # style = style.view(1, self.nOutput, Hs, Ws)
        # elif content.nDimension == 4:
            # assert(content:size(1) == style:size(1))
            # assert(content:size(2) == self.nOutput)
            # assert(style:size(2) == self.nOutput)
        N = content.size(1)
        Hc, Wc = content.size(3), content.size(4)
        Hs, Ws = style.size(3), style.size(4)

        # -- compute target mean and standard deviation from the style input
        styleView = style.view(N, self.nOutput, Hs*Ws)
        targetStd = styleView.std(3, True).view(-1)
        targetMean = styleView.mean(3).view(-1)

        # -- construct the internal BN layer
        if N != self.batchSize or (self.bn and self.type() != self.bn.type()):
            self.bn = nn.SpatialBatchNormalization(N * self.nOutput, self.eps)
            self.bn.type(self.type())
            self.batchSize = N

        # -- set affine params for the internal BN layer
        self.bn.weight.copy(targetStd)
        self.bn.bias.copy(targetMean)

        contentView = content.view(1, N * self.nOutput, Hc, Wc)
        self.bn.training()
        self.output = self.bn.forward(contentView).viewAs(content)
        return self.output

    # def updateGradInput(self, input, gradOutput):
    #     # -- Not implemented
    #     self.gradInput = nil
    #     return self.gradInput
    
    def clearState(self):
        self.output = self.output.new()
        self.gradInput[1] = self.gradInput[1].new()
        self.gradInput[2] = self.gradInput[2].new()
        if self.bn:
            self.bn.clearState
    