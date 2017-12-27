import torch
import torch.nn as nn
import AdaIN
from decoder import Decoder
import os

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--contentDir', type=str, default='data/coco/train2014', help='Directory containing content images for training')
parser.add_argument('--styleDir', type=str, default='data/wikiart/train', help='Directory containing style images for training')
parser.add_argument('--name', type=str, default='adain', help='Name of the checkpoint directory')
parser.add_argument('--gpu', type=int, default=0, help='Zero-indexed ID of the GPU to use')
parser.add_argument('--nThreads', type=int, default=2, help='Number of data loading threads')
parser.add_argument('--activation', type=str, default='relu', help='Activation function in the decoder')

# Preprocessing options
parser.add_argument('--finalSize', type=int, default=256, help='Size of images used for training')
parser.add_argument('--contentSize', type=int, default=512, help='Size of content images before cropping, keep original size if set to 0')
parser.add_argument('--styleSize', type=int, 512, help='Size of style images before cropping, keep original size if set to 0')
parser.add_argument('--crop', True, help='If True, crop training images')

# Training options
parser.add_argument('--resume', False, help='If True, resume training from the last checkpoint')
parser.add_argument('--optimizer', 'adam', help='Optimizer used, adam|sgd')
parser.add_argument('--learningRate', 1e-4, help='Learning rate')
parser.add_argument('--learningRateDecay', 5e-5, help='Learning rate decay')
parser.add_argument('--momentum', 0.9, help='Momentum')
parser.add_argument('--weightDecay', 0, help='Weight decay')
parser.add_argument('--batchSize', 8, 'Batch size')
parser.add_argument('--maxIter', 160000, 'Maximum number of iterations')
parser.add_argument('--targetContentLayer', 'relu4_1', 'Target content layer used to compute the loss')
parser.add_argument('--targetStyleLayers', 'relu1_1,relu2_1,relu3_1,relu4_1', 'Target style layers used to compute the loss')
parser.add_argument('--tvWeight', 0, 'Weight of TV loss')
parser.add_argument('--styleWeight', 1e-2, 'Weight of style loss')
parser.add_argument('--contentWeight', 1, 'Weight of content loss')
parser.add_argument('--reconStyle', False, 'If True, the decoder is also trained to reconstruct style images')
parser.add_argument('--normalize', False, 'If True, gradients at the loss function are normalized')

# Verbosity
parser.add_argument('--printDetails', True, 'If True, print style loss at individual layers')
parser.add_argument('--display', True, 'If True, display the training progress')
parser.add_argument('--displayAddr', '0.0.0.0', 'Display address')
parser.add_argument('--displayPort', 8000, 'Display port')
parser.add_argument('--displayEvery', 100, 'Display interval')
parser.add_argument('--saveEvery', 2000, 'Save interval')
parser.add_argument('--printEvery', 10, 'Print interval')
opt = parser.parse_args()
torch.setDevice(opt.gpu+1)
print(opt)

## Prepare ##
if opt.contentSize == 0:
    opt.contentSize = nil

if opt.styleSize == 0 :
    opt.styleSize = nil


assert(os.path.dirp(opt.contentDir),
    '-contentDir does not exist.')
assert(os.path.dirp(opt.styleDir),
    '-styleDir does not exist.')

if not opt.resume :
    os.path.mkdir(opt.save)
    torch.save(os.path.join(opt.save, 'options.pth'), opt)


if opt.activation == 'relu' :
    decoderActivation = nn.ReLU
elif opt.activation == 'prelu' :
    decoderActivation = nn.PReLU
elif opt.activation == 'elu' :
    decoderActivation = nn.ELU
# else:
    # error('Unknown activation option ' .. opt.activation)


## Load VGG ##
# if not opt.use_vgg_normalized:
import vgg
enc = vgg.Vgg16(requires_grad=False)
# else:
#     vgg = torch.load('models/vgg_normalised.pth')
#     enc = nn.Sequential()
#     for i in range(1, len(vgg)):
#         layer = vgg.get(i)
#         enc.append(layer)
#         name = layer.name
#         if name == opt.targetContentLayer :
#             break
    


# ## Build AdaIN layer ##
adain = AdaIN.AdaIN()

## Build decoder ##
dec = Decoder()
if opt.resume :
    loc = os.path.join(opt.save, string.format('decoder_latest.pth', startIter))
    print("Resume training from: ", loc)
    dec.model.load_state_dict(torch.load(loc))


## Build encoder ##
layers = {}
layers.content = {opt.targetContentLayer}
layers.style = opt.targetStyleLayers:split(',')
weights = {}
weights.content = opt.contentWeight
weights.style  = opt.styleWeight
weights.tv = opt.tvWeight
criterion = nn.ArtisticStyleLossCriterion(enc, layers, weights, opt.normalize)

## Move to GPU ##
criterion.net = cudnn.convert(criterion.net, cudnn):cuda()
adain = adain:cuda()
dec = cudnn.convert(dec, cudnn):cuda()

print("encoder:")
print(criterion.net)
print("decoder:")
print(dec)

## Build data loader ##
contentLoader = ImageLoaderAsync(opt.contentDir, opt.batchSize, {len=opt.contentSize, H=opt.finalSize, W=opt.finalSize, n=opt.nThreads}, opt.crop)
styleLoader = ImageLoaderAsync(opt.styleDir, opt.batchSize, {len=opt.styleSize, H=opt.finalSize, W=opt.finalSize, n=opt.nThreads}, opt.crop)
print("Number of content images: " .. contentLoader:size())
print("Number of style images: " .. styleLoader:size())

## Training ##-
if opt.resume :
    optimState = torch.load(os.path.join(opt.save, 'optimState.pth'))
else:
    optimState = {
        learningRate = opt.learningRate,
        learningRateDecay = opt.learningRateDecay,
        weightDecay = opt.weightDecay,
        beta1 = opt.momentum,
        momentum = opt.momentum
    }


def maybe_print(trainLoss, contentLoss, styleLoss, tvLoss, timer)
    if optimState.iterCounter % opt.printEvery == 0 :
        print(string.format('%7d\t\t%e\t%e\t%e\t%e\t%.2f\t%e',
        optimState.iterCounter, trainLoss, contentLoss, styleLoss, tvLoss,
            timer:time().real, optimState.learningRate / (1 + optimState.iterCounter*optimState.learningRateDecay)))
        allStyleLoss = {}
        for _, mod in ipairs(criterion.style_layers) do
            table.insert(allStyleLoss, mod.loss)
        
        if opt.printDetails :
            print(allStyleLoss)
        
        timer:reset()
    


def maybe_display(inputs, reconstructions, history)
    if opt.display and (optimState.iterCounter % opt.displayEvery == 0) :
        disp = torch.cat(reconstructions:float(), inputs:float(), 1)
        displayWindow = 1
        if displayWindow :
            styleNamesDisplayed = {}
            for i=1,#styleNames do
                stylename = styleNames[i]
                temp = stylename:split('/')
                tempname = temp[#temp]
                table.insert(styleNamesDisplayed, tempname)
            
            display.image(disp, {win=displayWindow, max=1, min=0, nperrow=opt.batchSize, labels=styleNamesDisplayed})
        else:
            displayWindow = display.image(disp, {max=1, min=0})
        
        display.plot(history, {win=displayWindow+1, title="loss: " .. opt.name, 
            labels = {"iteration", "loss", "content", "style", 'style_recon'}})
    


def maybe_save()
    if optimState.iterCounter % opt.saveEvery == 0 :
        loc = os.path.join(opt.save, string.format('dec-%06d.pth', optimState.iterCounter))
        decSaved = dec:clearState():clone()
        torch.save(loc, cudnn.convert(decSaved:float(), nn))
        torch.save(os.path.join(opt.save, 'history.pth'), history)
        torch.save(os.path.join(opt.save, 'optimState.pth'), optimState)
        dec:clearState()
        criterion.net:clearState()
        decSaved = nil
        collectgarbage()
    


def train()
    optimState.iterCounter = optimState.iterCounter or 0
    weights, gradients = dec.getParameters()
    print('Training...\tTrainErr\tContent\t\tStyle\t\tTVLoss\t\ttime\tLearningRate')
    timer = torch.Timer()
    while optimState.iterCounter < opt.maxIter do
        def feval(x)
            gradients:zero()
            optimState.iterCounter = optimState.iterCounter + 1
            contentInput = contentLoader:nextBatch()
            styleInput, styleNames = styleLoader:nextBatch()
            contentInput = contentInput:float():cuda()
            styleInput = styleInput:float():cuda()

            # Forward style image through the encoder
            criterion:setStyleTarget(styleInput)
            styleLatent = criterion.net.output:clone()

            # Forward content image through the encoder
            criterion:setContentTarget(contentInput)
            contentLatent = criterion.net.output:clone()
            
            # Perform AdaIN
            outputLatent = adain:forward({contentLatent, styleLatent})

            # Set content target
            criterion.content_layers[1]:setTarget(outputLatent)

            # Compute loss
            output = dec:forward(outputLatent):clone() # forward through decoder, generate transformed images
            loss = criterion:forward(output) # forward through loss network, compute loss functions
            contentLoss = criterion.contentLoss
            styleLoss = criterion.styleLoss
            tvLoss = 0
            if opt.tvWeight > 0 .
                tvLoss = criterion.net.get(2).loss
            

            # Backpropagate gradients
            decGrad = criterion.backward(output) # backprop through loss network, compute gradients w.r.t. the transformed images
            dec.backward(outputLatent, decGrad) # backprop gradients through decoder

            # Optionally train the decoder to reconstruct style images
            styleReconLoss = 0
            if opt.reconStyle :
                criterion.setContentTarget(styleInput)
                styleRecon = dec.forward(styleLatent).clone()
                styleReconLoss = criterion.forward(styleRecon)
                decGrad = criterion.backward(styleRecon)
                dec.backward(styleLatent, decGrad)
                loss =  loss + styleReconLoss
            
            
            table.insert(history, {optimState.iterCounter, loss, contentLoss, styleLoss, styleReconLoss})
            maybe_print(loss, contentLoss, styleLoss, tvLoss, timer)
            if opt.reconStyle :
                displayImages = torch.cat({output, styleRecon}, 1)
            else.
                displayImages = output
            
            criterion.net.clearState()
            maybe_display(torch.cat({contentInput, styleInput}, 1), displayImages, history)
            maybe_save()
            return loss, gradients
        

        if opt.optimizer == 'adam' :
            optim.adam(feval, weights, optimState)
        elif opt.optimizer == 'sgd' :
            optim.sgd(feval, weights, optimState)
        else:
            error("Not supported optimizer: " .. opt.optimizer)
        
    


train()
