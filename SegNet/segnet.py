# coding=utf-8

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cPickle, gzip, os, sys, PIL
import numpy as np
from deep8 import *

class Conv2d:
    def __init__(self,
                 executor,
                 inputChannel,
                 outputChannel,
                 filterHeight=1,
                 filterWidth=1,
                 covered = True,
                 strideY = 1,
                 strideX = 1,
                 dilationY = 1,
                 dilationX = 1):
        self.weight = parameter(executor, [outputChannel, filterHeight, filterWidth, inputChannel]).gaussian(0.0, 0.01)

        self.covered   = covered
        self.strideY   = strideY
        self.strideX   = strideX
        self.dilationY = dilationY
        self.dilationX = dilationX

    def forward(self, input):
        return input.conv2d(self.weight, self.covered, self.strideY, self.strideX, self.dilationY, self.dilationX)

class BatchNorm:
    isTraining = True

    def __init__(self, executor, outputChannel, epsilon = 1e-7, momentum = 0.9):
        self.epsilon  = epsilon
        self.momentum = momentum

        self.gamma = parameter(executor, [outputChannel]).gaussian(0.0, 0.01)
        self.beta  = parameter(executor, [outputChannel]).gaussian(0.0, 0.01)

        self.running_mean     = parameter(executor, [outputChannel], updateGradient = False).zero()
        self.running_variance = parameter(executor, [outputChannel], updateGradient = False).zero()

    def forward(self, input):
        '''trainning phrase'''
        if self.isTraining:
            mean = input.reduceMean([0, 1, 2], keepDims = False)
            variance  = (input - mean).square().reduceMean([0, 1, 2], keepDims = False)

            '''update running_mean and running_variance'''
            self.running_mean.assign(self.running_mean * self.momentum + mean * (1 - self.momentum))
            self.running_variance.assign(self.running_variance * self.momentum + variance * (1 - self.momentum))

            norm = (input - mean) / (variance + self.epsilon).sqrt()

            return self.gamma.multiply(norm) + self.beta
        else:
            norm = (input - self.running_mean) / (self.running_variance + self.epsilon).sqrt()

            return self.gamma.multiply(norm) + self.beta

def readData():
    sampleNames = []

    '''only read size is 320X240'''
    with open('iccv09Data/horizons.txt', 'r') as fp:
        for line in fp.readlines():
            strs = line.split(' ')

            if 320 == int(strs[1]) and 240 == int(strs[2]):
                sampleNames.append(strs[0])

    x = np.zeros((len(sampleNames), 240, 320, 3), dtype=np.float32)
    y = np.fromfile('iccv09Data/labels.bin', dtype=np.float32)

    y = np.reshape(y, (len(sampleNames), 240, 320, 9))

    '''read image'''
    for i in range(len(sampleNames)):
        print('read image ==> ', i)

        image = PIL.Image.open('iccv09Data/images/' + sampleNames[i] + '.jpg')

        '''convert to [-1.0, 1.0]'''
        x[i] = (np.array(image).astype(np.float32) / 255 - 0.5) * 2.0

    # '''
    #  (sky, tree, road, grass, water, building, mountain, or foreground object). A negative number indicates unknown
    #  [0-7] means object, 8 means nothing
    # '''
    # for i in range(len(sampleNames)):
    #     print('read region ==> ', i)
    #
    #     with open('iccv09Data/labels/' + sampleNames[i] + '.regions.txt', 'r') as fp:
    #         allLines = fp.readlines()
    #
    #         for k in range(240):
    #             labels = allLines[k].split(' ')
    #
    #             for l in range(320):
    #                 type = int(labels[l])
    #
    #                 if type >= 0:
    #                     y[i][k][l][type] = 1.0
    #                 else:
    #                     y[i][k][l][8] = 1.0

    return x, y

def SegNet():
    epoch = 10

    trainX, trainY = readData()

    imageHeight = 240
    imageWidth  = 320
    imageChannel = 3

    typeCount = 9

    batchSize = 8

    executor     = EagerExecutor(DeviceType.CPU)
    learningRate = LinearDecayLearningRateIterator(totalStep=epoch * len(trainX), start=1e-3, end=0.0)
    trainer      = AdamTrainer(learningRate=learningRate)

    conv1 = Conv2d(executor, 3, 64, 3, 3)
    bn1   = BatchNorm(executor, 64)

    conv2 = Conv2d(executor, 64, 128, 3, 3)
    bn2   = BatchNorm(executor, 128)

    conv3 = Conv2d(executor, 128, 256, 3, 3)
    bn3   = BatchNorm(executor, 256)

    conv4 = Conv2d(executor, 256, 512, 3, 3)
    bn4   = BatchNorm(executor, 512)

    conv4d = Conv2d(executor, 512, 512, 3, 3)
    bn4d   = BatchNorm(executor, 512)

    conv3d = Conv2d(executor, 512, 256, 3, 3)
    bn3d   = BatchNorm(executor, 256)

    conv2d = Conv2d(executor, 256, 128, 3, 3)
    bn2d   = BatchNorm(executor, 128)

    conv1d = Conv2d(executor, 128, 64, 3, 3)
    bn1d   = BatchNorm(executor, 64)

    convLast = Conv2d(executor, 64, typeCount, 1, 1)

    for e in range(epoch):
        for i in range(0, len(trainX) - batchSize + 1, batchSize):
            x = inputParameter(executor, [imageHeight, imageWidth, imageChannel]).feed(trainX[i])
            y = inputParameter(executor, [imageHeight, imageWidth, typeCount]).feed(trainY[i])

            layer1 = bn1.forward(conv1.forward(x))
            pool1  = layer1.maxPooling2d(False, 2, 2, 2, 2)
            index1 = layer1.maxIndex2d(False, 2, 2, 2, 2)

            layer2 = bn2.forward(conv2.forward(pool1))
            pool2  = layer2.maxPooling2d(False, 2, 2, 2, 2)
            index2 = layer2.maxIndex2d(False, 2, 2, 2, 2)

            layer3 = bn3.forward(conv3.forward(pool2))
            pool3  = layer3.maxPooling2d(False, 2, 2, 2, 2)
            index3 = layer3.maxIndex2d(False, 2, 2, 2, 2)

            layer4 = bn4.forward(conv4.forward(pool3))
            pool4  = layer4.maxPooling2d(False, 2, 2, 2, 2)
            index4 = layer4.maxIndex2d(False, 2, 2, 2, 2)

            layer4d = bn4d.forward(conv4d.forward(pool4))
            pool4d  = layer4d.maxUnPooling2d(index4, False, 2, 2, 2, 2)

            layer3d = bn3d.forward(conv3d.forward(pool4d))
            pool3d  = layer3d.maxUnPooling2d(index3, False, 2, 2, 2, 2)

            layer2d = bn2d.forward(conv2d.forward(pool3d))
            pool2d  = layer2d.maxUnPooling2d(index2, False, 2, 2, 2, 2)

            layer1d = bn1d.forward(conv1d.forward(pool2d))
            pool1d  = layer1d.maxUnPooling2d(index1, False, 2, 2, 2, 2)

            pred = convLast.forward(pool1d)

            loss = pred.softmaxCrossEntropyLoss(y)

            print 'epoch:', e, ', step:', i, 'loss ==> ', loss.valueStr()

            backward(loss)
            trainer.train(executor)

if __name__ == '__main__':
    SegNet()


























