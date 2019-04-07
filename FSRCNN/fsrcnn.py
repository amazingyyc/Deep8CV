# coding=utf-8

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cPickle, gzip, os, sys, PIL
import numpy as np
from deep8 import *

'''read traning image from file'''
def read_train_images(folder, scale):
    image_path = [os.path.join(folder, i) for i in os.listdir(folder)]

    x = []
    y = []

    for i in image_path:
        yimage = PIL.Image.open(i)

        xheight, xwidth = (yimage.height - 1) / scale + 1, (yimage.width - 1) / scale + 1

        ximage = yimage.resize((xwidth, xheight))

        xarray = np.array(ximage)
        yarray = np.array(yimage)

        x.append((xarray.astype(np.float32) / 255 - 0.5) * 2.0)
        y.append((yarray.astype(np.float32) / 255 - 0.5) * 2.0)

    return x, y

def read_test_images(folder):
    image_path = [os.path.join(folder, i) for i in os.listdir(folder)]

    test_images = []

    for i in image_path:
        image = PIL.Image.open(i)

        test_images.append((np.array(image).astype(np.float32) / 255 - 0.5) * 2.0)

    return test_images

def FSRCNN(trainingPath, testPath, outputPath):
    epoch = 1
    scale = 2

    trainX, trainY = read_train_images(trainingPath, scale)

    executor     = EagerExecutor()
    learningRate = LinearDecayLearningRateIterator(totalStep=epoch * len(trainX), start=1e-3, end=0.0)
    trainer      = AdamTrainer(learningRate=learningRate)

    w1 = parameter(executor, [32, 5, 5, 3]).gaussian(0.0, 0.0378)
    b1 = parameter(executor, [32]).constant(0.0)

    w2 = parameter(executor, [5, 1, 1, 32]).gaussian(0.0, 0.3536)
    b2 = parameter(executor, [5]).constant(0.0)

    map3 = parameter(executor, [5, 3, 3, 5]).gaussian(0.0, 0.1179)
    b3   = parameter(executor, [5]).constant(0.0)

    map4 = parameter(executor, [5, 3, 3, 5]).gaussian(0.0, 0.1179)
    b4   = parameter(executor, [5]).constant(0.0)

    map5 = parameter(executor, [5, 3, 3, 5]).gaussian(0.0, 0.1179)
    b5   = parameter(executor, [5]).constant(0.0)

    w6 = parameter(executor, [32, 1, 1, 5]).gaussian(0.0, 0.189)
    b6 = parameter(executor, [32]).constant(0.0)

    w7 = parameter(executor, [3, 9, 9, 32]).gaussian(0.0, 0.0001)
    b7 = parameter(executor, [3]).constant(0.0)

    for e in range(epoch):
        for i in range(len(trainX) / 2):
            x = inputParameter(executor, trainX[i].shape).feed(trainX[i])
            y = inputParameter(executor, trainY[i].shape).feed(trainY[i])

            layer1 = (x.conv2d(w1, True, 1, 1) + b1).lRelu(0.01)
            layer2 = (layer1.conv2d(w2, True, 1, 1) + b2).lRelu(0.01)
            layer3 = (layer2.conv2d(map3, True, 1, 1) + b3).lRelu(0.01)
            layer4 = (layer3.conv2d(map4, True, 1, 1) + b4).lRelu(0.01)
            layer5 = (layer4.conv2d(map5, True, 1, 1) + b5).lRelu(0.01)
            layer6 = (layer5.conv2d(w6, True, 1, 1) + b6).lRelu(0.01)
            layer7 = layer6.deConv2d(w7, True, scale, scale) + b7

            loss = layer7.l1Loss(y)

            print "epoch:", e, ", step:", i, ", loss => ", loss.valueStr()

            backward(loss)

            trainer.train(executor)

    '''for test'''
    test_images = read_test_images(testPath)

    for i in range(len(test_images)):
        (height, width, channel) = test_images[i].shape

        yheight, ywidth = (height - 1) * scale + 1, (width - 1) * scale + 1

        x = inputParameter(executor, test_images[i].shape).feed(test_images[i])

        layer1 = (x.conv2d(w1, True, 1, 1) + b1).lRelu(0.01)
        layer2 = (layer1.conv2d(w2, True, 1, 1) + b2).lRelu(0.01)
        layer3 = (layer2.conv2d(map3, True, 1, 1) + b3).lRelu(0.01)
        layer4 = (layer3.conv2d(map4, True, 1, 1) + b4).lRelu(0.01)
        layer5 = (layer4.conv2d(map5, True, 1, 1) + b5).lRelu(0.01)
        layer6 = (layer5.conv2d(w6, True, 1, 1) + b6).lRelu(0.01)
        layer7 = layer6.deConv2d(w7, True, scale, scale) + b7

        yimage = np.zeros([yheight, ywidth, channel], dtype=np.float32)
        layer7.fetch(yimage)

        yimage = ((yimage + 1.0) * 127.5).astype(np.int32)
        np.clip(yimage, 0, 255)

        PIL.Image.fromarray(yimage.astype(np.uint8)).save(os.path.join(outputPath, '{0}.png'.format(i)))

        print 'saved a image'

if __name__ == '__main__':
    trainingPath = 'data/images/train'
    testPath     = 'data/images/test'
    outputPath   = 'data/images/output'

    FSRCNN(trainingPath, testPath, outputPath)