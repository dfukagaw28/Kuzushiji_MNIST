from fastai.vision import Learner
from fastai.vision import accuracy
from fastai.vision import torch

import numpy as np

import os

from utils import get_databunch
from make_vgg_resnet import VGG_ResNet


if __name__ == "__main__":
    datasetdir = os.path.join(os.path.dirname(__file__), './kuzu_mnist')
    datasetdir = os.path.abspath(datasetdir)

    # Load dataset
    databunch = get_databunch(datasetdir)

    # Create VGG + ResNet model
    learn = Learner(databunch, VGG_ResNet(), metrics=accuracy)

    # Load
    learn.load('vgg_resnet_model_with_norm')

    # Validate
    loss, acc = learn.validate()
    print('val_loss: {}, val_acc: {}'.format(loss, acc))

    mat = np.zeros((10, 10))
    for data in databunch.valid_ds:
        images, labels = data
        images = images.reshape((1, 1, 28, 28))
        outputs = learn.model(images)
        _, predicted = torch.max(outputs, 1)
        predicted = int(predicted)
        mat[labels][predicted] += 1
    
    print(mat)

