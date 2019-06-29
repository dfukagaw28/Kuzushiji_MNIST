import os

from fastai.vision import Learner
from fastai.vision import accuracy
from fastai.vision import math
from fastai.vision import nn
from fastai.vision import torch
import numpy as np

import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

from utils import get_databunch
from make_vgg import VGG
from make_resnet import MyResNet


#----------------------------------------------------------------
# ## Ensemble of VGG and ResNet-18

class VGG_ResNet(nn.Module):
    def __init__(self):
        super(VGG_ResNet, self).__init__()
        self.vgg = VGG()
        self.resnet = MyResNet(BasicBlock, [2, 2, 2, 2])
    
    def forward(self, x):
        vgg_out = self.vgg(x)
        resnet_out = self.resnet(x)
        out = (vgg_out + resnet_out) / 2
        return out
    
def vgg_resnet_load_model(learner, vgg_name, resnet_name):
        device = learner.data.device
        vgg_state = torch.load(learner.path/learner.model_dir/f'{vgg_name}.pth', map_location=device)
        learner.model.vgg.load_state_dict(vgg_state['model'], strict=True)
        
        resnet_state = torch.load(learner.path/learner.model_dir/f'{resnet_name}.pth', map_location=device)
        learner.model.resnet.load_state_dict(resnet_state['model'], strict=True)


if __name__ == "__main__":
    datasetdir = os.path.join(os.path.dirname(__file__), './kuzu_mnist')
    datasetdir = os.path.abspath(datasetdir)

    # Load dataset
    databunch = get_databunch(datasetdir)
    print('Dataset loaded')

    # Create VGG + ResNet model
    learn = Learner(databunch, VGG_ResNet(), metrics=accuracy)

    vgg_name = 'vgg_model_with_norm'
    resnet_name = 'resnet_model_with_norm'
    vgg_resnet_load_model(learn, vgg_name, resnet_name)

    # Train
    learn.fit(1)

    # Save
    learn.save('vgg_resnet_model_with_norm')
