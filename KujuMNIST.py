#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchvision.models.resnet import resnet18, ResNet, BasicBlock
from torchvision.datasets.mnist import MNIST
from fastai.vision import * 
from fastai import *
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable

from torchvision.datasets.utils import makedir_exist_ok, download_url
from torch.utils.data import BatchSampler, DataLoader
from torchvision import transforms
from KujuMNIST_dataset import KujuMNIST_DS


# Calculate mean and std of the dataset for normalization

# In[2]:


trn_data = np.load('./kuzu_mnist/kmnist-train-imgs.npz')
trn_data = trn_data['arr_0'] / 255
data_mean = trn_data.mean()
data_std = trn_data.std()
print(f'Mean: {data_mean}')
print(f'Std: {data_std}')


# ## Prepare Datasets, DataLoaders and DataBunch
# 
# Optional: A random transformations on the images in training

# In[31]:


default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

transform_train = transforms.Compose(
    [transforms.ToPILImage(), 
     #transforms.RandomAffine(degrees=7, translate=(0.1, 0.1), scale=(0.95, 1.05)), 
     transforms.ToTensor(),
     transforms.Normalize((data_mean,), (data_std,)),
    ])

transform_valid = transforms.Compose(
    [transforms.ToPILImage(), 
     transforms.ToTensor(),
     transforms.Normalize((data_mean,), (data_std,)),
    ])

ROOT_FOLDER = './kuzu_mnist/'

trn_ds = KujuMNIST_DS(ROOT_FOLDER, train_or_test='train', download=False, tfms=transform_train)
val_ds = KujuMNIST_DS(ROOT_FOLDER, train_or_test='test', download=False, tfms=transform_valid)

trn_dl = DataLoader(trn_ds, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
databunch = DataBunch(path=ROOT_FOLDER, train_dl=trn_dl, valid_dl=val_dl, device=default_device)


# MNIST Dataset

# In[32]:


# dataset_transform = transforms.Compose([
#                transforms.ToTensor(),
#                transforms.Normalize((0.1307,), (0.3081,))
#            ])

# mnist_trn_ds = MNIST('./', train=True, download=False, transform=dataset_transform)
# mnist_val_ds = MNIST('./', train=False, download=False, transform=dataset_transform)


# ## VGG Model
# 
# Based on - https://github.com/kkweon/mnist-competition

# In[33]:


class VGG(nn.Module):  
    """
    Based on - https://github.com/kkweon/mnist-competition
    """
    def two_conv_pool(self, in_channels, f1, f2):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s
    
    def three_conv_pool(self,in_channels, f1, f2, f3):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s
        
    
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.l1 = self.two_conv_pool(1, 64, 64)
        self.l2 = self.two_conv_pool(64, 128, 128)
        self.l3 = self.three_conv_pool(128, 256, 256, 256)
        self.l4 = self.three_conv_pool(256, 256, 256, 256)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1) 
              


# In[45]:


learn = Learner(databunch, VGG(), metrics=accuracy)
learn.load('vgg_model_with_norm')
print('Model was loaded')


# In[46]:


learn.fit(1)


# ## ResNet Model
# 
# Based on - https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

# In[34]:


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    
class MyResNet(nn.Module):
    # Based on PyTorch ResNet-18
    
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False):
        super(MyResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                    
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(512 * block.expansion, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(256, num_classes),
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
#         import pdb; pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


# In[42]:


learn = Learner(databunch, MyResNet(BasicBlock, [2, 2, 2, 2]), metrics=accuracy)
learn.load('resnet_model_with_norm')
print('Model was loaded')


# In[43]:


learn.fit(1)


# ## Ensemble of VGG and ResNet-18

# In[7]:


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


# In[8]:


learn = Learner(databunch, VGG_ResNet(), metrics=accuracy)
# vgg_resnet_load_model(learn, vgg_name, resnet_name)
learn.load('vgg_resnet_model_with_norm')
print('Model was loaded')


# In[9]:


learn.fit(1)


# ## Capsule Network
# 
# Taken from - https://github.com/higgsfield/Capsule-Network-Tutorial

# In[35]:


class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=1
                             )

    def forward(self, x):
        return F.relu(self.conv(x))
    
class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9):
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0) 
                          for _ in range(num_capsules)])
    
    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), 32 * 6 * 6, -1)
        return self.squash(u)
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor
    
class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        
        b_ij = b_ij.to(default_device)

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            
            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )
        
    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes)
        
        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.eye(10))

        masked = masked.to(default_device)
        masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)
        
        reconstructions = self.reconstraction_layers((x * masked[:, :, None, None]).view(x.size(0), -1))
        reconstructions = reconstructions.view(-1, 1, 28, 28)
        
        return reconstructions, masked

    
def caps_loss(inputs, targets):
    targets = torch.eye(10).index_select(dim=0, index=targets.cpu()).to(default_device)
    data, output, reconstructions, masked = inputs
    return margin_loss(output, targets) + reconstruction_loss(data, reconstructions)
    
def margin_loss(x, labels, size_average=True):
    batch_size = x.size(0)

    v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))
    left = F.relu(0.9 - v_c).view(batch_size, -1)
    right = F.relu(v_c - 0.1).view(batch_size, -1)
    
    loss = labels * left + 0.5 * (1.0 - labels) * right
    loss = loss.sum(dim=1).mean()

    return loss

def reconstruction_loss(data, reconstructions):
    loss = F.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
    return loss * 0.0005

def caps_accuracy(inputs, targs):
    masked = inputs[-1]
    predictions = np.argmax(masked.data.cpu().numpy(), 1)
    return torch.tensor((predictions == targs.cpu().numpy()).mean())


class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()
        self.decoder = Decoder()
        
    def forward(self, data):
        output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
        reconstructions, masked = self.decoder(output, data)
        return data, output, reconstructions, masked
    


# In[36]:


learn = Learner(databunch, CapsNet(), metrics=caps_accuracy, loss_func=caps_loss)
# vgg_resnet_load_model(learn, vgg_name, resnet_name)
learn.load('caps_net_model_with_norm')
print('Model was loaded')


# In[37]:


learn.fit(1)


# ## Ensemble of VGG and Capsule Network
# 
# Results are worse than VGG-ResNet Ensemble

# In[38]:


class VGG_Caps(nn.Module):
    def __init__(self):
        super(VGG_Caps, self).__init__()
        self.vgg = VGG()
        caps_model = CapsNet()
        self.capsnet = caps_model
    
    def forward(self, x):
        vgg_out = self.vgg(x)
        capsnet_out = self.capsnet(x)
        return vgg_out, capsnet_out

    
def vgg_capsnet_load_model(learner, vgg_name, caps_name):
        device = learner.data.device
        vgg_state = torch.load(learner.path/learner.model_dir/f'{vgg_name}.pth', map_location=device)
        learner.model.vgg.load_state_dict(vgg_state['model'], strict=True)
        
        capsnet_state = torch.load(learner.path/learner.model_dir/f'{caps_name}.pth', map_location=device)
        learner.model.capsnet.load_state_dict(capsnet_state['model'], strict=True)

def vgg_caps_accuracy(outputs, targs):
    caps_outputs = outputs[1][-1]
    vgg_outputs = outputs[0]
    batch_size = targs.size(0)    
    
    caps_outputs = F.softmax(caps_outputs, dim=1)
    
    final_preds = (caps_outputs + vgg_outputs) / 2
    final_preds = final_preds.argmax(dim=-1).view(batch_size,-1)

    targs = targs.view(batch_size,-1)
    return (final_preds==targs).float().mean()
    
def vgg_caps_loss(inputs, targets):
    vgg_loss = torch.functional.F.nll_loss(inputs[0], targets)
    return caps_loss(inputs[1], targets) + vgg_loss


# ## Training

# In[39]:


learn = Learner(databunch, VGG_Caps(), metrics=vgg_caps_accuracy, loss_func=vgg_caps_loss)
vgg_capsnet_load_model(learn, 'vgg_model_with_norm', 'caps_net_model_with_norm')


# In[40]:


learn.fit(1)
# learn.save('vgg_resnet_model_with_norm')


# In[ ]:




