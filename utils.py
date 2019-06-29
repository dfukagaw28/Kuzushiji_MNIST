import numpy as np
import os


def get_databunch(datasetdir):
    from fastai.vision import DataBunch
    from fastai.vision import torch

    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    from KujuMNIST_dataset import KujuMNIST_DS

    trn_data = np.load(os.path.join(datasetdir, 'kmnist-train-imgs.npz'))
    trn_data = trn_data['arr_0'] / 255
    data_mean = trn_data.mean()
    data_std = trn_data.std()
    print(f'Mean: {data_mean}')
    print(f'Std: {data_std}')

    default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    transform_train = transforms.Compose(
        [
            transforms.ToPILImage(),
            #transforms.RandomAffine(degrees=7, translate=(0.1, 0.1), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize((data_mean,), (data_std,)),
        ]
    )

    transform_valid = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((data_mean,), (data_std,)),
        ]
    )

    trn_ds = KujuMNIST_DS(datasetdir, train_or_test='train', download=False, tfms=transform_train)
    val_ds = KujuMNIST_DS(datasetdir, train_or_test='test', download=False, tfms=transform_valid)

    trn_dl = DataLoader(trn_ds, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
    databunch = DataBunch(path=datasetdir, train_dl=trn_dl, valid_dl=val_dl, device=default_device)

    return databunch
