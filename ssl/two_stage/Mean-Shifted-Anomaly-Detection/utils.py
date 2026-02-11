import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
import torchvision.models as models
import torch.nn.functional as F
from PIL import ImageFilter
import random
from torchvision.transforms import InterpolationMode
from datasets import *
from sklearn.covariance import LedoitWolf

BICUBIC = InterpolationMode.BICUBIC


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


transform_color = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_resnet18 = transforms.Compose([
    transforms.Resize(224, interpolation=BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


moco_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class Transform:
    def __init__(self):
        self.moco_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __call__(self, x):
        x_1 = self.moco_transform(x)
        x_2 = self.moco_transform(x)
        return x_1, x_2


class Model(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        if backbone == 152:
            self.backbone = models.resnet152(pretrained=True)
        else:
            self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = torch.nn.Identity()
        freeze_parameters(self.backbone, backbone, train_fc=False)

    def forward(self, x):
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n


def freeze_parameters(model, backbone, train_fc=False):
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False
    if backbone == 152:
        for p in model.conv1.parameters():
            p.requires_grad = False
        for p in model.bn1.parameters():
            p.requires_grad = False
        for p in model.layer1.parameters():
            p.requires_grad = False
        for p in model.layer2.parameters():
            p.requires_grad = False


def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)


# Personal codes
def gde_score(train_set, test_set):
    train_set = torch.tensor(train_set)
    test_set = torch.tensor(test_set)

    train_set = torch.nn.functional.normalize(train_set, p=2, dim=1)
    test_set = torch.nn.functional.normalize(test_set, p=2, dim=1)

    mean = torch.mean(train_set, dim=0)
    inv_cov = torch.Tensor(LedoitWolf().fit(train_set).precision_, device="cpu")

    distances = mahalanobis_distance(test_set, mean, inv_cov)
    return distances


def mahalanobis_distance(values, mean, inv_covariance):
    assert values.dim() == 2
    assert 1 <= mean.dim() <= 2
    assert len(inv_covariance.shape) == 2
    assert values.shape[1] == mean.shape[-1]
    assert mean.shape[-1] == inv_covariance.shape[0]
    assert inv_covariance.shape[0] == inv_covariance.shape[1]

    if mean.dim() == 1:  # Distribution mean.
        mean = mean.unsqueeze(0)
    x_mu = values - mean  # batch x features
    # Same as dist = x_mu.t() * inv_covariance * x_mu batch wise
    dist = torch.einsum("im,mn,in->i", x_mu, inv_covariance, x_mu)
    return dist.sqrt()


def get_loaders(dataset, label_class, batch_size, backbone):
    transform = transform_color if backbone == 152 else transform_resnet18
    # transform = transform_color
    if dataset == "cifar10":
        ds = torchvision.datasets.CIFAR10
        # transform = transform_color if backbone == 152 else transform_resnet18
        coarse = {}
        trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
        testset = ds(root='data', train=False, download=True, transform=transform, **coarse)
        trainset_1 = ds(root='data', train=True, download=True, transform=Transform(), **coarse)
        idx = np.array(trainset.targets) == label_class
        testset.targets = [int(t != label_class) for t in testset.targets]
        trainset.data = trainset.data[idx]
        trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        trainset_1.data = trainset_1.data[idx]
        trainset_1.targets = [trainset_1.targets[i] for i, flag in enumerate(idx, 0) if flag]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                                                   drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,
                                                  drop_last=False)
        return train_loader, test_loader, torch.utils.data.DataLoader(trainset_1, batch_size=batch_size,
                                                                      shuffle=True, num_workers=2, drop_last=False)
    else:
        data_path = get_data_path(dataset=dataset)
        img_size = 256
        if dataset in ['rsna', 'vin', 'brain', 'lag']:
            train_data = MedAD(main_path=data_path, img_size=img_size, transform=transform, mode='train')
            test_data = MedAD(main_path=data_path, img_size=img_size, transform=transform, mode='test')

            train_data_1 = MedAD(main_path=data_path, img_size=img_size, transform=Transform(), mode='train')
        elif dataset == 'brats':
            train_data = BraTSAD(main_path=data_path, img_size=img_size,
                                 transform=transform, mode='train')
            test_data = BraTSAD(main_path=data_path, img_size=img_size,
                                transform=transform, mode='test')
            train_data_1 = BraTSAD(main_path=data_path, img_size=img_size, transform=Transform(), mode='train')
        elif dataset == 'c16':
            train_data = Camelyon16AD(main_path=data_path, img_size=img_size,
                                      transform=transform, mode='train', n_channel=3)
            test_data = Camelyon16AD(main_path=data_path, img_size=img_size,
                                     transform=transform, mode='test', n_channel=3)
            train_data_1 = Camelyon16AD(main_path=data_path, img_size=img_size, transform=Transform(), mode='train',
                                        n_channel=3)
        elif dataset == "isic":
            train_data = ISIC2018(main_path=data_path, img_size=img_size, transform=transform, mode='train',
                                  n_channel=3)
            test_data = ISIC2018(main_path=data_path, img_size=img_size, transform=transform, mode='test',
                                 n_channel=3)
            train_data_1 = ISIC2018(main_path=data_path, img_size=img_size, transform=Transform(), mode='train')
        else:
            raise Exception("error")
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2,
                                                   drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2,
                                                  drop_last=False)
        train_loader_1 = torch.utils.data.DataLoader(train_data_1, batch_size=batch_size,
                                                     shuffle=True, num_workers=2, drop_last=False)

        return train_loader, test_loader, train_loader_1
