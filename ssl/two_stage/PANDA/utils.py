import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
import ResNet
from datasets import *
from sklearn.covariance import LedoitWolf


mvtype = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
          'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
          'wood', 'zipper']

transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_gray = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_resnet_model(resnet_type=152):
    """
    A function that returns the required pre-trained resnet model
    :param resnet_number: the resnet type
    :return: the pre-trained model
    """
    if resnet_type == 18:
        return ResNet.resnet18(pretrained=True, progress=True)
    elif resnet_type == 50:
        return ResNet.wide_resnet50_2(pretrained=True, progress=True)
    elif resnet_type == 101:
        return ResNet.resnet101(pretrained=True, progress=True)
    else:  # 152
        return ResNet.resnet152(pretrained=True, progress=True)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return


def freeze_parameters(model, train_fc=False):
    for p in model.conv1.parameters():
        p.requires_grad = False
    for p in model.bn1.parameters():
        p.requires_grad = False
    for p in model.layer1.parameters():
        p.requires_grad = False
    for p in model.layer2.parameters():
        p.requires_grad = False
    if not train_fc:
        for p in model.fc.parameters():
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


def get_outliers_loader(batch_size):
    dataset = torchvision.datasets.ImageFolder(root='./data/tiny', transform=transform_color)
    outlier_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return outlier_loader


def get_loaders(dataset, label_class, batch_size):
    if dataset in ['cifar10', 'fashion']:
        if dataset == "cifar10":
            ds = torchvision.datasets.CIFAR10
            transform = transform_color
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)
        elif dataset == "fashion":
            ds = torchvision.datasets.FashionMNIST
            transform = transform_gray
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)

        idx = np.array(trainset.targets) == label_class
        testset.targets = [int(t != label_class) for t in testset.targets]
        trainset.data = trainset.data[idx]
        trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                                                   drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,
                                                  drop_last=False)
        return train_loader, test_loader
    else:
        data_path = get_data_path(dataset=dataset)
        img_size = 256
        if dataset in ['rsna', 'vin', 'brain', 'lag']:
            train_data = MedAD(main_path=data_path, img_size=img_size, transform=transform_color, mode='train')
            test_data = MedAD(main_path=data_path, img_size=img_size, transform=transform_color, mode='test')
        elif dataset == 'brats':
            train_data = BraTSAD(main_path=data_path, img_size=img_size,
                                 transform=transform_color, mode='train')
            test_data = BraTSAD(main_path=data_path, img_size=img_size,
                                transform=transform_color, mode='test')
        elif dataset == 'c16':
            train_data = Camelyon16AD(main_path=data_path, img_size=img_size,
                                      transform=transform_color, mode='train', n_channel=3)
            test_data = Camelyon16AD(main_path=data_path, img_size=img_size,
                                     transform=transform_color, mode='test', n_channel=3)
        elif dataset == "isic":
            train_data = ISIC2018(main_path=data_path, img_size=img_size, transform=transform_color, mode='train',
                                  n_channel=3)
            test_data = ISIC2018(main_path=data_path, img_size=img_size, transform=transform_color, mode='test',
                                 n_channel=3)
        else:
            raise Exception("error")
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2,
                                                   drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2,
                                                  drop_last=False)

        return train_loader, test_loader


def clip_gradient(optimizer, grad_clip):
    assert grad_clip > 0, 'gradient clip value must be greater than 1'
    for group in optimizer.param_groups:
        for param in group['params']:
            # gradient
            if param.grad is None:
                continue
            param.grad.data.clamp_(-grad_clip, grad_clip)
