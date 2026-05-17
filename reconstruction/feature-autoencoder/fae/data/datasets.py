from functools import partial
from glob import glob
import os
import sys
from typing import List, Tuple, Sequence

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed
import SimpleITK as sitk
from scipy import ndimage
from torch.utils import data
import json
import time
from PIL import Image
from torchvision import transforms
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


# from fae import (
#     CAMCANROOT,
#     BRATSROOT,
#     MOODROOT,
# )
# from fae.data.data_utils import (
#     load_files_to_ram,
#     load_nii_nn,
#     load_segmentation,
# )
# from fae.data.artificial_anomalies import create_artificial_anomalies
#
#
# def get_camcan_files(path: str = CAMCANROOT, sequence: str = "t1") -> List[str]:
#     """Get all CamCAN files in a given sequence (t1, or t2).
#     Args:
#         path (str): Path to CamCAN root directory
#         sequence (str): One of "t1", or "t2"
#     Returns:
#         files (List[str]): List of files
#     """
#     files = glob(os.path.join(
#         path, f'normal/*/*{sequence.upper()}w_stripped_registered.nii.gz'))
#     assert len(files) > 0, "No files found in CamCAN"
#     return files
#
#
# def get_brats_files(path: str = BRATSROOT, sequence: str = "t1") \
#         -> Tuple[List[str], List[str]]:
#     """Get all BRATS files in a given sequence (t1, t2, or flair).
#     Args:
#         path (str): Path to BRATS root directory
#         sequence (str): One of "t1", "t2", or "flair"
#     Returns:
#         files (List[str]): List of files
#         seg_files (List[str]): List of segmentation files
#     """
#     files = glob(os.path.join(path, 'MICCAI_BraTS2020_TrainingData/*',
#                               f'*{sequence.lower()}*registered.nii.gz'))
#     seg_files = [os.path.join(os.path.dirname(
#         f), 'anomaly_segmentation.nii.gz') for f in files]
#     assert len(files) > 0, "No files found in BraTS"
#     return files, seg_files
#
#
# def get_mood_train_files(path: str = MOODROOT, **kwargs) \
#         -> Tuple[List[str], List[str]]:
#     """Get MOOD training files.
#     Args:
#         path (str): Path to MOOD root directory
#     Returns:
#         files (List[str]): List of files
#     """
#     train_files = glob(os.path.join(path, 'brain/train/*.nii.gz'))
#     # train_files = glob(os.path.join(path, 'brain/test_raw/*.nii.gz'))
#     # train_files.extend(glob(os.path.join(path, 'brain/train/*.nii.gz')))
#     # train_files = train_files[:round(len(train_files) * 0.8)]
#     assert len(train_files) > 0, "No files found in MOOD"
#     return train_files
#
#
# def get_mood_val_test_files(path: str = MOODROOT, **kwargs) \
#         -> Tuple[List[str], List[str]]:
#     """Get MOOD validation and test files.
#     Args:
#         sequence (str): One of "t1", or "t2"
#         path (str): Path to MOOD root directory
#     Returns:
#         files (List[str]): List of files
#     """
#     test_files = glob(os.path.join(path, 'brain/test_raw/*.nii.gz'))
#     # test_files = glob(os.path.join(path, 'brain/test_raw/*.nii.gz'))
#     # test_files.extend(glob(os.path.join(path, 'brain/train/*.nii.gz')))
#     # test_files = test_files[round(len(test_files) * 0.8):]
#     assert len(test_files) > 0, "No files found in MOOD"
#     return test_files, None
#
#
# def load_images(files: List[str], config) -> np.ndarray:
#     """Load images from a list of files.
#     Args:
#         files (List[str]): List of files
#         config (Namespace): Configuration
#     Returns:
#         images (np.ndarray): Numpy array of images
#     """
#     load_fn = partial(load_nii_nn,
#                       slice_range=config.slice_range,
#                       size=config.image_size,
#                       normalize=config.normalize,
#                       equalize_histogram=config.equalize_histogram)
#     return load_files_to_ram(files, load_fn)
#
#
# def load_segmentations(seg_files: List[str], config) -> np.ndarray:
#     """Load segmentations from a list of files.
#     Args:
#         seg_files (List[str]): List of files
#         config (Namespace): Configuration
#     Returns:
#         segmentations (np.ndarray): Numpy array of segmentations
#     """
#     load_fn = partial(load_segmentation,
#                       slice_range=config.slice_range,
#                       size=config.image_size)
#     return load_files_to_ram(seg_files, load_fn)
#
#
# class TrainDataset(Dataset):
#     """
#     Training dataset. No anomalies, no segmentation maps.
#     """
#
#     def __init__(self, imgs: np.ndarray):
#         """
#         Args:
#             imgs (np.ndarray): Training slices
#         """
#         self.imgs = imgs
#
#     def __len__(self):
#         return len(self.imgs)
#
#     def __getitem__(self, idx):
#         return self.imgs[idx]
#
#
# class TestDataset(Dataset):
#     """
#     Test dataset. With real anomalies.
#     """
#
#     def __init__(self, imgs: np.ndarray, segs: np.ndarray):
#         super().__init__()
#         self.imgs = imgs
#         self.segs = segs
#
#     def __len__(self):
#         return len(self.imgs)
#
#     def __getitem__(self, idx):
#         img = self.imgs[idx]  # (c, h, w)
#         seg = self.segs[idx]  # (1, h, w)
#         label = np.where(seg.sum(axis=(1, 2)) > 0, 1, 0)  # (1,)
#         return img, seg, label
#
#
# def get_files(ds_name: str, sequence: str):
#     if f"get_{ds_name}_files" in sys.modules[__name__].__dict__:
#         get_files_fn = sys.modules[__name__].__dict__[
#             f"get_{ds_name}_files"]
#     else:
#         raise ValueError(f'Dataset {ds_name} not found')
#
#     return get_files_fn(sequence=sequence)
#
#
# def val_test_split(files: Sequence, val_size: float, shuffle: bool = True) \
#         -> Tuple[List, List]:
#     """Split a list of files into validation and test sets"""
#     if shuffle:
#         np.random.shuffle(files)
#
#     # Split
#     val_size = int(len(files) * val_size)
#     val_files = files[:val_size]
#     test_files = files[val_size:]
#
#     return val_files, test_files


# def get_dataloaders(config):
#     """Returns the train-, val- and testloader.
#     Args:
#         config (Namespace): Configuration
#     Returns:
#         train_loader (torch.utils.data.DataLoader): Training loader
#         test_loader (torch.utils.data.DataLoader): Test loader
#     """
#     train_files = get_files(config.train_dataset, config.sequence)
#     test_files, test_seg_files = get_files(
#         config.test_dataset, config.sequence)
#
#     print(f"Found {len(train_files)} training files")
#     print(f"Found {len(test_files)} test files")
#
#     print("Loading images...")
#     if not config.train:
#         train_imgs = np.random.randn(1000, 1, 128, 128)
#     else:
#         train_imgs = np.concatenate(load_images(train_files, config))
#     test_imgs = np.concatenate(load_images(test_files, config))
#
#     if "mood" in config.test_dataset:
#         print("Creating artificial anomalies...")
#         assert config.anomaly_name is not None
#         test_imgs, test_segs = create_artificial_anomalies(
#             test_imgs, config.anomaly_name, radius_range=config.anomaly_size)
#     else:
#         print("Loading segmentations...")
#         test_segs = np.concatenate(load_segmentations(test_seg_files, config))
#
#     # Split into validation and test sets
#     val_size = int(len(test_imgs) * config.val_split)
#     val_imgs = test_imgs[:val_size]
#     test_imgs = test_imgs[val_size:]
#     val_segs = test_segs[:val_size]
#     test_segs = test_segs[val_size:]
#
#     # Shuffle validation and test data
#     val_perm = np.random.permutation(len(val_imgs))
#     val_imgs = val_imgs[val_perm]
#     val_segs = val_segs[val_perm]
#
#     test_perm = np.random.permutation(len(test_imgs))
#     test_imgs = test_imgs[test_perm]
#     test_segs = test_segs[test_perm]
#
#     # Create dataloaders
#     train_loader = DataLoader(TrainDataset(train_imgs),
#                               batch_size=config.batch_size,
#                               shuffle=True,
#                               num_workers=config.num_workers)
#     val_loader = DataLoader(TestDataset(val_imgs, val_segs),
#                             batch_size=config.batch_size,
#                             shuffle=False,
#                             num_workers=config.num_workers)
#     test_loader = DataLoader(TestDataset(test_imgs, test_segs),
#                              batch_size=config.batch_size,
#                              shuffle=False,
#                              num_workers=config.num_workers)
#     print(f"Len train_loader: {len(train_loader)}")
#     print(f"Len val_loader: {len(val_loader)}")
#     print(f"Len test_loader: {len(test_loader)}")
#
#     return train_loader, val_loader, test_loader


def get_dataloaders(config):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train, std=std_train)
    ])
    data_path = get_data_path(dataset=config.train_dataset)

    img_size = config.image_size
    if config.train_dataset in ['rsna', 'vin', 'brain', 'lag']:
        train_data = MedAD(main_path=data_path, img_size=img_size, transform=data_transform, mode='train')
        test_data = MedAD(main_path=data_path, img_size=img_size, transform=data_transform, mode='test')
    elif config.train_dataset == 'brats':
        train_data = BraTSAD(main_path=data_path, img_size=img_size,
                             transform=data_transform, mode='train')
        test_data = BraTSAD(main_path=data_path, img_size=img_size,
                            transform=data_transform, mode='test')
    elif config.train_dataset == 'c16':
        train_data = Camelyon16AD(main_path=data_path, img_size=img_size,
                                  transform=data_transform, mode='train', n_channel=3)
        test_data = Camelyon16AD(main_path=data_path, img_size=img_size,
                                 transform=data_transform, mode='test', n_channel=3)
    elif config.train_dataset == "isic":
        train_data = ISIC2018(main_path=data_path, img_size=img_size, transform=data_transform, mode='train',
                              n_channel=3)
        test_data = ISIC2018(main_path=data_path, img_size=img_size, transform=data_transform, mode='test', n_channel=3)
    else:
        raise Exception("error")

    # Create dataloaders
    train_loader = DataLoader(train_data,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers)
    val_loader = DataLoader(test_data,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1)
    test_loader = DataLoader(test_data,
                             batch_size=1,
                             shuffle=False,
                             num_workers=1)
    print(f"Len train_loader: {len(train_loader)}")
    print(f"Len val_loader: {len(val_loader)}")
    print(f"Len test_loader: {len(test_loader)}")

    return train_loader, val_loader, test_loader

def get_dataloaders_aucp(config):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train, std=std_train)
    ])
    data_path = get_data_path(dataset=config.train_dataset)

    img_size = config.image_size
    if config.train_dataset in ['rsna', 'vin', 'brain', 'lag']:
        train_data = MedAD_aucp(main_path=data_path, img_size=img_size, transform=data_transform, mode='train')
        test_data = MedAD_aucp(main_path=data_path, img_size=img_size, transform=data_transform, mode='test')
    elif config.train_dataset == 'brats':
        train_data = BraTSAD_aucp(main_path=data_path, img_size=img_size,
                             transform=data_transform, mode='train')
        test_data = BraTSAD_aucp(main_path=data_path, img_size=img_size,
                            transform=data_transform, mode='test')
    elif config.train_dataset == 'c16':
        train_data = Camelyon16AD_aucp(main_path=data_path, img_size=img_size,
                                  transform=data_transform, mode='train', n_channel=3)
        test_data = Camelyon16AD_aucp(main_path=data_path, img_size=img_size,
                                 transform=data_transform, mode='test', n_channel=3)
    elif config.train_dataset == "isic":
        train_data = ISIC2018_aucp(main_path=data_path, img_size=img_size, transform=data_transform, mode='train',
                              n_channel=3)
        test_data = ISIC2018_aucp(main_path=data_path, img_size=img_size, transform=data_transform, mode='test', n_channel=3)
    else:
        raise Exception("error")

    # Create dataloaders
    train_loader = DataLoader(train_data,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers)
    val_loader = DataLoader(test_data,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1)
    test_loader = DataLoader(test_data,
                             batch_size=1,
                             shuffle=False,
                             num_workers=1)
    print(f"Len train_loader: {len(train_loader)}")
    print(f"Len val_loader: {len(val_loader)}")
    print(f"Len test_loader: {len(test_loader)}")

    return train_loader, val_loader, test_loader


def get_data_path(dataset):
    # Data root is configurable via AUCP_DATA_ROOT (see aucp/paths.py).
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               os.pardir, os.pardir, os.pardir, os.pardir))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from aucp.paths import data_root, dataset_path

    if dataset in ('rsna', 'vin', 'brain', 'lag', 'brats', 'c16', 'isic'):
        return str(dataset_path(dataset))
    if dataset == 'oct':
        return os.path.join(os.path.expanduser("~"), "datasets", "OCT2017")
    if dataset == 'colon':
        return os.path.join(os.path.expanduser("~"), "datasets", "Colon_AD_public")
    raise Exception("Invalid dataset: {}".format(dataset))


def parallel_load(img_dir, img_list, img_size, n_channel=3, resample="bilinear", verbose=0):
    mode = "L" if n_channel == 1 else "RGB"
    if resample == "bilinear":
        resample = Image.BILINEAR
    elif resample == "nearest":
        resample = Image.NEAREST
    else:
        raise Exception
    return Parallel(n_jobs=-1, verbose=verbose)(delayed(
        lambda file: Image.open(os.path.join(img_dir, file)).convert(mode).resize(
            (img_size, img_size), resample=resample))(file) for file in img_list)


class MedAD(data.Dataset):
    def __init__(self, main_path, img_size, transform=None, mode="train"):
        super(MedAD, self).__init__()
        assert mode in ["train", "test"]
        self.root = main_path
        self.labels = []
        self.img_id = []
        self.mode = mode
        self.slices = []
        self.transform = transform if transform is not None else lambda x: x

        with open(os.path.join(main_path, "data.json")) as f:
            data_dict = json.load(f)

        print("Loading images")
        if mode == "train":
            train_normal = data_dict["train"]["0"]

            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "images"), train_normal, img_size)
            self.labels += len(train_normal) * [0]
            self.img_id += [img_name.split('.')[0] for img_name in train_normal]
            print("Loaded {} normal images, {:.3f}s".format(len(train_normal), time.time() - t0))

        else:  # test
            test_normal = data_dict["test"]["0"]
            test_abnormal = data_dict["test"]["1"]

            test_l = test_normal + test_abnormal
            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "images"), test_l, img_size)
            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_id += [img_name.split('.')[0] for img_name in test_l]
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        label = self.labels[index]
        img = self.transform(img)
        img_id = self.img_id[index]

        return {'img': img, 'label': label, 'name': img_id}

    def __len__(self):
        return len(self.slices)
    
class MedAD_aucp(data.Dataset):
    def __init__(self, main_path, img_size, transform=None, mode="train"):
        super(MedAD_aucp, self).__init__()
        assert mode in ["train", "test"]
        self.root = main_path
        self.labels = []
        self.img_id = []
        self.mode = mode
        self.slices = []
        self.transform = transform if transform is not None else lambda x: x

        with open(os.path.join(main_path, "data.json")) as f:
            data_dict = json.load(f)

        print("Loading images")
        if mode == "train":
            train_normal = data_dict["train"]["0"]

            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "images"), train_normal, img_size)
            self.labels += len(train_normal) * [0]
            self.img_id += [img_name.split('.')[0] for img_name in train_normal]
            print("Loaded {} normal images, {:.3f}s".format(len(train_normal), time.time() - t0))

        else:  # test
            test_normal = data_dict["test"]["0"]
            test_abnormal = data_dict["test"]["1"]

            train_normal = data_dict["train"]["0"]

            test_l = test_normal + test_abnormal
            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "images"), test_l, img_size)
            self.labels += len(test_normal) * [1] + len(test_abnormal) * [1]
            self.img_id += [img_name.split('.')[0] for img_name in test_l]

            self.slices += parallel_load(os.path.join(self.root, "images"), train_normal, img_size)
            self.labels += len(train_normal) * [0]
            self.img_id += [img_name.split('.')[0] for img_name in train_normal]

            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(train_normal), len(test_l), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        label = self.labels[index]
        img = self.transform(img)
        img_id = self.img_id[index]

        return {'img': img, 'label': label, 'name': img_id}

    def __len__(self):
        return len(self.slices)


class BraTSAD(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None, mode="train"):
        super(BraTSAD, self).__init__()
        assert mode in ["train", "test"]

        self.mode = mode
        self.root = main_path
        self.res = img_size
        self.labels = []
        self.masks = []
        self.img_ids = []
        self.slices = []
        self.transform = transform

        print("Loading images")
        if mode == "train":
            data_dir = os.path.join(self.root, "train")
            train_normal = os.listdir(data_dir)

            t0 = time.time()
            self.slices += parallel_load(data_dir, train_normal, img_size)
            self.labels += [0] * len(train_normal)
            self.img_ids += [img_name.split('.')[0] for img_name in train_normal]
            print("Loaded {} normal images, {:.3f}s".format(len(train_normal), time.time() - t0))

        else:  # test
            test_normal_dir = os.path.join(self.root, "test", "normal")
            test_abnormal_dir = os.path.join(self.root, "test", "tumor")
            test_mask_dir = os.path.join(self.root, "test", "annotation")

            test_normal = os.listdir(test_normal_dir)
            test_abnormal = os.listdir(test_abnormal_dir)
            test_masks = [e.replace("flair", "seg") for e in test_abnormal]

            test_l = test_normal + test_abnormal
            t0 = time.time()
            self.slices += parallel_load(test_normal_dir, test_normal, img_size)
            self.slices += parallel_load(test_abnormal_dir, test_abnormal, img_size)

            self.masks += len(test_normal) * [np.zeros((img_size, img_size))]
            self.masks += parallel_load(test_mask_dir, test_masks, img_size, resample="nearest", n_channel=1)  # 0/255

            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_ids += [img_name.split('.')[0] for img_name in test_l]
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        img = self.transform(img)

        label = self.labels[index]
        img_id = self.img_ids[index]

        if self.mode == "train":
            return {'img': img, 'label': label, 'name': img_id}
        else:
            mask = np.array(self.masks[index])
            mask = (mask > 0).astype(np.uint8)
            return {'img': img, 'label': label, 'name': img_id, 'mask': mask}

    def __len__(self):
        return len(self.slices)
    
class BraTSAD_aucp(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None, mode="train"):
        super(BraTSAD_aucp, self).__init__()
        assert mode in ["train", "test"]

        self.mode = mode
        self.root = main_path
        self.res = img_size
        self.labels = []
        self.masks = []
        self.img_ids = []
        self.slices = []
        self.transform = transform

        print("Loading images")
        if mode == "train":
            data_dir = os.path.join(self.root, "train")
            train_normal = os.listdir(data_dir)

            t0 = time.time()
            self.slices += parallel_load(data_dir, train_normal, img_size)
            self.labels += [0] * len(train_normal)
            self.img_ids += [img_name.split('.')[0] for img_name in train_normal]
            print("Loaded {} normal images, {:.3f}s".format(len(train_normal), time.time() - t0))

        else:  # test
            test_normal_dir = os.path.join(self.root, "test", "normal")
            test_abnormal_dir = os.path.join(self.root, "test", "tumor")
            test_mask_dir = os.path.join(self.root, "test", "annotation")

            test_normal = os.listdir(test_normal_dir)
            test_abnormal = os.listdir(test_abnormal_dir)
            test_masks = [e.replace("flair", "seg") for e in test_abnormal]

            data_dir = os.path.join(self.root, "train")
            train_normal = os.listdir(data_dir)

            test_l = test_normal + test_abnormal
            t0 = time.time()
            self.slices += parallel_load(test_normal_dir, test_normal, img_size)
            self.slices += parallel_load(test_abnormal_dir, test_abnormal, img_size)

            self.masks += len(test_normal) * [np.zeros((img_size, img_size))]
            self.masks += parallel_load(test_mask_dir, test_masks, img_size, resample="nearest", n_channel=1)  # 0/255

            self.labels += len(test_normal) * [1] + len(test_abnormal) * [1]
            self.img_ids += [img_name.split('.')[0] for img_name in test_l]

            # Add masks for train_normal images (all zeros)
            self.slices += parallel_load(data_dir, train_normal, img_size)
            self.labels += [0] * len(train_normal)
            self.img_ids += [img_name.split('.')[0] for img_name in train_normal]
            self.masks += len(train_normal) * [np.zeros((img_size, img_size))]

            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(train_normal), len(test_l), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        img = self.transform(img)

        label = self.labels[index]
        img_id = self.img_ids[index]

        if self.mode == "train":
            return {'img': img, 'label': label, 'name': img_id}
        else:
            mask = np.array(self.masks[index])
            mask = (mask > 0).astype(np.uint8)
            return {'img': img, 'label': label, 'name': img_id, 'mask': mask}

    def __len__(self):
        return len(self.slices)


class Camelyon16AD(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None, mode="train", n_channel=3, context_encoding=False):
        super(Camelyon16AD, self).__init__()
        assert mode in ["train", "test"]

        self.root = main_path
        self.res = img_size
        self.labels = []
        self.img_ids = []
        self.slices = []
        self.transform = transform
        if context_encoding:
            self.random_mask = transforms.RandomErasing(p=1., scale=(0.024, 0.024), ratio=(1., 1.), value=-1)
        else:
            self.random_mask = None

        print("Loading images")
        if mode == "train":
            data_dir = os.path.join(self.root, "train", "good")
            train_normal = os.listdir(data_dir)

            t0 = time.time()
            self.slices += parallel_load(data_dir, train_normal, img_size, n_channel=n_channel)
            self.labels += [0] * len(train_normal)
            self.img_ids += [img_name.split('.')[0] for img_name in train_normal]
            print("Loaded {} normal images, {:.3f}s".format(len(train_normal), time.time() - t0))

        else:  # test
            test_normal_dir = os.path.join(self.root, "test", "good")
            test_abnormal_dir = os.path.join(self.root, "test", "Ungood")

            test_normal = os.listdir(test_normal_dir)
            test_abnormal = os.listdir(test_abnormal_dir)

            test_l = test_normal + test_abnormal
            t0 = time.time()
            self.slices += parallel_load(test_normal_dir, test_normal, img_size, n_channel=n_channel)
            self.slices += parallel_load(test_abnormal_dir, test_abnormal, img_size, n_channel=n_channel)

            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_ids += [img_name.split('.')[0] for img_name in test_l]
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        img = self.transform(img)

        label = self.labels[index]
        img_id = self.img_ids[index]

        if self.random_mask is not None:
            img_masked = self.random_mask(img)
            return {'img': img, 'label': label, 'name': img_id, 'img_masked': img_masked}
        else:
            return {'img': img, 'label': label, 'name': img_id}

    def __len__(self):
        return len(self.slices)
    

class Camelyon16AD_aucp(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None, mode="train", n_channel=3, context_encoding=False):
        super(Camelyon16AD_aucp, self).__init__()
        assert mode in ["train", "test"]

        self.root = main_path
        self.res = img_size
        self.labels = []
        self.img_ids = []
        self.slices = []
        self.transform = transform
        if context_encoding:
            self.random_mask = transforms.RandomErasing(p=1., scale=(0.024, 0.024), ratio=(1., 1.), value=-1)
        else:
            self.random_mask = None

        print("Loading images")
        if mode == "train":
            data_dir = os.path.join(self.root, "train", "good")
            train_normal = os.listdir(data_dir)

            t0 = time.time()
            self.slices += parallel_load(data_dir, train_normal, img_size, n_channel=n_channel)
            self.labels += [0] * len(train_normal)
            self.img_ids += [img_name.split('.')[0] for img_name in train_normal]
            print("Loaded {} normal images, {:.3f}s".format(len(train_normal), time.time() - t0))

        else:  # test
            test_normal_dir = os.path.join(self.root, "test", "good")
            test_abnormal_dir = os.path.join(self.root, "test", "Ungood")

            test_normal = os.listdir(test_normal_dir)
            test_abnormal = os.listdir(test_abnormal_dir)

            data_dir = os.path.join(self.root, "train", "good")
            train_normal = os.listdir(data_dir)

            test_l = test_normal + test_abnormal
            t0 = time.time()
            self.slices += parallel_load(test_normal_dir, test_normal, img_size, n_channel=n_channel)
            self.slices += parallel_load(test_abnormal_dir, test_abnormal, img_size, n_channel=n_channel)

            self.labels += len(test_normal) * [1] + len(test_abnormal) * [1]
            self.img_ids += [img_name.split('.')[0] for img_name in test_l]

            self.slices += parallel_load(data_dir, train_normal, img_size, n_channel=n_channel)
            self.labels += [0] * len(train_normal)
            self.img_ids += [img_name.split('.')[0] for img_name in train_normal]

            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(train_normal), len(test_l), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        img = self.transform(img)

        label = self.labels[index]
        img_id = self.img_ids[index]

        if self.random_mask is not None:
            img_masked = self.random_mask(img)
            return {'img': img, 'label': label, 'name': img_id, 'img_masked': img_masked}
        else:
            return {'img': img, 'label': label, 'name': img_id}

    def __len__(self):
        return len(self.slices)


class ISIC2018(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None, mode="train", n_channel=3, context_encoding=False):
        super(ISIC2018, self).__init__()
        self.root = main_path
        self.res = img_size
        self.labels = []
        self.img_ids = []
        self.slices = []
        self.transform = transform
        if context_encoding:
            self.random_mask = transforms.RandomErasing(p=1., scale=(0.024, 0.024), ratio=(1., 1.), value=-1)
        else:
            self.random_mask = None

        print("Loading images")
        if mode == 'train':
            data_dir = os.path.join(self.root, "ISIC2018_Task3_Training_Input")
            data_csv = pd.read_csv(os.path.join(self.root, "ISIC2018_Task3_Training_GroundTruth",
                                                           "ISIC2018_Task3_Training_GroundTruth.csv"))
            train_normal = list(data_csv[data_csv['NV'] == 1]['image'])
            train_normal = [e+".jpg" for e in train_normal]
            t0 = time.time()
            self.slices += parallel_load(data_dir, train_normal, img_size, n_channel=n_channel)
            self.labels += [0] * len(train_normal)
            self.img_ids += train_normal
            print("Loaded {} normal images, {:.3f}s".format(len(train_normal), time.time() - t0))
        else:  # test
            data_dir = os.path.join(self.root, "ISIC2018_Task3_Test_Input")
            data_csv = pd.read_csv(os.path.join(self.root, "ISIC2018_Task3_Test_GroundTruth",
                                                           "ISIC2018_Task3_Test_GroundTruth.csv"))
            test_normal = list(data_csv[data_csv['NV'] == 1]['image'])
            test_abnormal = list(data_csv[data_csv['NV'] == 0]['image'])
            test_normal = [e + ".jpg" for e in test_normal]
            test_abnormal = [e + ".jpg" for e in test_abnormal]

            t0 = time.time()
            self.slices += parallel_load(data_dir, test_normal, img_size, n_channel=n_channel)
            self.slices += parallel_load(data_dir, test_abnormal, img_size, n_channel=n_channel)
            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_ids += test_normal + test_abnormal
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        img = self.transform(img)

        label = self.labels[index]
        img_id = self.img_ids[index]

        if self.random_mask is not None:
            img_masked = self.random_mask(img)
            return {'img': img, 'label': label, 'name': img_id, 'img_masked': img_masked}
        else:
            return {'img': img, 'label': label, 'name': img_id}

    def __len__(self):
        return len(self.slices)
    
class ISIC2018_aucp(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None, mode="train", n_channel=3, context_encoding=False):
        super(ISIC2018_aucp, self).__init__()
        self.root = main_path
        self.res = img_size
        self.labels = []
        self.img_ids = []
        self.slices = []
        self.transform = transform
        if context_encoding:
            self.random_mask = transforms.RandomErasing(p=1., scale=(0.024, 0.024), ratio=(1., 1.), value=-1)
        else:
            self.random_mask = None

        print("Loading images")
        if mode == 'train':
            data_dir = os.path.join(self.root, "ISIC2018_Task3_Training_Input")
            data_csv = pd.read_csv(os.path.join(self.root, "ISIC2018_Task3_Training_GroundTruth",
                                                           "ISIC2018_Task3_Training_GroundTruth.csv"))
            train_normal = list(data_csv[data_csv['NV'] == 1]['image'])
            train_normal = [e+".jpg" for e in train_normal]
            t0 = time.time()
            self.slices += parallel_load(data_dir, train_normal, img_size, n_channel=n_channel)
            self.labels += [0] * len(train_normal)
            self.img_ids += train_normal
            print("Loaded {} normal images, {:.3f}s".format(len(train_normal), time.time() - t0))
        else:  # test
            test_data_dir = os.path.join(self.root, "ISIC2018_Task3_Test_Input")
            data_csv = pd.read_csv(os.path.join(self.root, "ISIC2018_Task3_Test_GroundTruth",
                                                           "ISIC2018_Task3_Test_GroundTruth.csv"))
            test_normal = list(data_csv[data_csv['NV'] == 1]['image'])
            test_abnormal = list(data_csv[data_csv['NV'] == 0]['image'])
            test_normal = [e + ".jpg" for e in test_normal]
            test_abnormal = [e + ".jpg" for e in test_abnormal]

            train_data_dir = os.path.join(self.root, "ISIC2018_Task3_Training_Input")
            data_csv = pd.read_csv(os.path.join(self.root, "ISIC2018_Task3_Training_GroundTruth",
                                                           "ISIC2018_Task3_Training_GroundTruth.csv"))
            train_normal = list(data_csv[data_csv['NV'] == 1]['image'])
            train_normal = [e+".jpg" for e in train_normal]

            t0 = time.time()
            self.slices += parallel_load(test_data_dir, test_normal, img_size, n_channel=n_channel)
            self.slices += parallel_load(test_data_dir, test_abnormal, img_size, n_channel=n_channel)
            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_ids += test_normal + test_abnormal

            self.slices += parallel_load(train_data_dir, train_normal, img_size, n_channel=n_channel)
            self.labels += [0] * len(train_normal)
            self.img_ids += train_normal

            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(train_normal), len(test_abnormal + test_normal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        img = self.transform(img)

        label = self.labels[index]
        img_id = self.img_ids[index]

        if self.random_mask is not None:
            img_masked = self.random_mask(img)
            return {'img': img, 'label': label, 'name': img_id, 'img_masked': img_masked}
        else:
            return {'img': img, 'label': label, 'name': img_id}

    def __len__(self):
        return len(self.slices)
