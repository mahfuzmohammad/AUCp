import os
import sys
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from torch.utils import data
import json
import time
from PIL import Image
from torchvision import transforms


def get_data_path(dataset):
    # Data root is configurable via AUCP_DATA_ROOT (see aucp/paths.py).
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              os.pardir, os.pardir, os.pardir))
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
        self.targets = []
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
            self.targets += len(train_normal) * [0]
            self.img_id += [img_name.split('.')[0] for img_name in train_normal]
            print("Loaded {} normal images, {:.3f}s".format(len(train_normal), time.time() - t0))

        else:  # test
            test_normal = data_dict["test"]["0"]
            test_abnormal = data_dict["test"]["1"]

            test_l = test_normal + test_abnormal
            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "images"), test_l, img_size)
            self.targets += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_id += [img_name.split('.')[0] for img_name in test_l]
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        label = self.targets[index]
        img = self.transform(img)
        img_id = self.img_id[index]

        # return {'img': img, 'label': label, 'name': img_id}
        return img, label

    def __len__(self):
        return len(self.slices)


class BraTSAD(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None, mode="train"):
        super(BraTSAD, self).__init__()
        assert mode in ["train", "test"]

        self.mode = mode
        self.root = main_path
        self.res = img_size
        self.targets = []
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
            self.targets += [0] * len(train_normal)
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

            self.targets += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_ids += [img_name.split('.')[0] for img_name in test_l]
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        img = self.transform(img)

        label = self.targets[index]
        img_id = self.img_ids[index]

        return img, label
        # if self.mode == "train":
        #     return {'img': img, 'label': label, 'name': img_id}
        # else:
        #     mask = np.array(self.masks[index])
        #     mask = (mask > 0).astype(np.uint8)
        #     return {'img': img, 'label': label, 'name': img_id, 'mask': mask}

    def __len__(self):
        return len(self.slices)


class Camelyon16AD(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None, mode="train", n_channel=3):
        super(Camelyon16AD, self).__init__()
        assert mode in ["train", "test"]

        self.root = main_path
        self.res = img_size
        self.targets = []
        self.img_ids = []
        self.slices = []
        self.transform = transform

        print("Loading images")
        if mode == "train":
            data_dir = os.path.join(self.root, "train", "good")
            train_normal = os.listdir(data_dir)

            t0 = time.time()
            self.slices += parallel_load(data_dir, train_normal, img_size, n_channel=n_channel)
            self.targets += [0] * len(train_normal)
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

            self.targets += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_ids += [img_name.split('.')[0] for img_name in test_l]
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        img = self.transform(img)

        label = self.targets[index]
        img_id = self.img_ids[index]

        # return {'img': img, 'label': label, 'name': img_id}
        return img, label

    def __len__(self):
        return len(self.slices)


class ISIC2018(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None, mode="train", n_channel=3):
        super(ISIC2018, self).__init__()
        self.root = main_path
        self.res = img_size
        self.targets = []
        self.img_ids = []
        self.slices = []
        self.transform = transform

        print("Loading images")
        if mode == 'train':
            data_dir = os.path.join(self.root, "ISIC2018_Task3_Training_Input")
            data_csv = pd.read_csv(os.path.join(self.root, "ISIC2018_Task3_Training_GroundTruth",
                                                           "ISIC2018_Task3_Training_GroundTruth.csv"))
            train_normal = list(data_csv[data_csv['NV'] == 1]['image'])
            train_normal = [e+".jpg" for e in train_normal]
            t0 = time.time()
            self.slices += parallel_load(data_dir, train_normal, img_size, n_channel=n_channel)
            self.targets += [0] * len(train_normal)
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
            self.targets += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_ids += test_normal + test_abnormal
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        img = self.transform(img)

        label = self.targets[index]
        img_id = self.img_ids[index]
        # return {'img': img, 'label': label, 'name': img_id}
        return img, label

    def __len__(self):
        return len(self.slices)
