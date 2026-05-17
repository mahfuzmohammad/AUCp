import os
import sys
from torchvision import transforms

# Make ``aucp`` importable when these scripts run from the reconstruction/ dir.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aucp.paths import data_root, dataset_path


def get_transform(opt):
    normalize = transforms.Normalize((0.5,), (0.5,)) if opt.model['in_c'] == 1 else \
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    transform = transforms.Compose([transforms.ToTensor(),
                                    normalize])
    return transform


def get_data_path(dataset):
    if dataset in ('rsna', 'vin', 'brain', 'lag', 'brats', 'c16', 'isic'):
        return str(dataset_path(dataset))
    # Datasets that live outside MedIAnomaly-Data keep their previous layout.
    if dataset == 'oct':
        return os.path.join(os.path.expanduser("~"), "datasets", "OCT2017")
    if dataset == 'colon':
        return os.path.join(os.path.expanduser("~"), "datasets", "Colon_AD_public")
    if dataset == 'cpchild':
        return os.path.join(str(data_root()), "CP-CHILD", "CP-CHILD-A")
    raise Exception("Invalid dataset: {}".format(dataset))
